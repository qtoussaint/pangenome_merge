import logging
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pangenomerge.panaroo_functions.cdhit import *
from pangenomerge.panaroo_functions.merge_nodes import *
from pangenomerge.custom_functions.run_mmseqs import mmseqs_createdb, run_mmseqs_search
from pangenomerge.custom_functions.context_similarity import build_ident_lookup, init_parallel, compute_scores_parallel
from pangenomerge.custom_functions.representative_selection import write_representatives_fasta


# add collapse families from Panaroo

def del_dups(seq):
    seen = set()
    pos = 0
    for item in seq:
        if item not in seen:
            seen.add(item)
            seq[pos] = item
            pos += 1
    del seq[pos:]
    return (seq)

def gen_node_iterables(G, nodes, feature, split=None):
    for n in nodes:
        if split is None:
            yield G.nodes[n][feature]
        else:
            yield G.nodes[n][feature].split(split)


def gen_edge_iterables(G, edges, feature):
    for e in edges:
        yield G[e[0]][e[1]][feature]


def iter_del_dups(iterable):
    seen = {}
    for f in itertools.chain.from_iterable(iterable):
        seen[f] = None
    return (list(seen.keys()))


def single_linkage(G, distances_bwtn_centroids, centroid_to_index, neighbours):
    index = []
    neigh_array = []
    for neigh in neighbours:
        for sid in G.nodes[neigh]['centroid']:
            index.append(centroid_to_index[sid])
            neigh_array.append(neigh)
    index = np.array(index, dtype=int)
    neigh_array = np.array(neigh_array)

    n_components, labels = connected_components(
        csgraph=distances_bwtn_centroids[index][:, index],
        directed=False,
        return_labels=True)
    # labels = labels[index]
    for neigh in neighbours:
        l = list(set(labels[neigh_array == neigh]))
        if len(l) > 1:
            for i in l[1:]:
                labels[labels == i] = l[0]

    clusters = [
        del_dups(list(neigh_array[labels == i])) for i in np.unique(labels)
    ]

    return (clusters)

def collapse_families(G,
                      seqid_to_centroid,
                      outdir,
                      family_threshold=0.7,
                      dna_error_threshold=0.99,
                      family_len_dif_percent=0,
                      correct_mistranslations=False,
                      length_outlier_support_proportion=0.01,
                      n_cpu=1,
                      quiet=False,
                      distances_bwtn_centroids=None,
                      centroid_to_index=None,
                      depths = [1, 2, 3],
                      search_genome_ids = None):

    #node_count = max(list(G.nodes())) + 10
    # above relies on integer nodes, mine are all strings
    # instead:
    if any(isinstance(x, int) for x in list(G.nodes())):
        print("WARNING: will overwrite existing nodes!")
    else:
        node_count = 0

    if correct_mistranslations:
        threshold = [0.99, 0.98, 0.95, 0.9]
    else:
        threshold = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

    # precluster for speed
    if correct_mistranslations:
        cdhit_clusters = iterative_cdhit(G,
                                         outdir,
                                         thresholds=threshold,
                                         s=family_len_dif_percent,
                                         n_cpu=n_cpu,
                                         quiet=True,
                                         dna=True,
                                         word_length=7,
                                         accurate=False)
        distances_bwtn_centroids, centroid_to_index = pwdist_edlib(
            G, cdhit_clusters, dna_error_threshold, dna=True, n_cpu=n_cpu)
    elif distances_bwtn_centroids is None:
        cdhit_clusters = iterative_cdhit(G,
                                         outdir,
                                         thresholds=threshold,
                                         s=family_len_dif_percent,
                                         n_cpu=n_cpu,
                                         quiet=True,
                                         dna=False)
        distances_bwtn_centroids, centroid_to_index = pwdist_edlib(
            G, cdhit_clusters, family_threshold, dna=False, n_cpu=n_cpu)

    # keep track of centroids for each sequence. Need this to resolve clashes
    seqid_to_index = {}
    for node in G.nodes():
        for sid in G.nodes[node]['seqIDs']:
            if "refound" in sid:
                seqid_to_index[sid] = centroid_to_index[G.nodes[node]
                                                        ["longCentroidID"][1]]
            else:
                seqid_to_index[sid] = centroid_to_index[seqid_to_centroid[sid]]

    nonzero_dist = distances_bwtn_centroids.nonzero()
    nonzero_dist = set([(i, j)
                        for i, j in zip(nonzero_dist[0], nonzero_dist[1])])

    node_mem_index = {}
    for n in G.nodes():
        node_mem_index[n] = defaultdict(set)
        for sid in G.nodes[n]['seqIDs']:
            node_mem_index[n][int(sid.split("_")[0])].add(seqid_to_index[sid])

    for depth in depths:
        if not quiet: print("Processing depth: ", depth)
        if search_genome_ids is None:
            search_space = set(G.nodes())
        else:
            search_space = set()
            search_genome_ids = intbitset(search_genome_ids)
            for n in G.nodes():
                if len(G.nodes[n]['members'].intersection(search_genome_ids))>0:
                    search_space.add(n)
            
        iteration_num = 1
        while len(search_space) > 0:
            # look for nodes to merge
            temp_node_list = list(search_space)
            removed_nodes = set()
            if not quiet: print("Iteration: ", iteration_num)
            iteration_num += 1
            for node in tqdm(temp_node_list, disable=quiet):
                if node in removed_nodes: continue

                if G.degree[node] <= 2:
                    search_space.remove(node)
                    removed_nodes.add(node)
                    continue

                # find neighbouring nodes and cluster their centroid with cdhit
                neighbours = [
                    v
                    for u, v in nx.bfs_edges(G, source=node, depth_limit=depth)
                ] + [node]

                # find clusters
                clusters = single_linkage(G, distances_bwtn_centroids,
                                          centroid_to_index, neighbours)

                for cluster in clusters:

                    # check if there are any to collapse
                    if len(cluster) <= 1: continue

                    # check for conflicts
                    seen = G.nodes[cluster[0]]['members'].copy()
                    noconflict = True
                    for n in cluster[1:]:
                        if not set(seen).isdisjoint(G.nodes[n]['members']):
                            noconflict = False
                            break
                        seen |= G.nodes[n]['members']

                    if noconflict:
                        # no conflicts so merge
                        node_count += 1
                        for neig in cluster:
                            removed_nodes.add(neig)
                            if neig in search_space: search_space.remove(neig)

                        G = merge_node_cluster(
                            G,
                            cluster,
                            node_count,
                            multi_centroid=(not correct_mistranslations))

                        node_mem_index[node_count] = node_mem_index[cluster[0]]
                        for n in cluster[1:]:
                            for m in node_mem_index[n]:
                                node_mem_index[node_count][
                                    m] |= node_mem_index[n][m]
                            node_mem_index[n].clear()
                            node_mem_index[n] = None

                        search_space.add(node_count)
                    else:
                        # merge if the centroids don't conflict and the nodes are adjacent in the conflicting genome
                        # this corresponds to a mistranslation/frame shift/premature stop where one gene has been split
                        # into two in a subset of genomes

                        # sort by size
                        cluster = sorted(cluster,
                                         key=lambda x: G.nodes[x]['size'],
                                         reverse=True)

                        node_mem_count = Counter(
                            itertools.chain.from_iterable(
                                gen_node_iterables(G, cluster, 'members')))
                        mem_count = np.array(list(node_mem_count.values()))
                        merge_same_members = True
                        if np.sum(mem_count == 1) / float(
                                len(mem_count
                                    )) < length_outlier_support_proportion:
                            # do not merge nodes that have the same members as this is likely to be a spurious long gene
                            merge_same_members = False

                        while len(cluster) > 0:
                            sub_clust = [cluster[0]]
                            nA = cluster[0]
                            for nB in cluster[1:]:
                                mem_inter = list(
                                    set(G.nodes[nA]['members']).intersection(
                                        set(G.nodes[nB]['members'])))
                                if len(mem_inter) > 0:
                                    if merge_same_members:
                                        shouldmerge = True
                                        if len(
                                                set(G.nodes[nA]['centroid']).
                                                intersection(
                                                    set(G.nodes[nB]
                                                        ['centroid']))) > 0:
                                            shouldmerge = False

                                        if shouldmerge:
                                            edge_mem_count = Counter()
                                            for e in itertools.chain.from_iterable(
                                                    gen_edge_iterables(
                                                        G, G.edges([nA, nB]),
                                                        'members')):
                                                edge_mem_count[e] += 1
                                                if edge_mem_count[e] > 3:
                                                    shouldmerge = False
                                                    break

                                        if shouldmerge:
                                            for imem in mem_inter:
                                                for sidA in node_mem_index[nA][
                                                        imem]:
                                                    for sidB in node_mem_index[
                                                            nB][imem]:
                                                        if ((
                                                                sidA, sidB
                                                        ) in nonzero_dist) or (
                                                            (sidB, sidA) in
                                                                nonzero_dist):
                                                            shouldmerge = False
                                                            break
                                                    if not shouldmerge: break
                                                if not shouldmerge: break

                                        if shouldmerge:
                                            sub_clust.append(nB)
                                else:
                                    sub_clust.append(nB)

                            if len(sub_clust) > 1:

                                clique_clusters = single_linkage(
                                    G, distances_bwtn_centroids,
                                    centroid_to_index, sub_clust)
                                for clust in clique_clusters:
                                    if len(clust) <= 1: continue
                                    node_count += 1
                                    for neig in clust:
                                        removed_nodes.add(neig)
                                        if neig in search_space:
                                            search_space.remove(neig)
                                    G = merge_node_cluster(
                                        G,
                                        clust,
                                        node_count,
                                        multi_centroid=(
                                            not correct_mistranslations),
                                        check_merge_mems=False)

                                    node_mem_index[
                                        node_count] = node_mem_index[clust[0]]
                                    for n in clust[1:]:
                                        for m in node_mem_index[n]:
                                            node_mem_index[node_count][
                                                m] |= node_mem_index[n][m]
                                        node_mem_index[n].clear()
                                        node_mem_index[n] = None

                                    search_space.add(node_count)

                            cluster = [
                                n for n in cluster if n not in sub_clust
                            ]

                if node in search_space:
                    search_space.remove(node)

    return G, distances_bwtn_centroids, centroid_to_index


def write_centroids_to_fasta(G, query_fa):
    with open(query_fa, "w") as ft:
        for node, data in G.nodes(data=True):
            name = node
            #if name.endswith("_target") or "_target" in name:
                # pre-existing nodes -- already in target db
            #    continue
            #else:
            # new nodes
            seqs = data["protein"]
            if isinstance(seqs, (list, tuple)):
                seqs = max(seqs, key=len) # if list, pick longest sequence
            if isinstance(seqs, str):
                parts = seqs.split(";") # if string split on semicolon and pick longest
                seqs = max(parts, key=len)
            seqs = seqs.rstrip('*') # remove trailing stop
            name = node
            ft.write(f">{name}\n{seqs}\n")


def filter_mmseqs_to_current_nodes(mmseqs_frame, G):
    current_nodes = set(G.nodes())
    mask = mmseqs_frame["query"].isin(current_nodes) & mmseqs_frame["target"].isin(current_nodes)
    return mmseqs_frame[mask].copy()


def context_pass(sims, context_threshold):
    """True if the depth-1/2/3 context similarities clear the gate.

    Two accepted patterns:
      - depth-1 match AND (depth-2 OR depth-3 match)            [original]
      - no depth-1 match but BOTH depth-2 AND depth-3 match     [added: deeper-only support]
    """
    return (
        (sims[0] >= context_threshold
         and (sims[1] >= context_threshold or sims[2] >= context_threshold))
        or (sims[1] >= context_threshold and sims[2] >= context_threshold)
    )


def find_mergeable_pairs(G, mmseqs_frame, ident_lookup, context_threshold, identity_threshold, threads, tag=""):
    init_parallel(G, ident_lookup, context_threshold)
    scores = compute_scores_parallel(mmseqs_frame, threads)

    # debug statement...
    logging.debug(f"scores: {scores[:5]}")

    # diagnostic funnel (only when tagged, e.g. the representative step) to see where candidates drop
    if tag:
        n_total = len(scores)
        n_ident = sum(1 for _, _, ident, _ in scores if ident >= identity_threshold)
        n_ctx = sum(1 for _, _, ident, sims in scores
                    if ident >= identity_threshold and context_pass(sims, context_threshold))
        n_disj = sum(1 for nA, nB, ident, sims in scores
                     if ident >= identity_threshold and context_pass(sims, context_threshold)
                     and set(G.nodes[nA]['members']).isdisjoint(set(G.nodes[nB]['members'])))
        logging.info(f"[{tag} funnel] candidates={n_total} pass_identity={n_ident} "
                     f"pass_identity+context={n_ctx} pass_+disjoint(accepted)={n_disj}")

    # sort dataframe by scores
    scores_sorted = sorted(
        scores,
        key=lambda x: (x[2], x[3][0], x[3][1], x[3][2]),
        reverse=True
    )

    # debug statement...
    logging.debug(f"scores_sorted: {scores_sorted[:5]}")

    # filter accepted pairs by identity + context thresholds
    accepted_pairs = []
    for nA, nB, ident, sims in scores_sorted:
        if (
            ident >= identity_threshold
            and context_pass(sims, context_threshold)
            and set(G.nodes[nA]['members']).isdisjoint(set(G.nodes[nB]['members'])) # check they do not share any members (genes within same genome will not be merged)
        ):
            accepted_pairs.append((nA, nB, ident, sims))

    # debug statement...
    logging.debug(f"accepted pairs (by context): {accepted_pairs[:10]}")

    # filter out any duplicates (in order, so best match kept)
    unique_pairs = []
    seen_nodes = set()
    for nA, nB, ident, sims in accepted_pairs:
        if nA not in seen_nodes and nB not in seen_nodes:
            unique_pairs.append((nA, nB, ident, sims))
            seen_nodes.add(nA)
            seen_nodes.add(nB)

    # debug statement...
    logging.debug(f"accepted pairs (duplicates removed): {accepted_pairs[:10]}")

    # reorder to ensure 'a' is always the node with '_target'
    reordered_pairs = []
    for a, b, ident, sims in unique_pairs:
        if "_target" in b and "_target" not in a:
            a, b = b, a
        reordered_pairs.append((a, b))

    # debug statement...
    logging.debug(f"accepted pairs (reordered): {reordered_pairs[:10]}")

    return reordered_pairs


def find_small_fragment_pairs(G, mmseqs_frame, min_identity, min_coverage, max_component_size):
    """Identity/coverage-only merge candidates restricted to small connected components.

    Final-pass cleanup for clusters that context-gated passes could not match because
    their components are too small to support a meaningful neighbourhood signal.
    Both endpoints of every returned pair lie in a connected component of size
    <= max_component_size at call time.
    """
    max_deg = max_component_size - 1

    # cheap filter first: identity + coverage thresholds, restricted to surviving nodes
    df = filter_mmseqs_to_current_nodes(mmseqs_frame, G)
    df = df[(df["fident"] >= min_identity) & (df["cov"] >= min_coverage)]
    if len(df) == 0:
        logging.debug("Small-fragment: no hits pass identity/coverage thresholds.")
        return []

    # restrict to nodes that (a) appear in surviving hits AND (b) have degree <= max_deg
    # (necessary condition for being in a component of size <= max_component_size)
    hit_nodes = set(df["query"]).union(df["target"])
    candidate_nodes = {n for n in hit_nodes if G.degree[n] <= max_deg}
    df = df[df["query"].isin(candidate_nodes) & df["target"].isin(candidate_nodes)]
    if len(df) == 0:
        logging.debug("Small-fragment: no surviving hits after degree filter.")
        return []

    # bounded BFS per candidate; aborts the moment the component exceeds max_component_size.
    # cached by component, so siblings skip the walk. never touches the global graph.
    status = {}  # node -> "small" or "big"
    for n in candidate_nodes:
        if n in status:
            continue
        seen = {n}
        frontier = [n]
        too_big = False
        while frontier and not too_big:
            nxt = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v in seen:
                        continue
                    seen.add(v)
                    if len(seen) > max_component_size:
                        too_big = True
                        break
                    nxt.append(v)
                if too_big:
                    break
            frontier = nxt
        label = "big" if too_big else "small"
        for u in seen:
            status[u] = label

    small_nodes = {node for node, s in status.items() if s == "small"}
    df = df[df["query"].isin(small_nodes) & df["target"].isin(small_nodes)]
    if len(df) == 0:
        logging.debug("Small-fragment: no hits where both endpoints are in small components.")
        return []

    # greedy pair selection: best identity/coverage first, disjoint members,
    # each node appears in at most one accepted pair
    pairs = []
    used = set()
    df_sorted = df.sort_values(["fident", "cov"], ascending=False)
    for q, t in df_sorted[["query", "target"]].itertuples(index=False, name=None):
        if q in used or t in used:
            continue
        if not set(G.nodes[q]["members"]).isdisjoint(set(G.nodes[t]["members"])):
            continue
        # match find_mergeable_pairs convention: '_target' node kept by merge_pair
        a, b = q, t
        if "_target" in b and "_target" not in a:
            a, b = b, a
        pairs.append((a, b))
        used.add(q)
        used.add(t)

    logging.debug(f"Small-fragment: {len(pairs)} accepted pairs.")
    return pairs


def find_representative_merge_pairs(merged_graph, allele_lookup, base_strategy,
                                    query_strategy, seqtype, query_suffix, ident_lookup,
                                    family_threshold, context_threshold, outdir, threads):
    """Representative-based cross-graph rescue step.

    Compares only nodes that have not yet merged across graphs: pure-base nodes (represented
    via base_strategy) against pure-query nodes (query_strategy). Each node may contribute
    several representative alleles (modal / stratified / all-unique). Candidate node pairs
    are gated through the same context check as the family step (find_mergeable_pairs),
    reusing the preexisting protein ident_lookup for neighbour-context scoring.
    """
    tmp = Path(outdir) / "mmseqs_tmp"
    base_fa = tmp / "rep_base.fa"
    query_fa = tmp / "rep_query.fa"
    n_base, n_query = write_representatives_fasta(
        merged_graph, allele_lookup, base_strategy, query_strategy,
        query_suffix, base_fa, query_fa)
    logging.debug(f"Representative step: {n_base} base nodes, {n_query} query nodes.")
    if n_base == 0 or n_query == 0:
        logging.debug("Representative step: one side has no eligible nodes; skipping.")
        return []

    base_db = tmp / "rep_base_db"
    query_db = tmp / "rep_query_db"
    resultm8 = tmp / "rep_clusters.m8"
    # The rescue step is optional: if its mmseqs search fails (e.g. the large multi-representative
    # nucleotide query exhausts the prefilter's memory), warn and skip rather than abort the merge.
    try:
        # nucleotide dbs must be uncompressed -- see mmseqs_createdb's `compressed` note
        db_compressed = 0 if seqtype == "dna" else 1
        mmseqs_createdb(fasta=base_fa, outdb=base_db, threads=threads, nt2aa=False,
                        compressed=db_compressed)
        mmseqs_createdb(fasta=query_fa, outdb=query_db, threads=threads, nt2aa=False,
                        compressed=db_compressed)
        run_mmseqs_search(
            targetdb=str(base_db), querydb=str(query_db),
            resultdb=str(tmp / "rep_resultdb"), resultm8=str(resultm8),
            tmpdir=str(tmp), threads=threads,
            fident=family_threshold,
            coverage=round(family_threshold * 0.95, 3),
            # nucleotide-vs-nucleotide search needs an explicit search type; protein auto-detects
            search_type=(3 if seqtype == "dna" else None),
            # bound nucleotide prefilter index memory (default heuristic over-allocates); also a bit
            # more sensitive for a rescue step. harmless for the (auto-k) protein path.
            kmer=(13 if seqtype == "dna" else None),
        )
    except subprocess.CalledProcessError as e:
        # CalledProcessError's own message carries only the command, so surface mmseqs' own
        # output too -- without it the failure is undiagnosable after the fact.
        logging.warning(f"Representative step: mmseqs search failed ({e}); skipping this round.")
        if e.stderr:
            logging.warning(f"Representative step: mmseqs stderr: {e.stderr.strip()[-4000:]}")
        if e.stdout:
            logging.warning(f"Representative step: mmseqs stdout: {e.stdout.strip()[-4000:]}")
        return []

    hits = pd.read_csv(resultm8, sep="\t")
    if len(hits) == 0:
        return []
    hits["fident"] = pd.to_numeric(hits["fident"], errors="coerce")
    # strip the ||i representative index to recover originating node names.
    # NB: use a literal (python) split — pandas str.split treats "||" as a regex (empty
    # alternation) and would shred every name to "".
    hits["query"] = hits["query"].map(lambda s: s.split("||", 1)[0])
    hits["target"] = hits["target"].map(lambda s: s.split("||", 1)[0])
    hits = hits[hits["query"] != hits["target"]].copy()
    if len(hits) == 0:
        return []
    # collapse multiple rep-vs-rep hits to one row per node pair, keeping best identity
    hits = hits.sort_values("fident", ascending=False).drop_duplicates(
        subset=["query", "target"], keep="first")

    # target endpoints are base nodes (already carry _target in the merged graph); queries
    # are pure-query nodes. find_mergeable_pairs reorders so the _target node is kept by merge.
    candidate_frame = hits[["query", "target", "fident"]].copy()

    return find_mergeable_pairs(
        merged_graph, candidate_frame, ident_lookup,
        context_threshold, family_threshold, threads, tag="rep",
    )


def collapse_spurious_paralogs(merged_graph, base_db, options, outdir,
                               family_threshold, context_threshold,
                               frameshift_identity, frameshift_coverage,
                               context_search_iterations=-1,
                               allele_lookup=None, rep_merge_base=None,
                               rep_merge_query=None, rep_merge_seqtype="protein",
                               query_suffix=None):

    # write query centroid fasta (stream to reduce memory)
    query_fa = Path(outdir) / "mmseqs_tmp" / "centroids_query.fa"
    write_centroids_to_fasta(merged_graph, query_fa)

    # info statement
    logging.info("Computing pairwise identities...")

    # info statement...
    logging.info("Creating MMSeqs2 database...")

    # create AA mmseqs database for query
    query_db = Path(outdir) / "mmseqs_tmp" / "query_db"
    mmseqs_createdb(fasta=query_fa, outdb=query_db, threads=options.threads, nt2aa=False)

    # info statement...
    logging.info("Running MMSeqs2...")

    # run mmseqs to get hits, keeping only those above the minimum useful threshold (family_threshold, which is LOWER than context threshold)
    run_mmseqs_search(
        targetdb=base_db,
        querydb=query_db,
        resultdb = str(Path(outdir) / "mmseqs_tmp" / "resultdb"),
        resultm8=str(Path(outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8"),
        tmpdir=str(Path(outdir) / "mmseqs_tmp"),
        threads=options.threads,
        fident=family_threshold,
        coverage=float(round(min(family_threshold * 0.95, frameshift_coverage), 3))
    )

    # info statement...
    logging.info("MMSeqs2 complete. Reading and filtering results...")

    # read mmseqs results
    mmseqs = pd.read_csv(Path(outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8", sep="\t")

    # debugging statements...
    logging.debug(f"Unfiltered: {len(mmseqs)} one-to-one hits.")
    logging.debug(f"{mmseqs}")

    # ensure numeric columns
    for col in ["fident", "evalue", "tlen", "qlen"]:
        mmseqs[col] = pd.to_numeric(mmseqs[col], errors="coerce")

    # define length difference
    max_len = np.maximum(mmseqs["tlen"], mmseqs["qlen"])
    mmseqs["len_dif"] = 1 - (np.abs(mmseqs["tlen"] - mmseqs["qlen"]) / max_len)

    # per-row bidirectional coverage (cov-mode 0 style): min of query-cov and target-cov
    mmseqs["cov"] = np.minimum(mmseqs["alnlen"] / mmseqs["qlen"], mmseqs["alnlen"] / mmseqs["tlen"])

    # remove self-matches (target == query) once, up front so both pass-1 and frameshift frames inherit it
    mmseqs = mmseqs[mmseqs["target"] != mmseqs["query"]].copy()

    # carve out frameshift (pass-2) hits BEFORE pass-1 filter narrows mmseqs:
    # higher identity, lower coverage, no len_dif requirement (frameshifts create length differences by design)
    mmseqs_frameshift = mmseqs[(mmseqs["fident"] >= frameshift_identity) & (mmseqs["cov"] >= frameshift_coverage)].copy()
    mmseqs_frameshift["target"] += "_target"

    # filter for identity ≥ family_threshold, length difference ≥ family_threshold*0.95, and coverage ≥ family_threshold*0.95
    # (cov clause re-imposes what MMSeqs -c used to enforce before we lowered the search floor for the frameshift pass)
    mmseqs = mmseqs[(mmseqs["fident"] >= family_threshold) & (mmseqs["len_dif"] >= family_threshold*0.95) & (mmseqs["cov"] >= family_threshold*0.95)].copy()

    # add _target to target node names
    # possibly pretty memory/time intensive for big dataframes, see if can make this more efficient later
    mmseqs["target"] += "_target"

    # debugging statements...
    logging.debug(f"mmseqs filtered: {len(mmseqs)} hits remaining")
    logging.debug(f"filtered mmseqs hits: {mmseqs.head()}")

    # info statement...
    logging.debug(f"Beginning context search...")

    ### compute contextual similarity

    # can still accidentally map together things from same genome by mapping a target node that's been merged into with a g2 node
    # thus we check that member sets for the nodes are disjoint (don't contain any of the same genomes)

    ident_lookup = build_ident_lookup(mmseqs)

    # build frameshift ident lookup ONCE — depends only on hit tables, not graph state
    if len(mmseqs_frameshift) > 0:
        ident_lookup_frameshift = build_ident_lookup(pd.concat([mmseqs, mmseqs_frameshift], ignore_index=True))
    else:
        ident_lookup_frameshift = None

    # info statement...
    logging.info("Merging nodes and edges...")

    ### strict alternating fixed-point: one family round, then one frameshift round, repeat.
    ###   exits when a full refinement iteration (identity+context + frameshift detection) produces 0 merges ⇒ graph is stable.
    ###   context_search_iterations caps refinement iterations (-1 = unlimited).
    # mmseqs_frameshift: persistently shrunk each refinement iteration as graph nodes decrease, avoiding redundant isin work.
    refinement_iteration = 0
    while context_search_iterations < 0 or refinement_iteration < context_search_iterations:
        refinement_iteration += 1
        any_merged = False

        # --- single family-style round ---
        mmseqs_cur = filter_mmseqs_to_current_nodes(mmseqs, merged_graph)
        if len(mmseqs_cur) == 0:
            logging.info(f"Refinement merge {refinement_iteration} identity+context step: no hits survive staleness filter.")
        else:
            pairs_iter = find_mergeable_pairs(
                merged_graph, mmseqs_cur, ident_lookup,
                context_threshold, family_threshold, options.threads
            )
            if len(pairs_iter) == 0:
                logging.info(f"Refinement merge {refinement_iteration} identity+context step: no new mergeable pairs.")
            else:
                logging.info(f"Refinement merge {refinement_iteration} identity+context step: merging {len(pairs_iter)} pairs.")
                apply_merges(merged_graph, pairs_iter)
                any_merged = True

        logging.debug(f"After refinement merge {refinement_iteration} identity+context step: {len(merged_graph.nodes())} nodes")

        # --- representative-based cross-graph rescue step (optional) ---
        # richer per-node representatives (modal / stratified / all-unique) to capture hits
        # the single-centroid family step missed; same context gating, preexisting ident_lookup
        if rep_merge_base is not None and rep_merge_query is not None and allele_lookup is not None:
            pairs_rep = find_representative_merge_pairs(
                merged_graph, allele_lookup, rep_merge_base, rep_merge_query,
                rep_merge_seqtype, query_suffix, ident_lookup,
                family_threshold, context_threshold, outdir, options.threads,
            )
            if len(pairs_rep) > 0:
                logging.info(f"Refinement merge {refinement_iteration} representative step: merging {len(pairs_rep)} pairs.")
                apply_merges(merged_graph, pairs_rep)
                any_merged = True
            else:
                logging.info(f"Refinement merge {refinement_iteration} representative step: no new mergeable pairs.")

        logging.debug(f"After refinement merge {refinement_iteration} representative step: {len(merged_graph.nodes())} nodes")

        # --- single frameshift round on full current graph ---
        if ident_lookup_frameshift is not None and len(mmseqs_frameshift) > 0:
            mmseqs_frameshift = filter_mmseqs_to_current_nodes(mmseqs_frameshift, merged_graph)
            logging.info(f"Refinement merge {refinement_iteration} frameshift detection step: {len(mmseqs_frameshift)} frameshift hits on current graph.")
            if len(mmseqs_frameshift) > 0:
                pairs_fs = find_mergeable_pairs(
                    merged_graph, mmseqs_frameshift, ident_lookup_frameshift,
                    context_threshold, frameshift_identity, options.threads
                )
                if len(pairs_fs) > 0:
                    logging.info(f"Refinement merge {refinement_iteration} frameshift detection step: merging {len(pairs_fs)} pairs.")
                    apply_merges(merged_graph, pairs_fs)
                    any_merged = True

        logging.debug(f"After refinement merge {refinement_iteration}: {len(merged_graph.nodes())} nodes")

        # Fixed point: neither pass produced any merges this round ⇒ graph is stable under both.
        if not any_merged:
            logging.info(f"Refinement merge: fixed point reached after {refinement_iteration} iteration(s).")
            break
    else:
        logging.info(f"Refinement merge: reached max iterations ({context_search_iterations}); stopping.")

    # --- final small-fragment pass: identity/coverage only, no context, both endpoints in components of size <= 3 ---
    if ident_lookup_frameshift is not None and len(mmseqs_frameshift) > 0:
        pairs_small = find_small_fragment_pairs(
            merged_graph, mmseqs_frameshift,
            min_identity=0.90, min_coverage=0.85, max_component_size=3,
        )
        if len(pairs_small) > 0:
            logging.info(f"Small-fragment pass: merging {len(pairs_small)} pairs.")
            apply_merges(merged_graph, pairs_small)
        else:
            logging.info("Small-fragment pass: no mergeable pairs.")

    # update degrees across graph (single pass, after both family + frameshift merges)
    for node in merged_graph:
        merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])

    # debug statement...
    logging.debug(f"After collapse: {len(merged_graph.nodes())} nodes")
