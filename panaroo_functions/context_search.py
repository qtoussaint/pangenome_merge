import networkx as nx
from panaroo_functions.cdhit import *

# add collapse families from Panaroo

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
                        if not seen.isdisjoint(G.nodes[n]['members']):
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
                                    G.nodes[nA]['members'].intersection(
                                        G.nodes[nB]['members']))
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
