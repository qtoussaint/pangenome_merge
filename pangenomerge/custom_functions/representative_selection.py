"""Representative-allele selection for the cross-graph "representative merge" step.

The default merge compares each node by a single representative (the longest protein
allele). This module enables richer representatives per node so that divergent or
length-variable alleles can still match across two split populations:

  - "modal":      the single most common member allele
  - "stratified": the most common allele within each length bin (bins opened when an
                  allele's length exceeds the current bin's shortest by >15%)
  - "allunique":  every distinct member allele

Allele sequences are sourced from each component graph's gene_data.csv (the authoritative
per-gene store; the GML protein/dna attributes are a pruned subset). A node's seqIDs are
exactly the clustering_ids in gene_data.csv, so frequencies and lengths are computable.
"""

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

# fraction by which an allele length must exceed a bin's shortest member to open a new bin
STRATIFIED_LEN_TOLERANCE = 0.15


def build_allele_lookup(graph_dirs, mode, graph_all_dir, seqtype):
    """Return {seqID: sequence} keyed to match the seqIDs carried on graph nodes.

    Parameters
    ----------
    graph_dirs : list[str]
        Component Panaroo directories, in merge order.
    mode : str
        'run' or 'test'. In run mode node seqIDs are suffixed _g{N}; in test mode they are
        remapped to the clustering_id of the combined ("all") run and not suffixed.
    graph_all_dir : str or None
        The --graph-all directory (required in test mode).
    seqtype : str
        'protein' -> prot_sequence, 'dna' -> dna_sequence.
    """
    col = "prot_sequence" if seqtype == "protein" else "dna_sequence"

    if mode == "test":
        # node seqIDs are remapped to the combined-run clustering_ids, no _g suffix
        gd = pd.read_csv(
            str(Path(graph_all_dir) / "gene_data.csv"),
            usecols=["clustering_id", col],
        )
        return dict(zip(gd["clustering_id"].astype(str), gd[col].astype(str)))

    # run mode: seqIDs are suffixed _g{i+1} for the i-th component graph
    lookup = {}
    for i, gdir in enumerate(graph_dirs):
        gd = pd.read_csv(str(Path(gdir) / "gene_data.csv"), usecols=["clustering_id", col])
        suffix = f"_g{i + 1}"
        for cid, seq in zip(gd["clustering_id"].astype(str), gd[col].astype(str)):
            lookup[f"{cid}{suffix}"] = seq
    return lookup


def _member_alleles(node_seqids, lookup):
    """List of member allele sequences for a node (with repetition), skipping unknown IDs."""
    seqs = []
    for sid in node_seqids:
        seq = lookup.get(str(sid))
        if seq:
            seqs.append(seq)
    return seqs


def _modal(seqs):
    """Most common sequence; ties broken by longest then lexicographic (deterministic)."""
    counts = Counter(seqs)
    return max(counts, key=lambda s: (counts[s], len(s), s))


def node_representatives(node_seqids, lookup, strategy):
    """Return a list of representative sequences for a node under the given strategy."""
    seqs = _member_alleles(node_seqids, lookup)
    if not seqs:
        return []

    if strategy == "allunique":
        # preserve deterministic order by length then sequence
        return sorted(set(seqs), key=lambda s: (len(s), s))

    if strategy == "modal":
        return [_modal(seqs)]

    if strategy == "stratified":
        counts = Counter(seqs)
        # greedy single pass over unique alleles sorted by length ascending
        uniq = sorted(counts, key=len)
        reps = []
        bin_alleles = []
        bin_anchor_len = None
        for s in uniq:
            if bin_anchor_len is None:
                bin_anchor_len = len(s)
                bin_alleles = [s]
            elif len(s) > bin_anchor_len * (1 + STRATIFIED_LEN_TOLERANCE):
                # close current bin, take its modal allele
                reps.append(max(bin_alleles, key=lambda a: (counts[a], len(a), a)))
                bin_anchor_len = len(s)
                bin_alleles = [s]
            else:
                bin_alleles.append(s)
        if bin_alleles:
            reps.append(max(bin_alleles, key=lambda a: (counts[a], len(a), a)))
        return reps

    raise ValueError(f"Unknown representative strategy: {strategy}")


def is_uncrossed(G, node, query_suffix):
    """Classify a node by whether it still belongs purely to one side of the merge.

    Members are tagged _g{N}. A node whose members are ALL from the newest query graph
    (query_suffix) is a pure query node; one with NONE from it is a pure base node; a node
    with some-but-not-all has already been merged across the two graphs.

    Returns "base", "query", or None (already cross-merged -> skip).
    """
    members = G.nodes[node]["members"]
    if not members:
        return None
    n_query = sum(1 for m in members if str(m).endswith(query_suffix))
    if n_query == 0:
        return "base"
    if n_query == len(members):
        return "query"
    return None


def write_representatives_fasta(G, lookup, base_strategy, query_strategy,
                                query_suffix, base_fa, query_fa):
    """Write representatives of not-yet-cross-merged nodes to two FASTAs (base, query).

    Pure-base nodes (base_strategy) go to base_fa, pure-query nodes (query_strategy) to
    query_fa. Each record is headed '>{node_name}||{i}' so hits can be mapped back to the
    originating node. Returns (n_base_nodes, n_query_nodes) actually written.
    """
    n_base = 0
    n_query = 0
    with open(base_fa, "w") as fb, open(query_fa, "w") as fq:
        for node in G.nodes():
            side = is_uncrossed(G, node, query_suffix)
            if side is None:
                continue
            strategy = base_strategy if side == "base" else query_strategy
            reps = node_representatives(G.nodes[node]["seqIDs"], lookup, strategy)
            if not reps:
                continue
            handle = fb if side == "base" else fq
            if side == "base":
                n_base += 1
            else:
                n_query += 1
            for i, seq in enumerate(reps):
                seq = seq.replace("*", "").rstrip()
                handle.write(f">{node}||{i}\n{seq}\n")
    logging.debug(f"Representative FASTA: {n_base} base nodes, {n_query} query nodes written.")
    return n_base, n_query
