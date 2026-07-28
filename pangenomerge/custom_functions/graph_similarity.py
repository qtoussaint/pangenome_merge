"""Pair-level clustering-similarity machinery for scoring a merge against a truth graph.

Three partitions of the same set of genes are involved throughout:

    C  the component clustering -- the union of the input panaroo graphs, each cluster made
       globally unique. This is pangenomerge's "do nothing" starting point.
    M  the merged clustering produced by pangenomerge.
    T  the truth clustering -- panaroo run over all the isolates at once.

pangenomerge only ever *coarsens* C: it groups whole component clusters together and never
splits one. Two consequences drive everything here.

1. A per-gene random labelling is the wrong null. Leaving C untouched already scores
   ARI 0.66 / AMI 0.93 against T, so standard scores compress every real difference into
   the fourth decimal. Both metric families below therefore put 0 at C, not at random.

2. Because merging only ever *adds* co-clustered pairs, any pair of genes that C groups
   but T separates is wrong forever. If C does not refine T -- and in practice it does not
   -- then a score of 1 against T is unreachable, and a metric anchored at T reports a
   number against an impossible target.

Hence two families:

    pcARI / pcNMI   post-clustering:  0 = C, 1 = T
    acARI / acNMI   attainability-corrected:  0 = C, 1 = C*, the best merge reachable from C

pcARI has a clean identity. Writing b for pairs wrongly co-clustered and c for pairs wrongly
separated, RI = 1 - (b+c)/N, so N cancels:

    pcARI = (RI(M,T) - RI(C,T)) / (1 - RI(C,T)) = 1 - (b_M + c_M) / (b_C + c_C)

i.e. the fraction of the component clustering's pairwise errors that the merge removed.

The MI side uses *normalized* MI, not raw MI, deliberately. Raw MI is monotonic under
coarsening, so MI(M,T) <= MI(C,T) always: it can only ever penalise a correct merge, and
ranks "do nothing" above every method. NMI's denominator shrinks as M coarsens, so a good
merge can raise it.
"""

import logging
from collections import defaultdict
from math import nan

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _labels(mapping, keys):
    return [mapping[k] for k in keys]


def contingency(labels_a, labels_b):
    """Sparse contingency table between two labellings of the same ordered gene list."""
    a_codes, a_idx = np.unique(np.asarray(labels_a, dtype=object), return_inverse=True)
    b_codes, b_idx = np.unique(np.asarray(labels_b, dtype=object), return_inverse=True)
    data = np.ones(len(a_idx), dtype=np.int64)
    table = coo_matrix((data, (a_idx, b_idx)),
                       shape=(len(a_codes), len(b_codes))).tocsr()
    return table, a_codes, b_codes


def _n_choose_2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def pair_counts(labels_x, labels_t):
    """Pair-level agreement between a clustering X and the truth T.

    Returns a dict with:
        a  pairs co-clustered in both X and T
        b  pairs co-clustered in X only  (wrongly merged)
        c  pairs co-clustered in T only  (wrongly split)
        n_pairs  total pairs
    """
    table, _, _ = contingency(labels_x, labels_t)
    counts = np.asarray(table.data, dtype=np.float64)
    a = float(_n_choose_2(counts).sum())
    x_sizes = np.asarray(table.sum(axis=1)).ravel()
    t_sizes = np.asarray(table.sum(axis=0)).ravel()
    pairs_x = float(_n_choose_2(x_sizes).sum())
    pairs_t = float(_n_choose_2(t_sizes).sum())
    n = float(len(labels_x))
    return {
        "a": a,
        "b": pairs_x - a,
        "c": pairs_t - a,
        "n_pairs": n * (n - 1.0) / 2.0,
    }


def _set_partitions(k):
    """All set partitions of range(k), as restricted growth strings."""
    if k == 0:
        return
    a = [0] * k
    while True:
        yield list(a)
        # next restricted growth string
        i = k - 1
        while i > 0:
            ceiling = max(a[:i]) + 1
            if a[i] <= ceiling - 1:
                a[i] += 1
                for j in range(i + 1, k):
                    a[j] = 0
                break
            i -= 1
        else:
            return


def _grouping_cost(groups, sizes, s):
    """Pairwise error of a grouping of C-blocks within one connected component.

    Merging blocks i and j costs |Bi||Bj| - s_ij wrongly-merged pairs; keeping them apart
    costs s_ij wrongly-split pairs. Pairs inside a single C-block are fixed either way and
    are excluded, so this is comparable across groupings but is not an absolute error.
    """
    k = len(sizes)
    cost = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            if groups[i] == groups[j]:
                cost += sizes[i] * sizes[j] - s[i, j]
            else:
                cost += s[i, j]
    return cost


def _optimal_grouping(sizes, s, exact_max_blocks=8):
    """Minimise pairwise error over groupings of one component's C-blocks.

    Exhaustive below `exact_max_blocks` (Bell(8) = 4140, trivial); greedy agglomerative
    above it, merging the most beneficial pair of groups until no merge helps.
    """
    k = len(sizes)
    if k == 1:
        return [0]

    if k <= exact_max_blocks:
        return min(_set_partitions(k), key=lambda g: _grouping_cost(g, sizes, s))

    groups = list(range(k))
    while True:
        best_delta, best_pair = 0.0, None
        labels = sorted(set(groups))
        for gi_idx, gi in enumerate(labels):
            for gj in labels[gi_idx + 1:]:
                members_i = [i for i in range(k) if groups[i] == gi]
                members_j = [j for j in range(k) if groups[j] == gj]
                # delta = cost(together) - cost(apart), summed over the crossing pairs
                delta = sum(sizes[i] * sizes[j] - 2.0 * s[i, j]
                            for i in members_i for j in members_j)
                if delta < best_delta:
                    best_delta, best_pair = delta, (gi, gj)
        if best_pair is None:
            return groups
        gi, gj = best_pair
        groups = [gi if g == gj else g for g in groups]


def best_reachable_merge(labels_c, labels_t, exact_max_blocks=8):
    """Best clustering reachable from C by merging whole component clusters.

    pangenomerge can only group whole component clusters, so the reachable clusterings are
    exactly the coarsenings of C. This returns the coarsening minimising pairwise error
    against T -- the true ceiling for acARI/acNMI.

    An earlier version simply merged every C-block in a connected component of the
    C-block/T-block bipartite graph. That drives c to zero but chains truth COGs together
    whenever one component cluster straddles two of them, adding more b than it removes;
    real merges then scored above the supposed ceiling (acARI > 1). Each component is now
    optimised on its own: merging blocks i and j trades s_ij wrongly-split pairs for
    |Bi||Bj| - s_ij wrongly-merged ones, so it is worth doing only when s_ij exceeds half
    of |Bi||Bj|.

    Returns (labels_cstar, diagnostics).
    """
    table, c_codes, t_codes = contingency(labels_c, labels_t)
    n_c, n_t = table.shape
    table = table.tocsr()

    coo = table.tocoo()
    rows = np.concatenate([coo.row, coo.col + n_c])
    cols = np.concatenate([coo.col + n_c, coo.row])
    data = np.ones(len(rows), dtype=np.int8)
    bip = coo_matrix((data, (rows, cols)), shape=(n_c + n_t, n_c + n_t))
    n_comp, comp_labels = connected_components(bip, directed=False)

    c_blocks_by_component = defaultdict(list)
    for c_i in range(n_c):
        c_blocks_by_component[comp_labels[c_i]].append(c_i)
    t_per_component = defaultdict(set)
    for t_i in range(n_t):
        t_per_component[comp_labels[n_c + t_i]].add(t_i)

    block_group = {}          # C-block index -> final C* group id
    next_group = 0
    n_chained = n_split = n_greedy = 0

    for comp, c_blocks in c_blocks_by_component.items():
        k = len(c_blocks)
        multi_truth = len(t_per_component[comp]) > 1
        if multi_truth:
            n_chained += 1

        if k == 1 or not multi_truth:
            # one truth COG in play: merging every block is optimal (all crossing pairs
            # are same-COG, so together costs 0 and apart costs s_ij > 0)
            for c_i in c_blocks:
                block_group[c_i] = next_group
            next_group += 1
            continue

        sub = table[c_blocks, :]
        sizes = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
        s = (sub @ sub.T).toarray().astype(np.float64)
        if k > exact_max_blocks:
            n_greedy += 1
        grouping = _optimal_grouping(sizes, s, exact_max_blocks)

        remap = {}
        for c_i, g in zip(c_blocks, grouping):
            if g not in remap:
                remap[g] = next_group
                next_group += 1
            block_group[c_i] = remap[g]
        if len(set(grouping)) > 1:
            n_split += 1        # chaining avoided: component left as >1 block

    code_to_idx = {code: i for i, code in enumerate(c_codes)}
    labels_cstar = [block_group[code_to_idx[lab]] for lab in labels_c]

    diagnostics = {
        "n_component_clusters": int(n_c),
        "n_truth_clusters": int(n_t),
        "n_cstar_clusters": int(len(set(labels_cstar))),
        "n_chained_components": int(n_chained),
        "n_components_left_split": int(n_split),
        "n_components_greedy": int(n_greedy),
        "n_connected_components": int(n_comp),
    }
    return labels_cstar, diagnostics


def _rescale(value, low, high):
    """Linear rescale putting `low` at 0 and `high` at 1; nan if the anchors coincide."""
    span = high - low
    if span <= 0:
        return nan
    return (value - low) / span


def graph_similarity_scores(truth_map, merged_map, component_map,
                            compute_attainable=True):
    """Score a merge M against truth T, with the component clustering C as the baseline.

    Each argument is {seqID: cluster_label}. Scores are computed on the seqIDs common to
    all three (unmapped 'error' ids dropped).

    Returns standard sklearn scores, the post-clustering pair (pcARI/pcNMI) and, unless
    disabled, the attainability-corrected pair (acARI/acNMI) plus the pair-count
    decomposition the latter is built from.
    """
    from sklearn.metrics import (rand_score, adjusted_rand_score, mutual_info_score,
                                 adjusted_mutual_info_score,
                                 normalized_mutual_info_score)

    common = sorted((set(truth_map) & set(merged_map) & set(component_map)) - {"error"})
    truth = _labels(truth_map, common)
    merged = _labels(merged_map, common)
    component = _labels(component_map, common)

    out = {"n_seqIDs": len(common)}

    # --- standard scores (per-gene random-permutation null) ---
    out["RI"] = rand_score(truth, merged)
    out["ARI"] = adjusted_rand_score(truth, merged)
    out["MI"] = mutual_info_score(truth, merged)
    out["AMI"] = adjusted_mutual_info_score(truth, merged)

    # --- pair decomposition: which errors exist, and which are even fixable ---
    pc_m = pair_counts(merged, truth)
    pc_c = pair_counts(component, truth)
    err_m = pc_m["b"] + pc_m["c"]
    err_c = pc_c["b"] + pc_c["c"]
    out.update({
        "pairs_total": pc_c["n_pairs"],
        "err_merged": err_m, "err_component": err_c,
        "b_merged": pc_m["b"], "c_merged": pc_m["c"],
        # b_component is the irreducible floor: pairs C groups that T splits can never be
        # undone by merging, so no reachable merge can score better than this
        "b_component": pc_c["b"], "c_component": pc_c["c"],
    })

    # --- post-clustering scores: 0 = C, 1 = T ---
    ri_ct = rand_score(truth, component)
    nmi_mt = normalized_mutual_info_score(truth, merged)
    nmi_ct = normalized_mutual_info_score(truth, component)
    out["RI_component"] = ri_ct
    out["NMI_merged"] = nmi_mt
    out["NMI_component"] = nmi_ct
    out["pcARI"] = _rescale(out["RI"], ri_ct, 1.0)
    out["pcNMI"] = _rescale(nmi_mt, nmi_ct, 1.0)

    if not compute_attainable:
        return out

    # --- attainability-corrected scores: 0 = C, 1 = C* (best reachable merge) ---
    cstar, diag = best_reachable_merge(component, truth)
    pc_cstar = pair_counts(cstar, truth)
    err_cstar = pc_cstar["b"] + pc_cstar["c"]
    nmi_cstar = normalized_mutual_info_score(truth, cstar)

    out.update({f"cstar_{k}": v for k, v in diag.items()})
    out["err_cstar"] = err_cstar
    out["NMI_cstar"] = nmi_cstar
    # fraction of *removable* error removed, vs pcARI's fraction of *total* error
    out["acARI"] = _rescale(err_c - err_m, 0.0, err_c - err_cstar)
    out["acNMI"] = _rescale(nmi_mt, nmi_ct, nmi_cstar)

    # C* chains truth COGs together whenever a component cluster straddles two of them, so
    # it is an upper bound rather than the exact optimum; a smarter merge can then beat it.
    if out["acARI"] is not nan and out["acARI"] > 1.0:
        logging.warning(
            f"acARI = {out['acARI']:.4f} > 1: the merge beat the reachable-merge ceiling. "
            f"{diag['n_chained_components']} of {diag['n_connected_components']} components "
            f"chain multiple truth COGs, so C* over-merges and understates the ceiling.")
    return out
