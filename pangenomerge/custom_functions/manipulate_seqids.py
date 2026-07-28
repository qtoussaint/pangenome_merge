import networkx as nx
import pandas as pd
from math import nan
from sklearn.metrics import (rand_score, adjusted_rand_score, mutual_info_score,
                             adjusted_mutual_info_score, normalized_mutual_info_score)

# then replace individual run seqID with seqID from run of all data
def indSID_to_allSID(G, gid_map):

    for node in G.nodes():
        node_SIDs = G.nodes[node].get("seqIDs", "")  
        if node_SIDs:
            updated_SIDs = [
                gid_map.get(sid.strip(), "error")  # put 'error' if not found
                for sid in node_SIDs
            ]
            G.nodes[node]["seqIDs"] = updated_SIDs
    return G

# get individual seqIDs as keys, and the cluster each seqID belongs to as its value
def get_seqIDs_in_nodes(G):  

    dictionary = {}

    for node in G.nodes():
        seq_ids = set(G.nodes[node].get("seqIDs", "error"))

        for SID in seq_ids:
            dictionary[SID] = node

    return dictionary

# Clustering metrics for merged_graph (M) vs graph_all truth (T), reported two ways:
#
#  * standard scores (RI/ARI/MI/AMI) with sklearn's per-gene random-permutation null
#  * reference-corrected scores (ARI*/AMI*) that instead use the unmerged
#    component-graph clustering (C) as the baseline. pangenomerge can only ever group
#    whole panaroo component clusters together, so C -- not a random labelling of
#    individual genes -- is the true "do nothing" starting point.
#
#   ARI* = (RI(M,T)  - RI(C,T) ) / (1 - RI(C,T) )
#   AMI* = (NMI(M,T) - NMI(C,T)) / (1 - NMI(C,T))
#
# where M = merged_graph, T = graph_all (truth), C = union of component graphs.
# Both base indices are raw [0,1] similarities that equal 1 when M reproduces T
# (RI(T,T)=NMI(T,T)=1), so the metrics read as "fraction of the gap between leaving
# the component graphs unmerged and reproducing graph_all that the merge closed":
# 0 = no better than the raw component clusters, 1 = reproduces graph_all,
# negative = the merge made agreement with the truth worse.
#
# The MI side uses *normalized* MI rather than raw MI on purpose: pangenomerge only
# coarsens C, and raw MI is monotonic under coarsening (data-processing inequality),
# so MI(M,T) <= MI(C,T) always -- raw MI can only penalise, never reward, a correct
# merge, and its ceiling collapses to 0 when C is finer than T. NMI's denominator
# shrinks as M coarsens, so (like RI) it is non-monotonic and a good merge can raise
# it above the component baseline.
def reference_corrected_scores(truth_map, merged_map, component_map):

    # align on seqIDs present in all three clusterings (drop unmapped 'error' ids)
    common = (set(truth_map) & set(merged_map) & set(component_map)) - {"error"}
    common = sorted(common)

    truth = [truth_map[sid] for sid in common]
    merged = [merged_map[sid] for sid in common]
    component = [component_map[sid] for sid in common]

    # standard scores: merged vs truth, sklearn per-gene random-permutation baseline
    ri_mt = rand_score(truth, merged)
    ari = adjusted_rand_score(truth, merged)
    mi = mutual_info_score(truth, merged)
    ami = adjusted_mutual_info_score(truth, merged)

    # reference-corrected scores: component-graph clustering as the baseline
    ri_ct = rand_score(truth, component)
    ari_star = (ri_mt - ri_ct) / (1.0 - ri_ct) if ri_ct < 1.0 else nan

    nmi_mt = normalized_mutual_info_score(truth, merged)
    nmi_ct = normalized_mutual_info_score(truth, component)
    ami_star = (nmi_mt - nmi_ct) / (1.0 - nmi_ct) if nmi_ct < 1.0 else nan

    return {
        "n_seqIDs": len(common),
        # standard (merged vs all)
        "RI": ri_mt, "ARI": ari, "MI": mi, "AMI": ami,
        # reference-corrected (component graphs as baseline)
        "RI_component": ri_ct, "ARI_star": ari_star,
        "NMI_merged": nmi_mt, "NMI_component": nmi_ct, "AMI_star": ami_star,
    }


# flatten seqID/cluster dictionaries for input into clustering metrics
def dict_to_2d_array(d):
    row_keys = []
    row_values = []

    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            for v in value:
                row_keys.append(key)
                row_values.append(v)
        else:
            row_keys.append(key)
            row_values.append(value)

    return pd.DataFrame([row_keys, row_values])
