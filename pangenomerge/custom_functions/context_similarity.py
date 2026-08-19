import networkx as nx
import logging
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# pre-index mmseqs for faster lookups of max identity per unordered pair
def build_ident_lookup(mmseqs: pd.DataFrame) -> dict:
    """{frozenset((nodeA, nodeB)): max fident} over the hit table.

    Unordered keys so (A,B) and (B,A) collapse, and max so a node pair with several hit rows
    keeps its best. That max is what makes the representative context lookup correct for the
    multi-representative strategies: an all-unique node contributes one row per allele pair,
    and the pair is scored on the best-matching allele.

    Built with an explicit loop rather than groupby(apply) -- a row-wise apply over the hit
    table dominated runtime once all-unique representatives pushed it into the millions of
    rows.
    """
    fident = pd.to_numeric(mmseqs["fident"], errors="coerce").to_numpy()
    lookup = {}
    for q, t, f in zip(mmseqs["query"].to_numpy(), mmseqs["target"].to_numpy(), fident):
        if f != f:  # NaN, as groupby(max) would also drop
            continue
        key = frozenset((q, t))
        prev = lookup.get(key)
        if prev is None or f > prev:
            lookup[key] = f
    return lookup

# define context similarity function
def context_similarity_seq(G: nx.Graph, nA, nB, ident_lookup: dict, depth: int = 1) -> float:

    if depth == 1:
        neighA = set(G.neighbors(nA))
        neighB = set(G.neighbors(nB))

        #print(f"depth 1 neighA - should be node names! {neighA}")
        #print(f"depth 1 neighB - should be node names! {neighB}")

    else:

        # use BFS expansion for larger depth
        neighA = set(nx.single_source_shortest_path_length(G, nA, cutoff=depth).keys())
        neighB = set(nx.single_source_shortest_path_length(G, nB, cutoff=depth).keys())

        # need to discard nA/nB since BFS includes self nodes
        neighA.discard(nA)
        neighB.discard(nB)

    # if any neighbor node name is shared between neighborhoods, identity is trivially 1.0
    if not neighA.isdisjoint(neighB):
        return 1.0

    best = 0.0
    for ca, cb in product(neighA, neighB):
        best = max(best, ident_lookup.get(frozenset((ca, cb)), 0.0))
        if best == 1.0:
            break  # early exit if perfect
    return best

# define scores per pair function previously implemented in main to allow for parallelization
def score_pair_context(row: dict):

    nA = row["query"]
    nB = row["target"]
    ident = row["fident"]

    G = GLOBAL_GRAPH
    ident_lookup = GLOBAL_IDENT_LOOKUP
    threshold = GLOBAL_CONTEXT_THRESHOLD

    s1 = context_similarity_seq(G, nA, nB, ident_lookup, depth=1)
    s2 = s1 if s1 >= 0.9 else context_similarity_seq(G, nA, nB, ident_lookup, depth=2)
    s3 = s2 if s2 >= 0.9 else context_similarity_seq(G, nA, nB, ident_lookup, depth=3)
    sims = [s1, s2, s3]

    return (nA, nB, ident, sims)

# initialize global graph object, ident lookup for // computation without pickling
GLOBAL_GRAPH = None
GLOBAL_IDENT_LOOKUP = None
GLOBAL_CONTEXT_THRESHOLD = None
def init_parallel(merged_graph, ident_lookup, context_threshold):
    global GLOBAL_GRAPH, GLOBAL_IDENT_LOOKUP, GLOBAL_CONTEXT_THRESHOLD
    GLOBAL_GRAPH = merged_graph
    GLOBAL_IDENT_LOOKUP = ident_lookup
    GLOBAL_CONTEXT_THRESHOLD = context_threshold

# parallel computation of scores
def compute_scores_parallel(mmseqs: pd.DataFrame, n_jobs: int):
    rows = mmseqs.to_dict(orient="records")
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_jobs) as pool:
        return pool.map(score_pair_context, rows)
