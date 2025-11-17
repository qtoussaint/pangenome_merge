import networkx as nx
import logging
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# pre-index mmseqs for faster lookups of max identity per unordered pair
def build_ident_lookup(mmseqs: pd.DataFrame) -> dict:
    
    # use unordered pairs so (A,B) and (B,A) are the same key
    keys = mmseqs.apply(lambda r: frozenset((r["query"], r["target"])), axis=1)
    
    # take max fident per unordered pair
    return mmseqs.assign(_key=keys).groupby("_key")["fident"].max().to_dict()

# define context similarity function
def context_similarity_seq(G: nx.Graph, nA, nB, ident_lookup: dict, depth: int = 1) -> float:

    if depth == 1:
        neighA = set(G.neighbors(nA))
        neighB = set(G.neighbors(nB))
    else:

        # use BFS expansion for larger depth
        neighA = set(nx.single_source_shortest_path_length(G, nA, cutoff=depth).keys())
        neighB = set(nx.single_source_shortest_path_length(G, nB, cutoff=depth).keys())

        # need to discard nA/nB since BFS includes self nodes
        neighA.discard(nA)
        neighB.discard(nB)

    best = 0.0
    for ca, cb in product(neighA, neighB):
        best = max(best, ident_lookup.get(frozenset((ca, cb)), 0.0))
        if best == 1.0:
            break  # early exit if perfect
    return best

# define scores per pair function previously implemented in main to allow for parallelization
def score_pair(row, ident_lookup, graph):

    nA = row.query
    nB = row.target
    ident = row.fident

    s1 = context_similarity_seq(merged_graph, nA, nB, ident_lookup, depth=1)
    s2 = s1 if s1 >= 0.9 else context_similarity_seq(merged_graph, nA, nB, ident_lookup, depth=2)
    s3 = s2 if s2 >= 0.9 else context_similarity_seq(merged_graph, nA, nB, ident_lookup, depth=3)
    sims = [s1, s2, s3]

    scores.append((nA, nB, ident, sims))
    
    return (nA, nB, ident, [s1, s2, s3])

# initialize global graph object, ident lookup for // computation without pickling
def init_parallel(merged_graph, ident_lookup):
    global GLOBAL_GRAPH, GLOBAL_IDENT_LOOKUP
    GLOBAL_GRAPH = merged_graph
    GLOBAL_IDENT_LOOKUP = ident_lookup

# parallel computation of scores
def compute_scores_parallel(mmseqs, n_jobs):
    rows = list(mmseqs.itertuples(index=False))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_jobs) as pool:
        return pool.map(_score_pair, rows)