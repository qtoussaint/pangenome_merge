import networkx as nx
import logging
import pandas as pd
from itertools import product

# define context similarity function
def context_similarity_seq_tmprm(G, nA, nB, mmseqs, depth=1):

    # depth = 1 → immediate neighbors only
    if depth == 1:
        neighA = set(G.neighbors(nA))
        neighB = set(G.neighbors(nB))

        # remove self nodes (they’re included by default)
        neighA.discard(nA)
        neighB.discard(nB)
    else:
        # use BFS expansion for larger depth
        neighA = set(nx.single_source_shortest_path_length(G, nA, cutoff=depth).keys())
        neighB = set(nx.single_source_shortest_path_length(G, nB, cutoff=depth).keys())

        # remove self nodes (they’re included by default)
        neighA.discard(nA)
        neighB.discard(nB)

    logging.debug(f"neighA: {neighA}, neighB: {neighB}")

    best_ident = 0.0

    # Iterate through all centroid pairs between the two neighborhoods
    for ca in neighA:
        for cb in neighB:
            # Check both query/target orientations
            match = mmseqs[
                ((mmseqs["query"] == ca) & (mmseqs["target"] == cb))
                | ((mmseqs["query"] == cb) & (mmseqs["target"] == ca))
            ]
            if not match.empty:
                ident = match["fident"].max()
                if ident > best_ident:
                    best_ident = ident

    logging.debug(f"best ident: {best_ident}")
    return best_ident

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