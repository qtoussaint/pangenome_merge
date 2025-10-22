import networkx as nx

# define context similarity function
def context_similarity_seq(G, nA, nB, centroid_identity, depth=1):
    neighA = set(nx.single_source_shortest_path_length(G, nA, cutoff=depth).keys())
    neighB = set(nx.single_source_shortest_path_length(G, nB, cutoff=depth).keys())

    centroidsA = [G.nodes[n]["centroid"][0] for n in neighA]
    centroidsB = [G.nodes[n]["centroid"][0] for n in neighB]

    best_ident = 0
    for ca in centroidsA:
        for cb in centroidsB:
            ident = centroid_identity.get((ca, cb), 0)
            if ident > best_ident:
                best_ident = ident
    return best_ident