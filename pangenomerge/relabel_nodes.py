import networkx as nx
import logging

# return a new graph with nodes relabeled according to mapping, preserving all attributes and without merging any nodes
# required due to networkx's relabel_nodes() being very naughty
def relabel_nodes_preserve_attrs(G, mapping):

    # copy global graph attrs
    H = G.__class__()
    H.graph.update(G.graph)

    # add nodes (rename if in mapping)
    for n, data in G.nodes(data=True):
        new_n = mapping.get(n, n)
        if new_n in H:
            logging.error(f"Duplicate node target detected: {new_n}")
        H.add_node(new_n, **data)

    # add edges (map endpoints)
    for u, v, data in G.edges(data=True):
        new_u = mapping.get(u, u)
        new_v = mapping.get(v, v)
        H.add_edge(new_u, new_v, **data)

    return H

# overwrite names attribute with node labels
# use after node labels are updated
def sync_names(G):
    for n in G.nodes:
        G.nodes[n]['name'] = str(n)
    return G