import itertools
import logging
from collections import Counter
#from .isvalid import del_dups
import numpy as np
#from intbitset import intbitset

from pangenomerge.custom_functions.relabel_nodes import relabel_nodes_preserve_attrs, sync_names


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


def merge_node_cluster(G,
                       nodes,
                       newNode,
                       multi_centroid=True,
                       check_merge_mems=True):

    if check_merge_mems:
        mem_count = Counter(
            itertools.chain.from_iterable(
                gen_node_iterables(G, nodes, 'members')))
        if max(mem_count.values()) > 1:
            raise ValueError("merging nodes with the same genome IDs!")

    # take node with most support as the 'consensus'
    nodes = sorted(nodes, key=lambda x: G.nodes[x]['size'])

    # First create a new node and combine the attributes
    dna = iter_del_dups(gen_node_iterables(G, nodes, 'dna'))
    maxLenId = 0
    max_l = 0
    for i, s in enumerate(dna):
        if len(s) >= max_l:
            max_l = len(s)
            maxLenId = i

    members = G.nodes[nodes[0]]['members'].copy()
    for n in nodes[1:]:
        members |= G.nodes[n]['members']

    if multi_centroid:
        mergedDNA = any(gen_node_iterables(G, nodes, 'mergedDNA'))
    else:
        mergedDNA = True

    G.add_node(
        newNode,
        size=len(members),
        centroid=iter_del_dups(gen_node_iterables(G, nodes, 'centroid')),
        maxLenId=maxLenId,
        members=members,
        seqIDs=set(iter_del_dups(gen_node_iterables(G, nodes, 'seqIDs'))),
        hasEnd=any(gen_node_iterables(G, nodes, 'hasEnd')),
        protein=iter_del_dups(gen_node_iterables(G, nodes, 'protein')),
        dna=dna,
        annotation=";".join(
            iter_del_dups(gen_node_iterables(G, nodes, 'annotation',
                                             split=";"))),
        description=";".join(
            iter_del_dups(
                gen_node_iterables(G, nodes, 'description', split=";"))),
        lengths=list(
            itertools.chain.from_iterable(
                gen_node_iterables(G, nodes, 'lengths'))),
        longCentroidID=max(gen_node_iterables(G, nodes, 'longCentroidID')),
        paralog=any(gen_node_iterables(G, nodes, 'paralog')),
        mergedDNA=mergedDNA)
    if "prevCentroids" in G.nodes[nodes[0]]:
        G.nodes[newNode]['prevCentroids'] = ";".join(
            set(
                iter_del_dups(
                    gen_node_iterables(G, nodes, 'prevCentroids', split=";"))))

    # Now iterate through neighbours of each node and add them to the new node
    merge_nodes = set(nodes)
    for node in nodes:
        for neighbour in G.neighbors(node):
            if neighbour in merge_nodes: continue
            if G.has_edge(newNode, neighbour):
                G[newNode][neighbour]['members'] |= G[node][neighbour][
                    'members']
                G[newNode][neighbour]['size'] = len(G[newNode][neighbour]['members'])
            else:
                G.add_edge(newNode,
                           neighbour,
                           size=G[node][neighbour]['size'],
                           members=G[node][neighbour]['members'])

    # remove old nodes from Graph
    G.remove_nodes_from(nodes)

    return G


def merge_pair(G, a, b):
    # add metadata from second node

    # seqIDs
    merged_set = list(set(G.nodes[a]["seqIDs"]) | set(G.nodes[b]["seqIDs"]))
    G.nodes[a]["seqIDs"] = merged_set

    # geneIDs
    merged_set = ";".join([G.nodes[a]["geneIDs"], G.nodes[b]["geneIDs"]])
    G.nodes[a]["geneIDs"] = merged_set

    # members
    merged_set = list(set(G.nodes[a]["members"]) | set(G.nodes[b]["members"]))
    G.nodes[a]["members"] = merged_set

    # genome IDs
    G.nodes[a]["genomeIDs"] = ";".join([G.nodes[a]["genomeIDs"], G.nodes[b]["genomeIDs"]])

    # size
    size = len(G.nodes[a]["members"])

    # lengths
    merged_set = G.nodes[a]["lengths"] + G.nodes[b]["lengths"]
    G.nodes[a]["lengths"] = merged_set

    # move edges from b onto a before removing b
    for neighbor in list(G.neighbors(b)):
        #if neighbor == a:
        #    continue

        # get edge attributes of b
        edge_attrs = dict(G.get_edge_data(b, neighbor))

        # warn if don't have edge connecting neighbor
        if not G.has_edge(b, neighbor):
            #edge_attrs = {}
            logging.critical("neighbor not connected by edge -- this shouldn't happen!")

        if G.has_edge(a, neighbor):
            # if the edge exists, merge metadata
            merged_edge = G.edges[a, neighbor]
            merged_members = set(merged_edge.get("members", [])) | set(edge_attrs.get("members", []))
            merged_edge["members"] = list(merged_members)
            merged_edge["size"] = len(merged_members)
        else:
            # otherwise add the edge
            G.add_edge(a, neighbor, **edge_attrs)

    # remove second node
    G.remove_node(b)

    # (don't add centroid/longCentroidID/annotation/dna/protein/hasEnd/mergedDNA/paralog/maxLenId -- keep as original for now)


def apply_merges(G, reordered_pairs):
    for a, b in reordered_pairs:
        merge_pair(G, a, b)


def initial_graph_merge(base_graph, incoming_graph, graph_count):
    # rename base graph
    merged_graph = base_graph

    # debug statement...
    logging.debug(f"Merging graphs. merged_graph currently has {len(merged_graph.nodes())} nodes.")
    logging.debug(f"Incoming relabeled_graph_2 has {len(incoming_graph.nodes())} nodes.")

    # info statement...
    logging.info("Merging nodes...")

    # iterate, adding new node if node doesn't contain "_target"
    # and merging the nodes that both end in "_target"

    # create dictionary of nodes that will be added (not merged into existing nodes)
    mapping_groups_new = {}
    for node in incoming_graph.nodes:
        if not merged_graph.has_node(node):
            mapping_groups_new[node] = f"{node}_g{graph_count+2}"

    # relabel nodes that will be added and sync their names...
    # (faster to do this just on smaller g2 instead of afterwards on merged_graph)

    if mapping_groups_new:  # only relabel if there is something to change
        incoming_graph = relabel_nodes_preserve_attrs(incoming_graph, mapping_groups_new)
        incoming_graph = sync_names(incoming_graph)

    # merge the two sets of unique nodes into one set of unique nodes
    for node in incoming_graph.nodes:
        if merged_graph.has_node(node) == True:

            # add metadata from graph 2

            # (for centroids of nodes already in main graph, we leave them instead of updating with new centroids
            # to prevent centroids from drifting away over time, and instead maintain consistency)

            # seqIDs
            merged_set = list(set(incoming_graph.nodes[node]["seqIDs"]) | set(merged_graph.nodes[node]["seqIDs"]))
            merged_graph.nodes[node]["seqIDs"] = merged_set

            # geneIDs
            merged_set = ";".join([merged_graph.nodes[node]["geneIDs"], incoming_graph.nodes[node]["geneIDs"]])
            merged_graph.nodes[node]["geneIDs"] = merged_set

            # members
            merged_set = list(set(incoming_graph.nodes[node]["members"]) | set(merged_graph.nodes[node]["members"]))
            merged_graph.nodes[node]["members"] = merged_set

            # genome IDs
            merged_graph.nodes[node]["genomeIDs"] = ";".join([merged_graph.nodes[node]["genomeIDs"], incoming_graph.nodes[node]["genomeIDs"]])

            # size
            size = len(merged_graph.nodes[node]["members"])

            # lengths
            merged_set = merged_graph.nodes[node]["lengths"] + incoming_graph.nodes[node]["lengths"]
            merged_graph.nodes[node]["lengths"] = merged_set

            # (don't add centroid/longCentroidID/annotation/dna/protein/hasEnd/mergedDNA/paralog/maxLenId -- keep as original for now)

        else:

            # add node
            merged_graph.add_node(node,
                                name=incoming_graph.nodes[node]["name"],
                                centroid=incoming_graph.nodes[node]["centroid"],
                                size = incoming_graph.nodes[node]["size"],
                                maxLenId = incoming_graph.nodes[node]["maxLenId"],
                                lengths = incoming_graph.nodes[node]["lengths"],
                                members = incoming_graph.nodes[node]["members"],
                                seqIDs=incoming_graph.nodes[node]["seqIDs"],
                                hasEnd = incoming_graph.nodes[node]["hasEnd"],
                                protein=incoming_graph.nodes[node]["protein"],
                                dna = incoming_graph.nodes[node]["dna"],
                                annotation=incoming_graph.nodes[node]["annotation"],
                                description = incoming_graph.nodes[node]["description"],
                                longCentroidID=incoming_graph.nodes[node]["longCentroidID"],
                                paralog=incoming_graph.nodes[node]["paralog"],
                                mergedDNA=incoming_graph.nodes[node]["mergedDNA"],
                                genomeIDs=incoming_graph.nodes[node]["genomeIDs"],
                                geneIDs=incoming_graph.nodes[node]["geneIDs"],
                                degrees=incoming_graph.nodes[node]["degrees"])

    # info statement...
    logging.info("Merging edges...")

    # debug statement...
    logging.debug(f"After merge but before edge merge: merged_graph node sample: {list(merged_graph.nodes())[:20]}")
    logging.debug(f"After merge but before edge merge: {len(merged_graph.nodes())} nodes")

    # add in metadata from merged edges; add in new edges

    for edge in incoming_graph.edges:

        if merged_graph.has_edge(edge[0], edge[1]):

            # add edge metadata from graph 2 to merged graph
            # edge attributes: size (n members), members (list), genomeIDs (semicolon-separated string)

            unadded_metadata = incoming_graph.edges[edge]

            # members
            merged_graph.edges[edge]['members'].extend(unadded_metadata['members']) # combine members

            # genome IDs (assuming genomeIDs are always the same as members):
            merged_graph.edges[edge]['genomeIDs'] = ";".join(merged_graph.edges[edge]['members'])

            # size
            merged_graph.edges[edge]['size'] = str(len(merged_graph.edges[edge]['members']))

        else:

            # note that this statement is for NODES not EDGES

            # edge[0] and edge[1] are node names from nodes in g2
            # e.g. group_XXX_g2 or group_YYY_target

            # we first found any edges that exist in the merged graph, e.g. group_XXX_target <-> group_YYY_target
            # we are now looking for group_XXX_target <-> group_YYY_g2 and group_XXX_g2 <-> group_YYY_g2 and any group_XXX_target <-> group_YYY_target only present in the new graph

            # this finds new group_XXX_target <-> group_YYY_target mappings only present in the g2 (since part of "else" statement)
            if edge[0] in merged_graph.nodes() and edge[1] in merged_graph.nodes():
                merged_graph.add_edge(edge[0], edge[1]) # add edge
                merged_graph.edges[edge].update(incoming_graph.edges[edge]) # update with all metadata

            # these find group_XXX_target <-> group_YYY_g2
            if edge[0] in merged_graph.nodes() and edge[1] not in merged_graph.nodes():

                if f"{edge[1]}_g{graph_count+2}" in merged_graph.nodes():
                    merged_graph.add_edge(edge[0], f"{edge[1]}_g{graph_count+2}") # add edge
                    merged_graph.edges[edge[0], f"{edge[1]}_g{graph_count+2}"].update(incoming_graph.edges[edge]) # update with all metadata
                else:
                    logging.error(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

            if edge[0] not in merged_graph.nodes() and edge[1] in merged_graph.nodes():

                if f"{edge[0]}_g{graph_count+2}" in merged_graph.nodes():
                    merged_graph.add_edge(f"{edge[0]}_g{graph_count+2}", edge[1]) # add edge
                    merged_graph.edges[f"{edge[0]}_g{graph_count+2}", edge[1]].update(incoming_graph.edges[edge]) # update with all metadata
                else:
                    logging.error(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

            # this finds group_XXX_g2 <-> group_YYY_g2
            if edge[0] not in merged_graph.nodes() and edge[1] not in merged_graph.nodes():

                if f"{edge[0]}_g{graph_count+2}" in merged_graph.nodes() and f"{edge[1]}_g{graph_count+2}" in merged_graph.nodes():
                    merged_graph.add_edge(f"{edge[0]}_g{graph_count+2}", f"{edge[1]}_g{graph_count+2}") # add edge
                    merged_graph.edges[f"{edge[0]}_g{graph_count+2}", f"{edge[1]}_g{graph_count+2}"].update(incoming_graph.edges[edge]) # update with all metadata
                else:
                    logging.error(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

    # update degrees across graph
    for node in merged_graph:
        merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])

    # debug statement...
    logging.debug(f"After merge and edge merge: merged_graph node sample: {list(merged_graph.nodes())[:20]}")
    logging.debug(f"After merge and edge merge: {len(merged_graph.nodes())} nodes")

    return merged_graph