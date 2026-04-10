"""
Reference-based layout for pangenome_merge graphs.

Adapted from Panaroo's scripts/reference_based_layout.py.
The core BFS + min-cut algorithm is unchanged. The only modifications
(marked with '# CHANGED:') replace GML-attribute lookups with SQLite
queries, since pangenome_merge's final_graph.gml has empty genomeIDs/
geneIDs fields.
"""

import networkx as nxvisited
import sqlite3
import sys
import logging
import networkx.classes.function as function
import networkx.algorithms.connectivity.cuts as cuts
import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---- CHANGED: new helpers for SQLite-based lookups -------------------------

def is_refound_geneid(gid: str) -> bool:
    """True if any token in the gene ID equals 'refound'."""
    return any(tok == "refound" for tok in gid.split("_"))


def parse_geneid(gid: str):
    """Parse geneid like '0_3_60_g1' -> (scaffold=3, position=60).

    Token layout: {genome}_{scaffold}_{position}_g{graph_id}
    Panaroo sorts genes by genomic coordinate within each scaffold.
    """
    toks = gid.split("_")
    return (int(toks[1]), int(toks[2]))


def load_reference_data(db_path, ref_sample_name):
    """Pre-load all reference genome data from SQLite into memory.

    Returns:
        ref_member_id       — e.g. '0_g1'
        node_to_ref_geneid  — dict[node_id, geneid_str] (non-refound)
        node_to_order_key   — dict[node_id, (scaffold, position)]
        ref_node_ids        — set of all node_ids containing the reference
    """
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA query_only=ON;")

    # Resolve sample name -> member ID
    rows = con.execute(
        "SELECT graph_id, member_index FROM isolate_names WHERE sample_name = ?",
        (ref_sample_name,)
    ).fetchall()
    if not rows:
        available = [r[0] for r in con.execute(
            "SELECT DISTINCT sample_name FROM isolate_names "
            "ORDER BY sample_name LIMIT 20"
        ).fetchall()]
        con.close()
        sys.exit(
            f"ERROR: Sample '{ref_sample_name}' not found in isolate_names.\n"
            f"Available samples (first 20): {available}"
        )
    graph_id, member_index = rows[0]
    ref_member_id = f"{member_index}_g{graph_id}"
    logger.info(f"Resolved '{ref_sample_name}' -> member_id '{ref_member_id}'")

    # All nodes containing this member
    ref_node_ids = set()
    for (nid,) in con.execute(
        "SELECT node_id FROM node_members WHERE member = ?", (ref_member_id,)
    ):
        ref_node_ids.add(nid)

    # Node -> geneid mapping (excluding refound)
    node_to_ref_geneid = {}
    node_to_order_key = {}
    for node_id, geneid in con.execute(
        "SELECT node_id, geneid FROM node_geneids WHERE member = ?",
        (ref_member_id,)
    ):
        if is_refound_geneid(geneid):
            continue
        try:
            key = parse_geneid(geneid)
        except (IndexError, ValueError):
            logger.warning(f"Skipping unparseable geneid: {geneid!r}")
            continue
        if node_id not in node_to_ref_geneid or key < node_to_order_key[node_id]:
            node_to_ref_geneid[node_id] = geneid
            node_to_order_key[node_id] = key

    # Load edge sizes from SQLite (edges stored canonically with u <= v)
    edge_sizes = {}
    for u, v, size in con.execute("SELECT u, v, size FROM edges"):
        edge_sizes[(u, v)] = size if size is not None else 1

    con.close()

    scaffolds = set(k[0] for k in node_to_order_key.values())
    if len(scaffolds) > 10:
        logger.warning(
            f"Reference genome spans {len(scaffolds)} scaffolds. "
            "Gene ordering assumes a single chromosome; results may be "
            "less meaningful for highly fragmented assemblies."
        )
    logger.info(
        f"Reference: {len(ref_node_ids)} total nodes, "
        f"{len(node_to_ref_geneid)} mapped (non-refound), "
        f"{len(scaffolds)} scaffold(s)"
    )
    return ref_member_id, node_to_ref_geneid, node_to_order_key, ref_node_ids, edge_sizes


# ---- end new helpers -------------------------------------------------------


# CHANGED: distance now uses (scaffold, position) tuples instead of
# parsing gene-ID strings.  Same-scaffold distances use the original
# circular formula; different-scaffold pairs are treated as maximally distant.
def get_dist(key_s, key_t, max_positions_per_scaffold):
    if key_s[0] != key_t[0]:
        return float("inf")
    s = key_s[1]
    t = key_t[1]
    max_dist = max_positions_per_scaffold.get(key_s[0], abs(s - t))
    return min(abs(s - t), abs(abs(s - t) - max_dist))


# CHANGED: checks node membership via ref_node_ids set and positions via
# node_to_order_key dict, instead of parsing G.nodes[i]['genomeIDs'] and
# G.nodes[i]['geneIDs'].
def add_to_queue(G, s, nodes, visited, sink,
                 node_to_order_key, ref_node_ids, source_key,
                 max_positions_per_scaffold, distance_threshold):
    add = []
    for i in nodes:
        if i in visited:
            continue
        name_i = G.nodes[i]['name']                       # CHANGED: lookup by name
        if name_i not in ref_node_ids:                    # CHANGED: was genomeIDs check
            add.append(i)
        else:
            #if we have discovered a refound gene we just continue
            key_i = node_to_order_key.get(name_i)         # CHANGED: was geneIDs parse
            if key_i is None:
                # refound or unmapped
                sink["sink"] = i
                visited.add(i)
                continue
            dist = get_dist(source_key, key_i,            # CHANGED: was string parse
                            max_positions_per_scaffold)
            if dist > distance_threshold:
                sink["sink"] = i
        visited.add(i)
    return add


# CHANGED: builds mapping from pre-loaded dicts instead of iterating
# G.nodes and parsing geneIDs attribute.
def create_mapping(node_to_ref_geneid, node_to_order_key):
    gene_dict = {}
    for node_id, geneid in node_to_ref_geneid.items():
        gene_dict[node_id] = geneid
    mapping = pd.DataFrame.from_dict(gene_dict, orient='index')
    mapping.columns = ["gene_id"]
    # CHANGED: sort by (scaffold, position) instead of int(gene_id.split("_")[2])
    mapping["_sort_key"] = [node_to_order_key[n] for n in mapping.index]
    mapping.sort_values("_sort_key", inplace=True)
    mapping.drop(columns=["_sort_key"], inplace=True)
    return mapping


def add_ref_edges(G, mapping):
    name_dict = dict([(G.nodes[n]['name'], n) for n in G.nodes()])
    # CHANGED: removed the seq-parsing loop; mapping is already sorted
    j = 0
    for i in range(1, mapping.shape[0]):
        node1 = str(name_dict[mapping.index[i]])
        node2 = str(name_dict[mapping.index[i - 1]])
        if not G.has_edge(node1, node2):
            j += 1
            G.add_edge(node1, node2)
    return G


def layout(graph, ref_sample_name, metadata_db, cut_edges_out,
           add_reference_edges, distance_threshold=100):

    # CHANGED: load reference data from SQLite instead of parsing GML attributes
    ref_member_id, node_to_ref_geneid, node_to_order_key, ref_node_ids, edge_sizes = \
        load_reference_data(metadata_db, ref_sample_name)

    G = nx.read_gml(graph)

    # CHANGED: build mapping from SQLite data
    mapping = create_mapping(node_to_ref_geneid, node_to_order_key)
    if mapping.empty:
        logger.warning("No reference genes mapped — writing empty cut edges file")
        with open(cut_edges_out, "w") as f:
            f.write("shared name\tis_cut_edge\n")
        return

    # CHANGED: compute max position per scaffold for circular distance
    max_positions_per_scaffold = {}
    for key in node_to_order_key.values():
        scaff, pos = key
        if scaff not in max_positions_per_scaffold or pos > max_positions_per_scaffold[scaff]:
            max_positions_per_scaffold[scaff] = pos

    if add_reference_edges:
        G = add_ref_edges(G, mapping)
    name_dict = dict([(G.nodes[n]['name'], n) for n in G.nodes()])
    #set capacity for edges for the min cut algorithm as the weight of that edge
    for e in G.edges:
        name_u = G.nodes[e[0]]['name']
        name_v = G.nodes[e[1]]['name']
        # edges table stores canonical order (u <= v)
        key = (name_u, name_v) if name_u <= name_v else (name_v, name_u)
        G.edges[e]["capacity"] = edge_sizes.get(key, 1)
    #store edges to be taken out of the graph
    cut_edges = []
    i = 0
    cur_try = 0
    #iterate over all reference nodes in mapping table
    while i < len(mapping.index):
        n = mapping.index[i]
        print(i)
        if n not in name_dict:
            i += 1
            continue
        nid = name_dict[n]
        source_key = node_to_order_key[n]             # CHANGED: get position key
        visited = set([nid])
        sink = {"sink": None}
        # CHANGED: pass SQLite lookup structures instead of ref_g_id + mapping
        queue = add_to_queue(G, nid, G.neighbors(nid), visited, sink,
                             node_to_order_key, ref_node_ids, source_key,
                             max_positions_per_scaffold, distance_threshold)
        #depth first search
        last_target = None
        while len(queue) != 0:
            target = queue.pop(0)
            visited.add(target)
            neighbors = G.neighbors(target)
            #for each reference node explore all edges that lead to non-reference nodes
            queue = queue + add_to_queue(G, nid, neighbors, visited, sink,
                                         node_to_order_key, ref_node_ids,
                                         source_key, max_positions_per_scaffold,
                                         distance_threshold)
        last_target = None
        #did we find a long-range connection?
        if sink["sink"] is not None:
            print("found path")
            visited.add(sink["sink"])
            s_t_graph = function.induced_subgraph(G, visited)
            s_t_graph = nx.Graph(s_t_graph)

            # Ensure capacity values are valid floats for min-cut
            for u, v in list(s_t_graph.edges()):
                d = s_t_graph[u][v]
                cap = d.get("capacity", 1)
                try:
                    d["capacity"] = float(cap)
                    if not (d["capacity"] > 0):
                        d["capacity"] = 1.0
                except (TypeError, ValueError):
                    d["capacity"] = 1.0

            #the induced graph could contain reference edges which need to be removed
            remove = []
            for e in s_t_graph.edges:
                # CHANGED: use ref_node_ids set instead of genomeIDs string
                name0 = G.nodes[e[0]]['name']
                name1 = G.nodes[e[1]]['name']
                if name0 in ref_node_ids and name1 in ref_node_ids:
                    key0 = node_to_order_key.get(name0)
                    key1 = node_to_order_key.get(name1)
                    if key0 is None or key1 is None:
                        # refound or unmapped — keep the edge
                        continue
                    if get_dist(key0, key1, max_positions_per_scaffold) \
                            < distance_threshold:
                        remove.append(e)
            s_t_graph.remove_edges_from(remove)
            #print some info about that long-range connection
            #print(n)
            #print(nid, sink["sink"])
            #min cut between the two reference nodes
            cut = []
            cut_weight, partitions = nx.algorithms.flow.minimum_cut(
                s_t_graph, nid, sink["sink"])
            for p1_node in partitions[0]:
                for p2_node in partitions[1]:
                    if s_t_graph.has_edge(p1_node, p2_node):
                        cut.append((p1_node, p2_node))
            #cardinality cut TODO make this an option
            #cut = cuts.minimum_edge_cut(s_t_graph, nid, sink["sink"])
            for e in cut:
                print(G.nodes[e[0]]['name'], G.nodes[e[1]]['name'])
                cut_edges.append(e)
            #delete cut edges from the graph
            if len(cut) == 0:
                #something happened as no min cut can be found
                i += 1
                raise NameError(
                    "no min cut could be found; sorry this shouldn't happen")
            G.remove_edges_from(cut)
            sink["sink"] = None
            #there may be more paths from that node -> apply again on the same node
        else:
            #all nodes explored; move on
            i += 1
            sink["sink"] = None
    #write gml with reference edges (and cut edges removed) to disk
    if add_reference_edges:
        nx.write_gml(G, graph.replace(".gml", "_with_ref.gml"))
    #write cut edges to disk
    with open(cut_edges_out, "w") as f:
        f.write("shared name\tis_cut_edge\n")
        for e in cut_edges:
            f.write("%s (interacts with) %s\t1\n" % (e[0], e[1]))
            f.write("%s (interacts with) %s\t1\n" % (e[1], e[0]))
    #DEBUG to compress the graph
    #for n in G.nodes:
    #    gene_ids = [(i.split("_")[0], i) for i in  G.nodes[n]['geneIDs'].split(";")]
    #    gene_ids = list(filter(lambda x: ref_g_id == x[0],gene_ids))
    #    if len(gene_ids) == 1:
    #        G.nodes[n]['geneIDs'] = ""
    #    else:
    #        G.nodes[n]['geneIDs'] = gene_ids[0][1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "enable reference-based layouting through detecting long-range connection between otherwise distant genes in a reference genome"
    )
    # CHANGED: ref_sample_name instead of ref_g_id
    parser.add_argument(
        "ref_sample_name",
        help="reference sample name as in isolate_names table "
             "(e.g. 'SAMEA1033240.contigs.fa')")
    parser.add_argument("graph", help='path to final_graph.gml')
    # CHANGED: new metadata_db argument
    parser.add_argument("metadata_db", help='path to pangenome_metadata.sqlite')
    parser.add_argument("cut_edges_out", help='file for cut edges')
    parser.add_argument(
        "--add_reference_edges",
        action="store_true",
        help=
        'add edges between consecutive genes in the reference genome even if they have been removed by panaroo'
    )
    # CHANGED: configurable distance threshold (was hardcoded 100 in original)
    parser.add_argument("--distance_threshold",
                        type=int, default=100,
                        help='position distance threshold for long-range detection (default: 100)')
    args = parser.parse_args()
    layout(graph=args.graph,
           ref_sample_name=args.ref_sample_name,
           metadata_db=args.metadata_db,
           cut_edges_out=args.cut_edges_out,
           add_reference_edges=args.add_reference_edges,
           distance_threshold=args.distance_threshold)
