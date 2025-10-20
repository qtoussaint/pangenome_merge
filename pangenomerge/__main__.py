# imports

import argparse
import sys
import re
import networkx as nx
import scipy as scipy
import os
import itertools
import pandas as pd
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import rand_score,mutual_info_score,adjusted_rand_score,adjusted_mutual_info_score
from pathlib import Path

from Bio import SeqIO

# add directory above __main__.py to sys.path to allow searching for modules there
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .manipulate_seqids import indSID_to_allSID, get_seqIDs_in_nodes, dict_to_2d_array
from .run_mmseqs import run_mmseqs_easysearch
from panaroo_functions.load_graphs import load_graphs
from panaroo_functions.write_gml_metadata import format_metadata_for_gml
from panaroo_functions.context_search import collapse_families, single_linkage
from panaroo_functions.merge_nodes import merge_node_cluster, gen_edge_iterables, gen_node_iterables, iter_del_dups, del_dups



from .__init__ import __version__

def sync_names(G):
    # write over names with nodes
    # for use after nodes have been updated
    for n in G.nodes:
        G.nodes[n]['name'] = str(n)
    return G

def relabel_nodes_preserve_attrs(G, mapping):
    # Return a new graph with nodes relabeled according to mapping, preserving all attributes and without merging any nodes.
    # Required due to networkx's relabel_nodes() being very naughty

    H = G.__class__()  # same type (Graph, DiGraph, etc.)
    H.graph.update(G.graph)  # copy global graph attrs

    # Step 1: Add nodes (rename if in mapping)
    for n, data in G.nodes(data=True):
        new_n = mapping.get(n, n)
        if new_n in H:
            raise ValueError(f"Duplicate node target detected: {new_n}")
        H.add_node(new_n, **data)

    # Step 2: Add edges (map endpoints)
    for u, v, data in G.edges(data=True):
        new_u = mapping.get(u, u)
        new_v = mapping.get(v, v)
        H.add_edge(new_u, new_v, **data)

    return H

def get_options():
    description = 'Merges two or more Panaroo pan-genome gene graphs, or iteratively updates an existing graph.'
    parser = argparse.ArgumentParser(description=description,
                                    prog='pangenomerge')

    IO = parser.add_argument_group('Input and output options')
    IO.add_argument('--mode',
                    default='run',
                    choices=['run', 'test'],
                    help='Run pan-genome gene graph merge ("run") or calculate clustering accuracy metrics for merge ("test"). '
                        '[Default = Run] ')
    IO.add_argument('--outdir',
                    required=True,
                    default=None,
                    help='Output directory.')
    IO.add_argument('--component-graphs',
                    dest='component_graphs',
                    default=None,
                    required=False,
                    help='Tab-separated list of paths to Panaroo output directories of component subgraphs. \
                    Each directory must contain final_graph.gml and pan_genome_reference.fa. If running in test mode, \
                    must also contain gene_data.csv. Graphs will be merged in the order presented in the file.')
    IO.add_argument('--iterative',
                    dest='iterative',
                    default=None,
                    required=False,
                    help='Tab-separated list of GFFs and their sample IDs for iterative updating of the graph. \
                    Use only for single samples or sets of samples too diverse to create an initial pan-genome. \
                    Samples will be merged in the order presented in the file.')
    IO.add_argument('--graph-all',
                    dest='graph_all',
                    default=None,
                    help='Path to Panaroo output directory of pan-genome gene graph created from all samples in component-graphs. \
                    Only required for the test case, where it is used as the ground truth.')
    
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads',
                    dest="threads",
                    default=2,
                    type=int,
                    help='Number of threads')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # parse command line arguments
    options = get_options()

    if options.component_graphs is None and options.iterative is None:
        raise ValueError("Specifying either --component-graphs or --iterative is required!")

    ### read in two graphs

    graph_files = pd.read_csv(options.component_graphs, sep='\t', header=None)
    n_graphs = int(len(graph_files))
    graph_count = 0

    for graph in range(1, int(n_graphs)):
        
        if graph_count == 0:
            graph_file_1 = str(Path(graph_files.iloc[0][0]) / "final_graph.gml")
            graph_file_2 = str(Path(graph_files.iloc[1][0]) / "final_graph.gml")
        else:
            graph_file_1 = str(Path(options.outdir) / f"merged_graph_{graph_count}.gml")
            graph_file_2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "final_graph.gml")

        print(f"Beginning iteration {graph_count+1} of {n_graphs-1}...")
        print("graph_file_1: ", graph_file_1)
        print("graph_file_2: ", graph_file_2)

        graph_1, isolate_names, id_mapping = load_graphs([graph_file_1])
        graph_2, isolate_names, id_mapping = load_graphs([graph_file_2])

        graph_1 = graph_1[0]
        graph_2 = graph_2[0]

        if options.mode == 'test':

            ### match clustering_ids from overall run to clustering_ids from individual runs using annotation_ids (test only)

            gene_data_all = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))
            gene_data_g2 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count+1)][0]) / "gene_data.csv"))

            if graph_count == 0:
                print("Applying gene data...")
                gene_data_g1 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count)][0]) / "gene_data.csv"))
            else:
                gene_data_g1 = None
                # not necessary because merged graph already has gene_all seqIDs mapped

            # rename column
            gene_data_all = gene_data_all.rename(columns={'clustering_id': 'clustering_id_all'})
            if graph_count == 0:
                print("Applying rename...")
                gene_data_g1 = gene_data_g1.rename(columns={'clustering_id': 'clustering_id_indiv'})
            
            gene_data_g2 = gene_data_g2.rename(columns={'clustering_id': 'clustering_id_indiv'})

            # first match by annotation ids:
            if graph_count == 0:
                print("Applying match...")
                matches_g1 = gene_data_all[['annotation_id', 'clustering_id_all']].merge(
                    gene_data_g1[['annotation_id', 'clustering_id_indiv']],
                    on='annotation_id',
                    how='left'
                )

            matches_g2 = gene_data_all[['annotation_id', 'clustering_id_all']].merge(
                gene_data_g2[['annotation_id', 'clustering_id_indiv']],
                on='annotation_id',
                how='left'
            )

            # now drop rows where the individual seqID wasn't observed (or there's no corresponding seqID from all)
            if graph_count == 0:
                print("Applying dropna...")
                matches_g1 = matches_g1.dropna()
            matches_g2 = matches_g2.dropna()

            # convert to dict for faster lookup than with loc
            if graph_count == 0:
                print("Applying gidmap...")
                gid_map_g1 = dict(zip(matches_g1['clustering_id_indiv'], matches_g1['clustering_id_all']))
            gid_map_g2 = dict(zip(matches_g2['clustering_id_indiv'], matches_g2['clustering_id_all']))

            # apply to graphs:
            if graph_count == 0:
                print("Applying ind...")
                graph_1 = indSID_to_allSID(graph_1, gid_map_g1)
            graph_2 = indSID_to_allSID(graph_2, gid_map_g2)

        ### map nodes from ggcaller graphs to the COG labels in the centroid from pangenome

        ### run mmseqs2 to identify matching COGs
        if graph_count == 0:
            pangenome_reference_g1 = str(Path(graph_files.iloc[0][0]) / "pan_genome_reference.fa")
        else:
            pangenome_reference_g1 = str(Path(options.outdir) / f"pan_genome_reference_{graph_count}.fa")
        
        pangenome_reference_g2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "pan_genome_reference.fa")

        print("Running MMSeqs2...")
        run_mmseqs_easysearch(query=pangenome_reference_g1, target=pangenome_reference_g2, outdir=str(Path(options.outdir) / "mmseqs_clusters.m8"), tmpdir = str(Path(options.outdir) / "mmseqs_tmp"), threads=options.threads)
        print("MMSeqs2 complete. Reading and filtering results...")

        # read mmseqs results
        # each "group_" refers to the centroid of that group in the pan_genomes_reference.fa
        mmseqs = pd.read_csv(str(Path(options.outdir) / "mmseqs_clusters.m8"), sep='\t')

        ### match hits from mmseqs

        # change the second graph node names to the first graph node names for nodes that match according to mmseqs

        # make sure metrics are numeric
        mmseqs["fident"] = pd.to_numeric(mmseqs["fident"], errors='coerce')
        mmseqs["evalue"] = pd.to_numeric(mmseqs["evalue"], errors='coerce')
        mmseqs["tlen"] = pd.to_numeric(mmseqs["tlen"], errors='coerce')
        mmseqs["qlen"] = pd.to_numeric(mmseqs["qlen"], errors='coerce')
        mmseqs["nident"] = pd.to_numeric(mmseqs["nident"], errors='coerce')

        # define length difference
        max_len = np.maximum(mmseqs['tlen'], mmseqs['qlen'])
        mmseqs["len_dif"] = 1 - (np.abs(mmseqs["tlen"] - mmseqs["qlen"]) / max_len)

        # filter for fraction nt identity >= 98% (global) and length difference <= 5%
        mmseqs = mmseqs[(mmseqs["fident"] >= 0.98) & (mmseqs["len_dif"] >= 0.95)].copy()

        ### iterate over target with each unique value of target, and pick the match with the highest fident, then highest len_dif (see calculation)
        # if still multiple matches, pick the first one

        # sort by fident (highest first), len_dif (highest first -- see calculation), and evalue (lowest first)
        mmseqs_sorted = mmseqs.sort_values(by=["fident", "len_dif", "evalue"], ascending=[False, False, True],)

        # only keep the first occurrence per unique target (highest fident, lowest length difference, then smallest evalue if tie)
        mmseqs_filtered = mmseqs_sorted.drop_duplicates(subset=["target"], keep="first")
        mmseqs_filtered = mmseqs_filtered.drop_duplicates(subset=["query"], keep="first") # test if dropping query vs. target duplicates first changes results

        ### TEST

        print(f"Filtered to {len(mmseqs_filtered)} one-to-one hits.")
        dups_target = mmseqs_filtered["target"].duplicated().sum()
        dups_query = mmseqs_filtered["query"].duplicated().sum()
        print(f"Remaining duplicates — target: {dups_target}, query: {dups_query}")

        ### TEST

        # only keep the first occurrence per unique target (highest fident then smallest evalue if tie)
        #mmseqs_filtered = mmseqs_sorted.groupby("target", as_index=False).first()

        print("Hits filtered. Mapping between graphs...")
        print("mmseqs: ", mmseqs_filtered)

        # in mmseqs, the first graph entered (in this case graph_1) is the query and the second entered (in this case graph_2) is the target
        # so graph_1 is our query in mmseqs and the basegraph in the tokenized merge

        # when iterating over graph_2 to append to graph_1, we want to match nodes according to their graph_1 identity
        # so we need to replace all graph_2 nodes with graph_1 node ids

        ### THE NAME ("group_1") AND THE LABEL ('484') ARE DIFFERENT AND A NUMERIC STRING WILL CALL THE LABEL (not index)

        # the groups are not the same across the two graphs! we match by mmseqs (that's the whole point of this)
        # this chunk is just changing the node name from an integer to the group label of that node (which is originally just
        # metadata within the graph)
        # it doesn't map anything between the two graphs

        if graph_count == 0:
            mapping_groups_1 = dict()
            for node in graph_1.nodes():
                node_group = graph_1.nodes[node].get("name", "error")
                print(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
                mapping_groups_1[node] = str(node_group)
            groupmapped_graph_1 = relabel_nodes_preserve_attrs(graph_1, mapping_groups_1)
        else:
            groupmapped_graph_1 = graph_1

        mapping_groups_2 = dict()
        for node in graph_2.nodes():
            node_group = graph_2.nodes[node].get("name", "error")
            print(f"graph: 2, node_index_id: {node}, node_group_id: {node_group}")
            mapping_groups_2[node] = str(node_group)

        groupmapped_graph_2 = relabel_nodes_preserve_attrs(graph_2, mapping_groups_2)

        ### map filtered mmseqs2 hits to mapping of nodes between graphs

        # mapping format: dictionary with old labels (graph_2/target groups) as keys and new labels (graph_1/query) as values

        # convert df to dictionary with "target" as keys and "query" as values
        # this maps groups from graph_1 to groups from graph_2
        mapping = dict(zip(mmseqs_filtered["target"], mmseqs_filtered["query"]))

        ### to avoid matching nodes from target that have the same group_id but are not the same:
        # append all nodes in query graph with _query
        # append all query nodes in target graph with _query (for later matching)

        # this appends _query to values (graph_1/query groups)
        mapping = {key: f"{value}_query" for key, value in mapping.items()}

        #### TESTING 

        # Debug prints
        print(list(groupmapped_graph_2.nodes)[:10])
        print(mmseqs_filtered["target"].unique()[:10])

        # Now it's safe to compare
        missing_keys = set(mapping.keys()) - set(map(str, groupmapped_graph_2.nodes))
        if missing_keys:
            print(f"{len(missing_keys)} mapping targets not found in graph_2. Example: {list(missing_keys)[:5]}")

        #### TESTING 

        # relabel target graph from old labels (keys) to new labels (values, the _query-appended graph_1 groups)
        # MUST SET COPY=FALSE OR NODES NOT IN MAPPING WILL BE DROPPED
        # some of these will just be OG group_XXX (not query-appended) from graph 2; the rest will be group_XXX_query from graph 1
        relabeled_graph_2 = relabel_nodes_preserve_attrs(groupmapped_graph_2, mapping)
        relabeled_graph_2 = sync_names(relabeled_graph_2)

        # append _query to ALL nodes in query graph (ALL nodes of the form group_XXX_query with group_XXX from graph 1)
        mapping_query = dict(zip(groupmapped_graph_1.nodes, groupmapped_graph_1.nodes))
        mapping_query = {key: f"{value}_query" for key, value in mapping_query.items()}
        relabeled_graph_1 = relabel_nodes_preserve_attrs(groupmapped_graph_1, mapping_query)
        relabeled_graph_1 = sync_names(relabeled_graph_1)

        # now we can modify the tokenized code to iterate like usual, adding new node if string doesn't contain "_query"
        # and merging the nodes that both end in "_query"
        
        if options.mode == 'test':
            # read in graph_all
            graph_all = [str(Path(options.graph_all) / "final_graph.gml")]
            graph_all, isolate_names, id_mapping = load_graphs(graph_all)
            graph_all = graph_all[0]

            ### DO I NEED TO MAKE NAMES INTO NODE IDS HERE FOR GRAPH_ALL? > no, because not comparing names between graphs

        ### add suffix to relevant metadata to be able to identify which graph they refer to later

        # if other bits are too slow, replacing looping over nodes with the nodes.values method shown here

        print("Updating graph metadata to prepare for merge...")

        if options.mode != 'test':
            for node_data in relabeled_graph_2.nodes.values():

                node_data['centroid'] = [f"{centroid}_g{graph_count+1}" for centroid in node_data['centroid']] # list

                node_data['maxLenId'] = str(node_data['maxLenId']) + f'_g{graph_count+1}' # int

                node_data['members'] = [f"{member}_g{graph_count+1}" for member in node_data['members']] # list

                node_data['genomeIDs'] = ";".join(node_data['members']) # str

                seqid_set = {f"{seqid}{f'_g{graph_count+1}'}" for seqid in node_data['seqIDs']}
                node_data['seqIDs'] = seqid_set # set

                node_data['longCentroidID'].append(f'from_g{graph_count+1}') #list

                geneids = node_data['geneIDs'].split(";")
                geneids = [f"{gid}_g{graph_count+1}" for gid in geneids]
                node_data['geneIDs'] = ";".join(geneids) # str

        if graph_count == 0 and options.mode != 'test':
            
            for node_data in relabeled_graph_1.nodes.values():

                node_data['centroid'] = [f"{centroid}_g{graph_count}" for centroid in node_data['centroid']] # list

                node_data['maxLenId'] = str(node_data['maxLenId']) + f'_g{graph_count}' # int

                node_data['members'] = [f"{member}_g{graph_count}" for member in node_data['members']] # list

                node_data['genomeIDs'] = ";".join(node_data['members']) # str

                seqid_set = {f"{seqid}{f'_g{graph_count}'}" for seqid in node_data['seqIDs']}
                node_data['seqIDs'] = seqid_set # set

                node_data['longCentroidID'].append(f'from_g{graph_count}') #list

                geneids = node_data['geneIDs'].split(";")
                geneids = [f"{gid}_g{graph_count}" for gid in geneids]
                node_data['geneIDs'] = ";".join(geneids) # str

        ### merge graphs

        print("Beginning graph merge...")

        merged_graph = relabeled_graph_1.copy()

        group = []
        for record in SeqIO.parse(pangenome_reference_g1, "fasta"):
            group.append({"id": record.id, "sequence": str(record.seq)})
        pan_genome_reference_merged = pd.DataFrame(group)

        #if options.mode == 'test':
        #    gene_data_all_new = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))
        
        print("Merging nodes...")

        # merge the two sets of unique nodes into one set of unique nodes
        for node in relabeled_graph_2.nodes:
            if merged_graph.has_node(node) == True:

                # add metadata from g2

                # seqIDs
                merged_set = list(set(relabeled_graph_2.nodes[node]["seqIDs"]) | set(merged_graph.nodes[node]["seqIDs"]))
                merged_graph.nodes[node]["seqIDs"] = merged_set

                # geneIDs
                merged_set = ";".join([merged_graph.nodes[node]["geneIDs"], relabeled_graph_2.nodes[node]["geneIDs"]])
                merged_graph.nodes[node]["geneIDs"] = merged_set

                # members
                merged_set = list(set(relabeled_graph_2.nodes[node]["members"]) | set(merged_graph.nodes[node]["members"]))
                merged_graph.nodes[node]["members"] = merged_set

                # genome IDs
                merged_graph.nodes[node]["genomeIDs"] = ";".join([merged_graph.nodes[node]["genomeIDs"], relabeled_graph_2.nodes[node]["genomeIDs"]])

                # size
                size = len(merged_graph.nodes[node]["members"])

                # lengths
                merged_set = relabeled_graph_1.nodes[node]["lengths"] + relabeled_graph_2.nodes[node]["lengths"]
                merged_graph.nodes[node]["lengths"] = merged_set

                # (don't add centroid/longCentroidID/annotation/dna/protein/hasEnd/mergedDNA/paralog/maxLenId -- keep as original for now)

            else:

                # add node
                merged_graph.add_node(node,
                                    name=relabeled_graph_2.nodes[node]["name"],
                                    centroid=relabeled_graph_2.nodes[node]["centroid"],
                                    size = relabeled_graph_2.nodes[node]["size"],
                                    maxLenId = relabeled_graph_2.nodes[node]["maxLenId"],
                                    lengths = relabeled_graph_2.nodes[node]["lengths"],
                                    members = relabeled_graph_2.nodes[node]["members"],
                                    seqIDs=relabeled_graph_2.nodes[node]["seqIDs"],
                                    hasEnd = relabeled_graph_2.nodes[node]["hasEnd"],
                                    protein=relabeled_graph_2.nodes[node]["protein"],
                                    dna = relabeled_graph_2.nodes[node]["dna"],
                                    annotation=relabeled_graph_2.nodes[node]["annotation"],
                                    description = relabeled_graph_2.nodes[node]["description"],
                                    longCentroidID=relabeled_graph_2.nodes[node]["longCentroidID"],
                                    paralog=relabeled_graph_2.nodes[node]["paralog"],
                                    mergedDNA=relabeled_graph_2.nodes[node]["mergedDNA"],
                                    genomeIDs=relabeled_graph_2.nodes[node]["genomeIDs"],
                                    geneIDs=relabeled_graph_2.nodes[node]["geneIDs"],
                                    degrees=relabeled_graph_2.nodes[node]["degrees"])

                ### add centroid from g2 pan_genome_reference.fa to new merged pan_genome_reference.fa

                # get node centroid
                node_centroid = relabeled_graph_2.nodes[node]["centroid"]

                # relabel node from graph_2 group_xxx to group_xxx_graphcount
                mapping_groups_new = dict()
                #node_group = relabeled_graph_2.nodes[node].get("name", "error")
                mapping_groups_new[node] = f'{node}_{graph_count+1}'
                merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping_groups_new)
                merged_graph = sync_names(merged_graph) # could sync names at end of all merges if slow
                
                # (for centroids of nodes already in main graph, we leave them instead of updating with new centroids
                # to prevent centroids from drifting away over time, and instead maintain consistency)

                #if options.mode == 'test':
                    #node_centroid = next(iter(merged_graph.nodes[f'{node_group}_{graph_count+1}']["seqIDs"]))
                    #node_centroid = gene_data_all_new.loc[gene_data_all_new["clustering_id"] == node_centroid, "dna_sequence"].values
                    #node_centroid = node_centroid[0] # list to string; double check that this doesn't remove centroids

                node_centroid_df = pd.DataFrame([[f"{node}_{graph_count+1}", node_centroid]],
                                columns=["id", "sequence"])

                pan_genome_reference_merged = pd.concat([pan_genome_reference_merged, node_centroid_df])

        print("Merging edges...")

        for edge in relabeled_graph_2.edges:
            
            if merged_graph.has_edge(edge[0], edge[1]):

                # add edge metadata from graph 2 to merged graph
                # edge attributes: size (n members), members (list), genomeIDs (semicolon-separated string)

                unadded_metadata = relabeled_graph_2.edges[edge]

                # members
                merged_graph.edges[edge]['members'] = [str(member) + f"_{graph_count}" for member in merged_graph.edges[edge]['members']] # add underscore with graph count to members of g1
                unadded_metadata['members'] = [str(member) + f"_{graph_count+1}" for member in unadded_metadata['members']] # add underscore w graph count+1 to members of g2
                merged_graph.edges[edge]['members'].extend(unadded_metadata['members']) # combine members

                # genome IDs (assuming genomeIDs are always the same as members):
                merged_graph.edges[edge]['genomeIDs'] = ";".join(merged_graph.edges[edge]['members'])
                
                # size
                merged_graph.edges[edge]['size'] = str(len(merged_graph.edges[edge]['members']))

            else:

                # note that this statement is for NODES not EDGES
                # you could also update g2 edges that don't contain "query" to have _graph_count+1 and then update the edges
                if edge[0] in merged_graph.nodes() and edge[1] in merged_graph.nodes():
                    merged_graph.add_edge(edge[0], edge[1]) # add edge
                    merged_graph.edges[edge].update(relabeled_graph_2.edges[edge]) # update with all metadata

                if edge[0] in merged_graph.nodes() and edge[1] not in merged_graph.nodes():

                    if f"{edge[1]}_{graph_count+1}" in merged_graph.nodes():
                        merged_graph.add_edge(edge[0], f"{edge[1]}_{graph_count+1}") # add edge
                        merged_graph.edges[edge[0], f"{edge[1]}_{graph_count+1}"].update(relabeled_graph_2.edges[edge]) # update with all metadata
                    else:
                        print(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

                if edge[0] not in merged_graph.nodes() and edge[1] in merged_graph.nodes():

                    if f"{edge[0]}_{graph_count+1}" in merged_graph.nodes():
                        merged_graph.add_edge(f"{edge[0]}_{graph_count+1}", edge[1]) # add edge
                        merged_graph.edges[f"{edge[0]}_{graph_count+1}", edge[1]].update(relabeled_graph_2.edges[edge]) # update with all metadata

                    else:
                        print(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

                if edge[0] not in merged_graph.nodes() and edge[1] not in merged_graph.nodes():
                    
                    if f"{edge[0]}_{graph_count+1}" in merged_graph.nodes() and f"{edge[1]}_{graph_count+1}" in merged_graph.nodes():
                        merged_graph.add_edge(f"{edge[0]}_{graph_count+1}", f"{edge[1]}_{graph_count+1}") # add edge
                        merged_graph.edges[f"{edge[0]}_{graph_count+1}", f"{edge[1]}_{graph_count+1}"].update(relabeled_graph_2.edges[edge]) # update with all metadata
                    else: 
                        print(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

        # update degrees across graph
        for node in merged_graph:
            merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])

        print("Collapsing spurious paralogs...")

        # generate dictionary for panaroo collapse families
        seq_ids_nodes = get_seqIDs_in_nodes(merged_graph)
        seq_ids = list(seq_ids_nodes.keys())

        seqid_to_centroid = {}
        for i, id in enumerate(seq_ids):
            node = seq_ids_nodes[id]
            seqid_to_centroid[id] = merged_graph.nodes[node]["centroid"][0] # if multi-centroid, only use first centroid in list

        # collapse over-split families using sequence identity + context search
        # from panaroo main:
        #collapsed_merged_graph, distances_bwtn_centroids, centroid_to_index = collapse_families(
        #    merged_graph,
        #    seqid_to_centroid=seqid_to_centroid,
        #    outdir=options.outdir,
        #    correct_mistranslations=False,
        #    n_cpu=options.threads,
        #    quiet=True)     

        #merged_graph = collapsed_merged_graph 

        # update degrees across graph (post-collapsing)
        for node in merged_graph:
            merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])

        if options.mode == 'test' and graph_count == (n_graphs-2):

            print("Calculating adjusted Rand index (ARI) and adjusted mutual information (AMI)...")

            ### gather seqIDs to enable calculation of clustering metrics
            
            cluster_dict_merged = get_seqIDs_in_nodes(merged_graph)
            cluster_dict_all = get_seqIDs_in_nodes(graph_all)

            rand_input_merged = dict_to_2d_array(cluster_dict_merged)
            rand_input_all = dict_to_2d_array(cluster_dict_all)

            # obtain shared seq_ids
            seq_ids_1 = []

            for node in merged_graph.nodes():
                #print("node", node)
                #if node in merged_graph.nodes:
                    #print(f"merged_graph.nodes[node]['seqIDs']: {merged_graph.nodes[node]['seqIDs']}")
                seq_ids_1 += merged_graph.nodes[node]["seqIDs"]
                
            seq_ids_2 = []
            for node in graph_all.nodes():
                seq_ids_2 += graph_all.nodes[node]["seqIDs"]
                
            seq_ids_1 = set(seq_ids_1)
            seq_ids_2 = set(seq_ids_2)
                
            common_seq_ids = seq_ids_1 & seq_ids_2  # take intersection
            
            # print how many seq_ids were excluded
            only_in_graph_1 = seq_ids_1 - seq_ids_2
            only_in_graph_2 = seq_ids_2 - seq_ids_1
            print(f"shared seqIDs: {len(common_seq_ids)}")
            print(f"seqIDs only in merged (excluded): {len(only_in_graph_1)}")
            print(f"seqIDs only in all (excluded): {len(only_in_graph_2)}")

            rand_input_merged_filtered = rand_input_merged.loc[:, rand_input_merged.loc[0].isin(common_seq_ids)]
            rand_input_all_filtered = rand_input_all.loc[:, rand_input_all.loc[0].isin(common_seq_ids)]
            
            # get desired value order from row 0 of rand_input_all_filtered
            desired_order = list(rand_input_all_filtered.iloc[0])

            # Create mapping from row 0 values in rand_input_merged_filtered to column names
            val_to_col = {val: col for col, val in zip(rand_input_merged_filtered.columns, rand_input_merged_filtered.iloc[0])}

            # Reorder columns based on desired value order
            columns_in_order = [val_to_col[val] for val in desired_order if val in val_to_col]

            # Apply the column reordering
            rand_input_merged_filtered = rand_input_merged_filtered[columns_in_order]
            
            # put sorted clusters into Rand index
            ri = rand_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            print(f"Rand Index: {ri}")

            ari = adjusted_rand_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            print(f"Adjusted Rand Index: {ari}")

            # put sorted clusters into mutual information
            mutual_info = mutual_info_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            print(f"Mutual Information: {mutual_info}")

            adj_mutual_info = adjusted_mutual_info_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            print(f"Adjusted Mutual Information: {adj_mutual_info}")

        print("Merge complete. Preparing attribute metadata for export...")

        # update metadata
        #for node_data in merged_graph.nodes.values():
        #    node_data['seqIDs'] = ";".join(node_data['seqIDs'])
        #    node_data['centroid'] = ";".join(node_data['centroid'])

        if graph_count == 0:
            #for node_data in merged_graph.nodes.values():
            #    if node_data['name'] contains '_query':
            #        # relabel node from graph_1 group_xxx to group_xxx_graphcount
            #        mapping_groups_new = dict()
            #        new_label = node_data['name'].removesuffix('_query')
            #        mapping_groups_new[node] = f'{new_label}_{graph_count}'
            #        merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping_groups_new)
            
            # relabel node from graph_1 group_xxx to group_xxx_graphcount
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                print(f"node name {name}")
                if '_query' in name:
                    new_name = name.replace('_query', f'_{graph_count}')
                    mapping[node_id] = new_name
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)

        else:
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_query' in name:
                    new_name = name.removesuffix('_query')
                    mapping[node_id] = new_name
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)

        #for node in merged_graph.nodes():
        #    merged_graph.nodes[node]['seqIDs'] = ";".join(merged_graph.nodes[node]['seqIDs'])
            #merged_graph.nodes[node]['name'] = merged_graph.nodes[node]['name'].removesuffix('_query')
        
        #mapping_query = dict(zip(merged_graph.nodes, merged_graph.nodes))
        #mapping_query = {key: f"{value.removesuffix('query')}" for key, value in mapping_query.items()}
        #merged_graph_new = relabel_nodes_preserve_attrs(merged_graph, mapping_query)
        #merged_graph = merged_graph_new

        #for node in merged_graph.nodes():
            #print("node ", node)
            
        format_metadata_for_gml(merged_graph)
        
        print('Writing merged graph to outdir...')

        # write new graph to GML
        output_path = Path(options.outdir) / f"merged_graph_{graph_count+1}.gml"
        nx.write_gml(merged_graph, str(output_path))

        # write new pan-genome references to fasta
        reference_out = str(Path(options.outdir) / f"pan_genome_reference_{graph_count+1}.fa")

        with open(reference_out, "w") as fasta_out:
            for _, row in pan_genome_reference_merged.iterrows():
                fasta_out.write(f">{row['id']}\n{row['sequence']}\n")

        #print(pan_genome_reference_merged)
        #pan_genome_reference_merged.to_csv(reference_out, header=False, index=False)

        graph_count += 1

        print(f"Iteration {graph_count} of {n_graphs-1} complete.")

    print('Finished successfully.')

    #return merge
if __name__ == "__main__":
    main()

