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

# add directory above __main__.py to sys.path to allow searching for modules there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pangenomerge.manipulate_seqids import indSID_to_allSID, get_seqIDs_in_nodes, dict_to_2d_array
from panaroo_functions.load_graphs import load_graphs

#from .__init__ import __version__


def get_options():
    description = 'Merges two Panaroo pan-genome gene graphs.'
    parser = argparse.ArgumentParser(description=description,
                                    prog='pangenomerge')

    IO = parser.add_argument_group('Input/Output options')
    IO.add_argument('--graph_1',
                    default=None,
                    help='Path to first graph to merge. ("/path/to/final_graph.gml")')
    IO.add_argument('--graph_2',
                    default=None,
                    help='Second graph to merge. ("/path/to/final_graph.gml")')
    IO.add_argument('--mode',
                    default='run',
                    choices=['run', 'test'],
                    help='Run pan-genome gene graph merge ("run") or calculate clustering accuracy metrics for merge ("test"). '
                        '[Default = Run] ')
    IO.add_argument('--mmseqs',
                    default=None,
                    help='Path to mmseqs2 output file (temporary).')
    IO.add_argument('--outdir',
                    default=None,
                    help='Output directory.')
    IO.add_argument('--graph_all',
                    default=None,
                    help='Graph of all samples (test only).')
    IO.add_argument('--gene_data_all',
                    default=None,
                    help='gene_data.csv for graph of all samples (test only).')
    IO.add_argument('--gene_data_1',
                    default=None,
                    help='gene_data.csv for graph_1 (test only).')
    IO.add_argument('--gene_data_2',
                    default=None,
                    help='gene_data.csv for graph_2 (test only).')

    return parser.parse_args()

def main():

    # parse command line arguments
    options = get_options()

    ### read in two graphs

    graph_file_1 = [str(options.graph_1)]
    graph_file_2 = [str(options.graph_2)]

    graph_1, isolate_names, id_mapping = load_graphs(graph_file_1)
    graph_2, isolate_names, id_mapping = load_graphs(graph_file_2)

    graph_1 = graph_1[0]
    graph_2 = graph_2[0]


    if options.mode == 'test':

        ### match clustering_ids from overall run to clustering_ids from individual runs using annotation_ids (test only)

        gene_data_all = pd.read_csv(str(options.gene_data_all))
        gene_data_g1 = pd.read_csv(str(options.gene_data_1))
        gene_data_g2 = pd.read_csv(str(options.gene_data_2))

        # rename column
        gene_data_all = gene_data_all.rename(columns={'clustering_id': 'clustering_id_all'})
        gene_data_g1 = gene_data_g1.rename(columns={'clustering_id': 'clustering_id_indiv'})
        gene_data_g2 = gene_data_g2.rename(columns={'clustering_id': 'clustering_id_indiv'})

        # first match by annotation ids:
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
        matches_g1 = matches_g1.dropna()
        matches_g2 = matches_g2.dropna()

        # convert to dict for faster lookup than with loc
        gid_map_g1 = dict(zip(matches_g1['clustering_id_indiv'], matches_g1['clustering_id_all']))
        gid_map_g2 = dict(zip(matches_g2['clustering_id_indiv'], matches_g2['clustering_id_all']))

        # apply to graphs:
        graph_1 = indSID_to_allSID(graph_1, gid_map_g1)
        graph_2 = indSID_to_allSID(graph_2, gid_map_g2)

    ### map nodes from ggcaller graphs to the COG labels in the centroid from pangenome

    # read into df
    # each "group_" refers to the centroid of that group in the pan_genomes_reference.fa
    mmseqs = pd.read_csv(str(options.mmseqs), sep='\t')

    ### match hits from mmseqs

    # change the second graph node names to the first graph node names for nodes that match according to mmseqs

    # make sure metrics are numeric
    mmseqs["fident"] = pd.to_numeric(mmseqs["fident"], errors='coerce')
    mmseqs["evalue"] = pd.to_numeric(mmseqs["evalue"], errors='coerce')
    mmseqs["tlen"] = pd.to_numeric(mmseqs["tlen"], errors='coerce')
    mmseqs["qlen"] = pd.to_numeric(mmseqs["qlen"], errors='coerce')
    mmseqs["nident"] = pd.to_numeric(mmseqs["nident"], errors='coerce')

    # filter for nt identity >= 98% (global) and length difference <= 5%
    max_len = np.maximum(mmseqs['tlen'], mmseqs['qlen'])
    nt_identity = mmseqs['nident'] / max_len  >= 0.98
    nt_identity = max_len / max_len  >= 0.98
    len_dif = 1-(np.abs(mmseqs['tlen'] - mmseqs['qlen']) / max_len) >= 0.95

    scores = nt_identity & len_dif
    mmseqs = mmseqs[scores].copy()

    # iterate over target with each unique value of target, and pick the match with the highest fident; if multiple, pick the one with the smaller E value

    # sort by fident (highest first) and evalue (lowest first)
    mmseqs_sorted = mmseqs.sort_values(by=["fident", "evalue"], ascending=[False, True])

    # only keep the first occurrence per unique target (highest fident then smallest evalue if tie)
    mmseqs_filtered = mmseqs_sorted.groupby("target", as_index=False).first()

    # in mmseqs, the first graph entered (in this case graph_1) is the query and the second entered (in this case graph_2) is the target
    # so graph_1 is our query in mmseqs and the basegraph in the tokenized merge

    # when iterating over graph_2 to append to graph_1, we want to match nodes according to their graph_1 identity
    # so we need to replace all graph_2 nodes with graph_1 node ids

    ### THE NAME ("group_1") AND THE LABEL ('484') ARE DIFFERENT AND A NUMERIC STRING WILL CALL THE LABEL (not index)

    # the groups are not the same across the two graphs! we match by mmseqs (that's the whole point of this)
    # this chunk is just changing the node name from an integer to the group label of that node (which is originally just
    # metadata within the graph)
    # it doesn't map anything between the two graphs

    mapping_groups_1 = dict()
    for node in graph_1.nodes():
        node_group = graph_1.nodes[node].get("name", "error")
        #print(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
        mapping_groups_1[int(node)] = str(node_group)

    groupmapped_graph_1 = nx.relabel_nodes(graph_1, mapping_groups_1, copy=False)

    mapping_groups_2 = dict()
    for node in graph_2.nodes():
        node_group = graph_2.nodes[node].get("name", "error")
        #print(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
        mapping_groups_2[int(node)] = str(node_group)

    groupmapped_graph_2 = nx.relabel_nodes(graph_2, mapping_groups_2, copy=False)

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

    # relabel target graph from old labels (keys) to new labels (values, the _query-appended graph_1 groups)
    # MUST SET COPY=FALSE OR NODES NOT IN MAPPING WILL BE DROPPED
    relabeled_graph_2 = nx.relabel_nodes(groupmapped_graph_2, mapping, copy=False)

    # relabel query graph
    mapping_query = dict(zip(groupmapped_graph_1.nodes, groupmapped_graph_1.nodes))
    mapping_query = {key: f"{value}_query" for key, value in mapping_query.items()}
    relabeled_graph_1 = nx.relabel_nodes(groupmapped_graph_1, mapping_query, copy=False)

    ### need to do this to edges as well
    #print(relabeled_graph_1.edges)
    #print(relabeled_graph_2.edges)
    # looks like this is done automatically

    # now we can modify the tokenized code to iterate like usual, adding new node if string doesn't contain "_query"
    # and merging the nodes that both end in "_query"
    
    if options.mode == 'test':

        # read in graph_all

        graph_all = [str(options.graph_all)]

        graph_all, isolate_names, id_mapping = load_graphs(graph_all)
        graph_all = graph_all[0]

    ### merge graphs

    merged_graph = relabeled_graph_1
    
    # merge the two sets of unique nodes into one set of unique nodes
    for node in relabeled_graph_2.nodes:
        if merged_graph.has_node(node) == True:

            # add metadata
            merged_set = list(set(relabeled_graph_2.nodes[node]["seqIDs"]) | set(relabeled_graph_1.nodes[node]["seqIDs"]))
            merged_graph.nodes[node]["seqIDs"] = merged_set

        if merged_graph.has_node(node) == False:

            # add node
            merged_graph.add_node(node,
                                 seqIDs=relabeled_graph_2.nodes[node]["seqIDs"])

    for edge in relabeled_graph_2.edges:
        
            if merged_graph.has_edge(edge[0], edge[1]):

                break

            if not merged_graph.has_edge(edge[0], edge[1]):
                merged_graph.add_edge(edge[0], edge[1])
    
    if options.mode == 'test':

        ### gather seqIDs to enable calculation of clustering metrics
        
        cluster_dict_merged = get_seqIDs_in_nodes(merged_graph)
        cluster_dict_all = get_seqIDs_in_nodes(graph_all)

        rand_input_merged = dict_to_2d_array(cluster_dict_merged)
        rand_input_all = dict_to_2d_array(cluster_dict_all)

        # obtain shared seq_ids
        seq_ids_1 = []
        for node in merged_graph.nodes:
            seq_ids_1 += merged_graph.nodes[node]["seqIDs"]
            
        seq_ids_2 = []
        for node in graph_all.nodes:
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


    for node in merged_graph.nodes():
        merged_graph.nodes[node]['seqIDs'] = ";".join(merged_graph.nodes[node]['seqIDs'])
        
    #format_metadata_for_gml(merged_graph)
    
    print('Writing merged graph to outdir...')

    output_path = Path(options.outdir) / "merged_graph.gml"
    
    nx.write_gml(merged_graph, str(output_path))

    print('Finished successfully.')

    #return merged_graph
    
if __name__ == "__main__":
    main()
