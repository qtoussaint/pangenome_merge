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

from .__init__ import __version__


def get_options():
    description = 'Merges two or more Panaroo pan-genome gene graphs.'
    parser = argparse.ArgumentParser(description=description,
                                    prog='pangenomerge')

    IO = parser.add_argument_group('Input/Output options')
    IO.add_argument('--mode',
                    default='run',
                    choices=['run', 'test'],
                    help='Run pan-genome gene graph merge ("run") or calculate clustering accuracy metrics for merge ("test"). '
                        '[Default = Run] ')
    IO.add_argument('--outdir',
                    required=True,
                    default=None,
                    help='Output directory.')
    IO.add_argument('--component_graphs',
                    default=None,
                    required=True,
                    help='Tab-separated list of paths to Panaroo output directories of component subgraphs. \
                    Each directory must contain final_graph.gml and pan_genome_reference.fa. If running in test mode, must also contain \
                    gene_data.csv. Graphs will be merged in the order presented in the file.')
    IO.add_argument('--graph_all',
                    default=None,
                    help='Path to Panaroo output directory of pan-genome gene graph created from all samples in component-graphs. \
                    Only required for the test case, where it is used as the ground truth.')
    
    other = parser.add_argument_group('Other options')
    other.add_argument('--mmseqs-threads',
                    default=1,
                    type=int,
                    help='Number of threads for mmseqs2')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # parse command line arguments
    options = get_options()

    ### read in two graphs

    graph_files = pd.read_csv(options.component_graphs, sep='\t', header=None)
    print(graph_files)
    n_graphs = int(len(graph_files))

    print("n_graphs: ", n_graphs)

    graph_count = 0

    for graph in range(1, int(n_graphs)):

        print(graph_files.iloc[0][0])
        
        if graph_count == 0:
            graph_file_1 = str(Path(graph_files.iloc[0][0]) / "final_graph.gml")
            graph_file_2 = str(Path(graph_files.iloc[1][0]) / "final_graph.gml")
        else:
            graph_file_1 = str(Path(options.outdir) / f"merged_graph_{graph_count}.gml")
            graph_file_2 = str(Path(graph_files.iloc[int(graph_count)][0]) / "final_graph.gml")

        print("graph_file_1: ", graph_file_1)
        print("graph_file_2: ", graph_file_2)

        graph_1, isolate_names, id_mapping = load_graphs([graph_file_1])
        graph_2, isolate_names, id_mapping = load_graphs([graph_file_2])

        graph_1 = graph_1[0]
        graph_2 = graph_2[0]

        print(f"{graph_1.edges}")

        if options.mode == 'test':

            ### match clustering_ids from overall run to clustering_ids from individual runs using annotation_ids (test only)

            gene_data_all = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))
            gene_data_g2 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count+1)][0]) / "gene_data.csv"))

            if graph_count == 0:
                print("applying gene data...")
                gene_data_g1 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count)][0]) / "gene_data.csv"))
            else:
                gene_data_g1 = [""]
                # not necessary because merged graph already has gene_all seqIDs mapped
            
            #print("gene_data_all: ", gene_data_all)
            #print("gene_data_g1: ", gene_data_g1)
            #print("gene_data_g2: ", gene_data_g2)

            # rename column
            gene_data_all = gene_data_all.rename(columns={'clustering_id': 'clustering_id_all'})
            if graph_count == 0:
                print("applying rename...")
                gene_data_g1 = gene_data_g1.rename(columns={'clustering_id': 'clustering_id_indiv'})
            
            gene_data_g2 = gene_data_g2.rename(columns={'clustering_id': 'clustering_id_indiv'})

            #print("gene_data_all: ", gene_data_all[['annotation_id', 'clustering_id_all']])
            #print("gene_data_g1: ", gene_data_g1[['annotation_id', 'clustering_id_indiv']])
            print("gene_data_g2: ", gene_data_g2[['annotation_id', 'clustering_id_indiv']])

            # first match by annotation ids:
            if graph_count == 0:
                print("applying match...")
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

            #print("matches_g1", matches_g1)
            print("matches_g2", matches_g2)

            # now drop rows where the individual seqID wasn't observed (or there's no corresponding seqID from all)
            if graph_count == 0:
                print("applying dropna...")
                matches_g1 = matches_g1.dropna()
            matches_g2 = matches_g2.dropna()

            print("matches_g2", matches_g2)

            # convert to dict for faster lookup than with loc
            if graph_count == 0:
                print("applying gidmap...")
                gid_map_g1 = dict(zip(matches_g1['clustering_id_indiv'], matches_g1['clustering_id_all']))
            gid_map_g2 = dict(zip(matches_g2['clustering_id_indiv'], matches_g2['clustering_id_all']))

            # apply to graphs:
            if graph_count == 0:
                print("applying ind...")
                graph_1 = indSID_to_allSID(graph_1, gid_map_g1)
            graph_2 = indSID_to_allSID(graph_2, gid_map_g2)

        ### map nodes from ggcaller graphs to the COG labels in the centroid from pangenome

        ### run mmseqs2 to identify matching COGs
        if graph_count == 0:
            pangenome_reference_g1 = str(Path(graph_files.iloc[0][0]) / "pan_genome_reference.fa")
        else:
            pangenome_reference_g1 = str(Path(options.outdir) / f"pan_genome_reference_{graph_count}.fa")
        
        pangenome_reference_g2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "pan_genome_reference.fa")

        print("pangenome_reference_g1: ", pangenome_reference_g1)
        print("pangenome_reference_g2: ", pangenome_reference_g2)

        print("Running mmseqs2...")
        run_mmseqs_easysearch(query=pangenome_reference_g1, target=pangenome_reference_g2, outdir=str(Path(options.outdir) / "mmseqs_clusters.m8"), tmpdir = str(Path(options.outdir) / "mmseqs_tmp"))
        print("mmseqs2 complete. Reading and filtering results...")
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

        print("Filtered mmseqs...")
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
                #print(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
                mapping_groups_1[node] = str(node_group)
            groupmapped_graph_1 = nx.relabel_nodes(graph_1, mapping_groups_1, copy=False)
        else:
            groupmapped_graph_1 = graph_1

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
        #if graph_count == 0:
        mapping_query = dict(zip(groupmapped_graph_1.nodes, groupmapped_graph_1.nodes))
        mapping_query = {key: f"{value}_query" for key, value in mapping_query.items()}
        relabeled_graph_1 = nx.relabel_nodes(groupmapped_graph_1, mapping_query, copy=False)
        #else:
        #    relabeled_graph_1 = graph_1



        ### need to do this to edges as well
        #print(relabeled_graph_1.edges)
        #print(relabeled_graph_2.edges)
        # looks like this is done automatically

        # now we can modify the tokenized code to iterate like usual, adding new node if string doesn't contain "_query"
        # and merging the nodes that both end in "_query"
        
        if options.mode == 'test':

            # read in graph_all

            graph_all = [str(Path(options.graph_all) / "final_graph.gml")]

            graph_all, isolate_names, id_mapping = load_graphs(graph_all)
            graph_all = graph_all[0]

        ### merge graphs

        merged_graph = relabeled_graph_1

        group = []
        for record in SeqIO.parse(pangenome_reference_g1, "fasta"):
            group.append({"id": record.id, "sequence": str(record.seq)})
        pan_genome_reference_merged = pd.DataFrame(group)

        #print("pan_genome_reference_merged: ", pan_genome_reference_merged)

        gene_data_all_new = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))

        #print("gene_data_all_new: ", gene_data_all_new)
        
        # merge the two sets of unique nodes into one set of unique nodes
        for node in relabeled_graph_2.nodes:
            if merged_graph.has_node(node) == True:


                if node == "group_52":
                    print("has_node group_52")
                if node == "group_52_1":
                    print("has_node group_52_1")

                # add metadata
                merged_set = list(set(relabeled_graph_2.nodes[node]["seqIDs"]) | set(relabeled_graph_1.nodes[node]["seqIDs"]))
                merged_graph.nodes[node]["seqIDs"] = merged_set

            else:

                if node == "group_52":
                    print("does not have node group_52")
                if node == "group_52_1":
                    print("does not have group_52_1")

                # add node
                merged_graph.add_node(node,
                                    seqIDs=relabeled_graph_2.nodes[node]["seqIDs"])
                                    #centroid=relabeled_graph_2.nodes[node]["centroid"]) # note: still in indSID format!!

                # add centroid from pan_genome_reference.fa to new merged reference
                # temporarily just take the sequence from any seqID in node

                # print("relabeled_graph_2.nodes[node]name :", relabeled_graph_2.nodes[node]["name"])
                #print("relabeled_graph_2.nodes[node]seqIDs :", relabeled_graph_2.nodes[node]["seqIDs"])

                #print("relabeled_graph_2.nodes[node]label :", relabeled_graph_2.nodes[node]["label"])
                #print("merged_graph.nodes[node][name]", merged_graph.nodes[node]["name"])
                #print("merged_graph.nodes[node][seqIDs]", merged_graph.nodes[node]["seqIDs"])
                
                #print("node", node)

                #if graph_count != 0:
                mapping_groups_new = dict()
                node_group = relabeled_graph_2.nodes[node].get("name", "error")
                print("node_group", node_group) # should be group_xxx from graph_2 gene data

                mapping_groups_new[node] = f'{node_group}_{graph_count+1}'
                #print("mapping_groups_new[node]", mapping_groups_new[node])
                merged_graph = nx.relabel_nodes(merged_graph, mapping_groups_new, copy=False)

                #merged_graph.nodes[f'{node_group}_{graph_count+1}']["label"] = str(f'{node_group}_{graph_count+1}')
                
                if node == "group_52":
                    print("merged_graph.nodes[f'{node_group}_{graph_count+1}'][seqIDs]", merged_graph.nodes[f'{node_group}_{graph_count+1}']["seqIDs"])
                    print("merged_graph.nodes[f'group_52']", merged_graph.nodes['group_52'])
                #print("merged_graph.nodes[f'{node_group}_{graph_count+1}'][seqIDs]", merged_graph.nodes[f'{node_group}_{graph_count+1}']["seqIDs"])
                
                # for centroids of nodes already in main graph, turn graph_1 node centroids into all_seqIDs then leave them that way forever (instead of updating with new centroids)
                # to prevent centroids from drifting away over time, and instead maintain consistency
                node_centroid = next(iter(merged_graph.nodes[f'{node_group}_{graph_count+1}']["seqIDs"]))

                #print("node_centroid", node_centroid)
                node_centroid = gene_data_all_new.loc[gene_data_all_new["clustering_id"] == node_centroid, "dna_sequence"].values
                #print("node_centroid", node_centroid)
                node_centroid = node_centroid[0] # list to string; double check that this doesn't remove centroids

                #node_name = merged_graph.nodes[node].get("name", "error")
                #label = f"{node_name}_{graph_count+1}"
                #print("label :", label)
                #merged_graph.nodes[node]["name"] = label

                node_centroid_df = pd.DataFrame([[f"{node}_{graph_count+1}", node_centroid]],
                                columns=["id", "sequence"])

                pan_genome_reference_merged = pd.concat([pan_genome_reference_merged, node_centroid_df])

        for edge in relabeled_graph_2.edges:
            
                if merged_graph.has_edge(edge[0], edge[1]):

                    # add bit to add edge metadata here

                    break

                else:

                    # note that this statement is for NODES not EDGES
                    if (edge[0] in merged_graph.nodes() == True) and (edge[1] in merged_graph.nodes() == True):
                        merged_graph.add_edge(edge[0], edge[1])
                    else:
                        print(f"Nodes in edge not present in merged graph (discarded): {edge}")
        
        if options.mode == 'test' and graph_count == (n_graphs-2):

            ### gather seqIDs to enable calculation of clustering metrics
            
            cluster_dict_merged = get_seqIDs_in_nodes(merged_graph)
            cluster_dict_all = get_seqIDs_in_nodes(graph_all)

            rand_input_merged = dict_to_2d_array(cluster_dict_merged)
            rand_input_all = dict_to_2d_array(cluster_dict_all)

            # obtain shared seq_ids
            seq_ids_1 = []


            for node in merged_graph.nodes():
                print("node", node)
                #if node in merged_graph.nodes:
                    #print(f"merged_graph.nodes[node]['seqIDs']: {merged_graph.nodes[node]['seqIDs']}")
                #seq_ids_1 += merged_graph.nodes[node]["seqIDs"]
                
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


        for node in merged_graph.nodes():
            print("node ", node)
            merged_graph.nodes[node]['seqIDs'] = ";".join(merged_graph.nodes[node]['seqIDs'])
            merged_graph.nodes[node]['name'] = merged_graph.nodes[node]['name'].removesuffix('_query')
            
        mapping_query = dict(zip(merged_graph.nodes, merged_graph.nodes))
        mapping_query = {key: f"{value.removesuffix('query')}" for key, value in mapping_query.items()}
        merged_graph_new = nx.relabel_nodes(merged_graph, mapping_query, copy=False)
        merged_graph = merged_graph_new
            
        #format_metadata_for_gml(merged_graph)
        
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

        print(f"Iteration {graph_count} complete. Merging next graph...")

    print('Finished successfully.')

    #return merge
if __name__ == "__main__":
    main()

