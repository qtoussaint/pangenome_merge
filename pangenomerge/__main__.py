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
import logging
from itertools import combinations
from edlib import align
from collections import defaultdict

# import custom functions
from .manipulate_seqids import indSID_to_allSID, get_seqIDs_in_nodes, dict_to_2d_array
from .run_mmseqs import run_mmseqs_easysearch
from panaroo_functions.load_graphs import load_graphs
from panaroo_functions.write_gml_metadata import format_metadata_for_gml
from panaroo_functions.context_search import collapse_families, single_linkage
from panaroo_functions.merge_nodes import merge_node_cluster, gen_edge_iterables, gen_node_iterables, iter_del_dups, del_dups
from .relabel_nodes import relabel_nodes_preserve_attrs,sync_names
from .context_similarity import context_similarity_seq
from .context_similarity import build_ident_lookup

from .__init__ import __version__

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

    parameters = parser.add_argument_group('Parameters')
    parameters.add_argument('--family-threshold',
                    dest='family_threshold',
                    default=0.7,
                    type=float,
                    required=False,
                    help='Sequence identity threshold for putative spurious paralogs. Default: 0.7')
    parameters.add_argument('--context-threshold',
                    dest='context_threshold',
                    default=0.9,
                    type=float,
                    required=False,
                    help='Sequence identity threshold for neighbors of putative spurious paralogs. Default: 0.9')
    
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads',
                    dest="threads",
                    default=1,
                    type=int,
                    help='Number of threads')
    other.add_argument('--debug',
                    action='store_true',
                    help="Set logging to 'debug' instead of 'info' (default)")
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # parse command line arguments
    options = get_options()

    if options.component_graphs is None and options.iterative is None:
        logging.critical("Specifying either --component-graphs or --iterative is required!")

    # set logging to 'debug' or 'info' (default)
    if options.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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

        logging.info(f"Beginning iteration {graph_count+1} of {n_graphs-1}...")
        logging.info(f"graph_file_1: {graph_file_1}")
        logging.info(f"graph_file_2: {graph_file_2}")

        graph_1, isolate_names, id_mapping = load_graphs([graph_file_1])
        graph_2, isolate_names, id_mapping = load_graphs([graph_file_2])

        graph_1 = graph_1[0]
        graph_2 = graph_2[0]

        if options.mode == 'test':

            ### match clustering_ids from overall run to clustering_ids from individual runs using annotation_ids (test only)

            gene_data_all = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))
            gene_data_g2 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count+1)][0]) / "gene_data.csv"))

            if graph_count == 0:
                logging.info("Applying gene data...")
                gene_data_g1 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count)][0]) / "gene_data.csv"))
            else:
                gene_data_g1 = None
                # not necessary because merged graph already has gene_all seqIDs mapped

            # rename column
            gene_data_all = gene_data_all.rename(columns={'clustering_id': 'clustering_id_all'})
            if graph_count == 0:
                logging.info("Applying rename...")
                gene_data_g1 = gene_data_g1.rename(columns={'clustering_id': 'clustering_id_indiv'})
            
            gene_data_g2 = gene_data_g2.rename(columns={'clustering_id': 'clustering_id_indiv'})

            # first match by annotation ids:
            if graph_count == 0:
                logging.info("Applying match...")
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
                logging.info("Applying dropna...")
                matches_g1 = matches_g1.dropna()
            matches_g2 = matches_g2.dropna()

            # convert to dict for faster lookup than with loc
            if graph_count == 0:
                logging.info("Applying gidmap...")
                gid_map_g1 = dict(zip(matches_g1['clustering_id_indiv'], matches_g1['clustering_id_all']))
            gid_map_g2 = dict(zip(matches_g2['clustering_id_indiv'], matches_g2['clustering_id_all']))

            # apply to graphs:
            if graph_count == 0:
                logging.info("Applying ind...")
                graph_1 = indSID_to_allSID(graph_1, gid_map_g1)
            graph_2 = indSID_to_allSID(graph_2, gid_map_g2)

        # debug statement...
        logging.debug(f"--- GRAPH {graph_count} LOAD ---")
        logging.debug(f"graph_1 nodes: {list(graph_1.nodes())[:20]} ... total {len(graph_1.nodes())}")
        logging.debug(f"graph_2 nodes: {list(graph_2.nodes())[:20]} ... total {len(graph_2.nodes())}")

        ### map nodes from ggcaller graphs to the COG labels in the centroid from pangenome

        ### run mmseqs2 to identify matching COGs

        # read in pangenome reference from graph 1
        if graph_count == 0:
            pangenome_reference_g1 = str(Path(graph_files.iloc[0][0]) / "pan_genome_reference.fa")
        else:
            pangenome_reference_g1 = str(Path(options.outdir) / f"pan_genome_reference_{graph_count}.fa")
        
        # read in pangenome reference from graph 2
        pangenome_reference_g2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "pan_genome_reference.fa")

        # debug statement...
        logging.debug(f"pangenome reference g1: {pangenome_reference_g1}")
        logging.debug(f"pangenome reference g2: {pangenome_reference_g2}")

        # info statement...
        logging.info("Running MMSeqs2...")

        # run mmseqs on the two pangenome references
        run_mmseqs_easysearch(query=pangenome_reference_g1, target=pangenome_reference_g2, outdir=str(Path(options.outdir) / "mmseqs_clusters.m8"), tmpdir = str(Path(options.outdir) / "mmseqs_tmp"), threads=options.threads)
        
        # info statement...
        logging.info("MMSeqs2 complete. Reading and filtering results...")

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

        # debug statement...
        logging.debug(f" {len(mmseqs_sorted)} one-to-one hits.")
        logging.debug(f"{mmseqs_sorted}")

        # only keep the first occurrence per unique target (highest fident, lowest length difference, then smallest evalue if tie)
        mmseqs_filtered = mmseqs_sorted.drop_duplicates(subset=["target"], keep="first")
        mmseqs_filtered = mmseqs_filtered.drop_duplicates(subset=["query"], keep="first") # test if dropping query vs. target duplicates first changes results

        # debug statement...
        logging.debug(f"Filtered to {len(mmseqs_filtered)} one-to-one hits.")
        logging.debug(f"{mmseqs_filtered}")
        dups_target = mmseqs_filtered["target"].duplicated().sum()
        dups_query = mmseqs_filtered["query"].duplicated().sum()
        logging.debug(f"Remaining duplicates — target: {dups_target}, query: {dups_query}")

        # info statement...
        logging.info("Hits filtered. Mapping between graphs...")

        # in mmseqs, the first graph entered (in this case graph_1) is the query and the second entered (in this case graph_2) is the target
        # so graph_1 is our query in mmseqs and the basegraph in the tokenized merge

        # when iterating over graph_2 to append to graph_1, we want to match nodes according to their graph_1 identity
        # so we need to replace all graph_2 nodes with graph_1 node ids

        ### THE NAME ("group_1") AND THE LABEL ('484') ARE DIFFERENT AND A NUMERIC STRING WILL CALL THE LABEL (not index)

        # the groups are not the same across the two graphs! we match by mmseqs (that's the whole point of this)
        # this chunk is just changing the node name from an integer to the group label of that node (which is originally just
        # metadata within the graph)
        # it doesn't map anything between the two graphs

        # change integer node labels into group labels from 'name' attribute (graph 1)
        if graph_count == 0:

            # map 'name' attribute to integer node label
            mapping_groups_1 = dict()
            for node in graph_1.nodes():
                node_group = graph_1.nodes[node].get("name", "error")
                #logging.debug(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
                mapping_groups_1[node] = str(node_group)

            # debug statement...
            logging.debug("mapping_groups_1 sample:")
            for k,v in list(mapping_groups_1.items())[:10]:
                logging.debug(f"  {k} to {v}")

            # relabel nodes
            groupmapped_graph_1 = relabel_nodes_preserve_attrs(graph_1, mapping_groups_1)

            # debug statement...
            logging.debug(f"After relabel_nodes_preserve_attrs() graph_1 node sample: {list(groupmapped_graph_1.nodes())[:10]}")
        
        else:

            # map 'name' attribute to integer node label
            mapping_groups_1 = dict()
            for node in graph_1.nodes():
                node_group = graph_1.nodes[node].get("name", "error")
                #logging.debug(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
                mapping_groups_1[node] = str(node_group)

            # debug statement...
            logging.debug("mapping_groups_1 sample:")
            for k,v in list(mapping_groups_1.items())[:10]:
                logging.debug(f"  {k} to {v}")
            
            # relabel nodes
            groupmapped_graph_1 = relabel_nodes_preserve_attrs(graph_1, mapping_groups_1)

            # debug statement...
            logging.debug(f"After relabel_nodes_preserve_attrs() graph_1 node sample: {list(groupmapped_graph_1.nodes())[:10]}")

        # change integer node labels into group labels from 'name' attribute (graph 2)
        # map 'name' attribute to integer node label
        mapping_groups_2 = dict()
        for node in graph_2.nodes():
            node_group = graph_2.nodes[node].get("name", "error")
            #logging.debug(f"graph: 2, node_index_id: {node}, node_group_id: {node_group}")
            mapping_groups_2[node] = str(node_group)

        # debug statement...
        logging.debug("mapping_groups_2 sample:")
        for k,v in list(mapping_groups_2.items())[:10]:
            logging.debug(f"  {k} to {v}")

        # relabel nodes
        groupmapped_graph_2 = relabel_nodes_preserve_attrs(graph_2, mapping_groups_2)

        # debug statement...
        logging.debug(f"After relabel_nodes_preserve_attrs() graph_2 node sample: {list(groupmapped_graph_2.nodes())[:10]}")

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
        # some of these will just be OG group_XXX (not query-appended) from graph 2; the rest will be group_XXX_query from graph 1
        relabeled_graph_2 = relabel_nodes_preserve_attrs(groupmapped_graph_2, mapping)
        relabeled_graph_2 = sync_names(relabeled_graph_2)

        # append _query to ALL nodes in query graph 
        mapping_query = dict(zip(groupmapped_graph_1.nodes, groupmapped_graph_1.nodes))
        mapping_query = {key: f"{value}_query" for key, value in mapping_query.items()}
        relabeled_graph_1 = relabel_nodes_preserve_attrs(groupmapped_graph_1, mapping_query)
        relabeled_graph_1 = sync_names(relabeled_graph_1)

        # debug statement...
        logging.debug("MMseqs mapping (target to query_query):")
        for k,v in list(mapping.items())[:10]:
            logging.debug(f"  {k} to {v}")

        # debug statement...
        logging.debug(f"Relabeled graph_2 nodes sample: {list(relabeled_graph_2.nodes())[:10]}")
        logging.debug(f"Relabeled graph_1 (_query appended) nodes sample: {list(relabeled_graph_1.nodes())[:10]}")

        # read in graph of all isolates (for ARI/AMI calculation)
        if options.mode == 'test':
            graph_all = [str(Path(options.graph_all) / "final_graph.gml")]
            graph_all, isolate_names, id_mapping = load_graphs(graph_all)
            graph_all = graph_all[0]

        # info statement...
        logging.info("Updating graph metadata to prepare for merge...")
        
        ### add suffix to relevant metadata to be able to identify which graph they refer to later
        # if other bits are too slow, replacing looping over nodes with the nodes.values method shown here

        # NODES

        for node_data in relabeled_graph_2.nodes.values():

            node_data['members'] = [f"{member}_g{graph_count+2}" for member in node_data['members']] # list

            node_data['genomeIDs'] = ";".join(node_data['members']) # str

            if options.mode != 'test':

                seqid_set = {f"{seqid}{f'_g{graph_count+2}'}" for seqid in node_data['seqIDs']}
                node_data['seqIDs'] = seqid_set # set

                geneids = node_data['geneIDs'].split(";")
                geneids = [f"{gid}_g{graph_count+2}" for gid in geneids]
                node_data['geneIDs'] = ";".join(geneids) # str

                node_data['longCentroidID'].append(f'from_g{graph_count+2}') #list

                node_data['maxLenId'] = str(node_data['maxLenId']) + f'_g{graph_count+2}' # int

                node_data['centroid'] = [f"{centroid}_g{graph_count+2}" for centroid in node_data['centroid']] # list

        if graph_count == 0: # if first iteration, do the same to the base graph

            for node_data in relabeled_graph_1.nodes.values():

                node_data['members'] = [f"{member}_g{graph_count+1}" for member in node_data['members']] # list

                node_data['genomeIDs'] = ";".join(node_data['members']) # str

            if options.mode != 'test':

                seqid_set = {f"{seqid}{f'_g{graph_count+1}'}" for seqid in node_data['seqIDs']}
                node_data['seqIDs'] = seqid_set # set

                node_data['longCentroidID'].append(f'from_g{graph_count+1}') #list

                node_data['centroid'] = [f"{centroid}_g{graph_count+1}" for centroid in node_data['centroid']] # list

                node_data['maxLenId'] = str(node_data['maxLenId']) + f'_g{graph_count+1}' # int

                geneids = node_data['geneIDs'].split(";")
                geneids = [f"{gid}_g{graph_count+1}" for gid in geneids]
                node_data['geneIDs'] = ";".join(geneids) # str

        # EDGES

        # edge attributes: size (n members), members (list), genomeIDs (semicolon-separated string)

        for edge in relabeled_graph_2.edges:
            
            # members
            relabeled_graph_2.edges[edge]['members'] = [str(member) + f"_g{graph_count+2}" for member in relabeled_graph_2.edges[edge]['members']] # add underscore w graph count+2 to members of g2

            # genome IDs (assuming genomeIDs are always the same as members):
            relabeled_graph_2.edges[edge]['genomeIDs'] = ";".join(relabeled_graph_2.edges[edge]['members'])

            # size
            relabeled_graph_2.edges[edge]['size'] = str(len(relabeled_graph_2.edges[edge]['members']))

        if graph_count == 0: # if first iteration, do the same to the base graph

            for edge in relabeled_graph_1.edges:
                
                # members
                relabeled_graph_1.edges[edge]['members'] = [str(member) + f"_g{graph_count+1}" for member in relabeled_graph_1.edges[edge]['members']] # add underscore with graph count+1 to members of g1

                # genome IDs (assuming genomeIDs are always the same as members):
                relabeled_graph_1.edges[edge]['genomeIDs'] = ";".join(relabeled_graph_1.edges[edge]['members'])

                # size
                relabeled_graph_1.edges[edge]['size'] = str(len(relabeled_graph_1.edges[edge]['members']))

        ### merge graphs

        # info statement...
        logging.info("Beginning graph merge...")

        # make copy of base graph to avoid accidentally changing base graph
        merged_graph = relabeled_graph_1.copy()

        # debug statement...
        logging.debug(f"Merging graphs. merged_graph currently has {len(merged_graph.nodes())} nodes.")
        logging.debug(f"Incoming relabeled_graph_2 has {len(relabeled_graph_2.nodes())} nodes.")

        # read in pangenome reference for base graph
        #group = []
        #for record in SeqIO.parse(pangenome_reference_g1, "fasta"):
        #    group.append({"id": record.id, "sequence": str(record.seq)})
        #pan_genome_reference_merged = pd.DataFrame(group)

        # debug statement...
        #logging.debug(f"pan_genome_reference_merged: {pan_genome_reference_merged}")

        # update the pangenome reference group labels from the base graph the first time it's used
        #if graph_count == 0:
        #    pan_genome_reference_merged["id"] = pan_genome_reference_merged["id"].astype(str) + f"_{graph_count+1}"

        # debug statement...
        #logging.debug(f"pan_genome_reference_merged_updated: {pan_genome_reference_merged}")

        # info statement...
        logging.info("Merging nodes...")

        # iterate, adding new node if node doesn't contain "_query"
        # and merging the nodes that both end in "_query"

        # merge the two sets of unique nodes into one set of unique nodes
        for node in relabeled_graph_2.nodes:
            if merged_graph.has_node(node) == True:

                # add metadata from graph 2

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


                ### add centroid from g2 pan_genome_reference.fa to merged pan_genome_reference.fa

                # (for centroids of nodes already in main graph, we leave them instead of updating with new centroids
                # to prevent centroids from drifting away over time, and instead maintain consistency)

                # relabel node from graph_2 group_xxx to group_xxx_graphcount
                mapping_groups_new = {node: f"{node}_g{graph_count+2}"}
                merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping_groups_new)
                merged_graph = sync_names(merged_graph)  # could sync names at end of all merges if slow

                # get updated node name
                new_node_name = f"{node}_g{graph_count+2}"

                # get node centroid sequence
                #node_centroid_seq = relabeled_graph_2.nodes[node]["dna"][0] \
                #    if isinstance(relabeled_graph_2.nodes[node]["dna"], list) \
                #    else relabeled_graph_2.nodes[node]["dna"]

                # update merged pangenome reference
                #pan_genome_reference_merged.loc[len(pan_genome_reference_merged)] = [new_node_name, node_centroid_seq]

        # info statement...
        logging.info("Merging edges...")

        # debug statement...
        logging.debug(f"After merge but before edge merge: merged_graph node sample: {list(merged_graph.nodes())[:20]}")
        logging.debug(f"After merge but before edge merge: {len(merged_graph.nodes())} nodes")

        # add in metadata from merged edges; add in new edges

        for edge in relabeled_graph_2.edges:
            
            if merged_graph.has_edge(edge[0], edge[1]):

                # add edge metadata from graph 2 to merged graph
                # edge attributes: size (n members), members (list), genomeIDs (semicolon-separated string)

                unadded_metadata = relabeled_graph_2.edges[edge]

                # members
                merged_graph.edges[edge]['members'].extend(unadded_metadata['members']) # combine members

                # genome IDs (assuming genomeIDs are always the same as members):
                merged_graph.edges[edge]['genomeIDs'] = ";".join(merged_graph.edges[edge]['members'])
                
                # size
                merged_graph.edges[edge]['size'] = str(len(merged_graph.edges[edge]['members']))

            else:

                # note that this statement is for NODES not EDGES

                # edge[0] and edge[1] are node names from nodes in g2
                # e.g. group_XXX_g2 or group_YYY_query

                # we first found any edges that exist in the merged graph, e.g. group_XXX_query <-> group_YYY_query
                # we are now looking for group_XXX_query <-> group_YYY_g2 and group_XXX_g2 <-> group_YYY_g2 and any group_XXX_query <-> group_YYY_query only present in the new graph

                # this finds new group_XXX_query <-> group_YYY_query mappings only present in the g2 (since part of "else" statement)
                if edge[0] in merged_graph.nodes() and edge[1] in merged_graph.nodes():
                    merged_graph.add_edge(edge[0], edge[1]) # add edge
                    merged_graph.edges[edge].update(relabeled_graph_2.edges[edge]) # update with all metadata

                # these find group_XXX_query <-> group_YYY_g2
                if edge[0] in merged_graph.nodes() and edge[1] not in merged_graph.nodes():

                    if f"{edge[1]}_g{graph_count+2}" in merged_graph.nodes():
                        merged_graph.add_edge(edge[0], f"{edge[1]}_g{graph_count+2}") # add edge
                        merged_graph.edges[edge[0], f"{edge[1]}_g{graph_count+2}"].update(relabeled_graph_2.edges[edge]) # update with all metadata
                    else:
                        logging.error(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

                if edge[0] not in merged_graph.nodes() and edge[1] in merged_graph.nodes():

                    if f"{edge[0]}_g{graph_count+2}" in merged_graph.nodes():
                        merged_graph.add_edge(f"{edge[0]}_g{graph_count+2}", edge[1]) # add edge
                        merged_graph.edges[f"{edge[0]}_g{graph_count+2}", edge[1]].update(relabeled_graph_2.edges[edge]) # update with all metadata
                    else:
                        logging.error(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

                # this finds group_XXX_g2 <-> group_YYY_g2
                if edge[0] not in merged_graph.nodes() and edge[1] not in merged_graph.nodes():
                    
                    if f"{edge[0]}_g{graph_count+2}" in merged_graph.nodes() and f"{edge[1]}_g{graph_count+2}" in merged_graph.nodes():
                        merged_graph.add_edge(f"{edge[0]}_g{graph_count+2}", f"{edge[1]}_g{graph_count+2}") # add edge
                        merged_graph.edges[f"{edge[0]}_g{graph_count+2}", f"{edge[1]}_g{graph_count+2}"].update(relabeled_graph_2.edges[edge]) # update with all metadata
                    else: 
                        logging.error(f"Nodes in edge not present in merged graph (ghost nodes): {edge}")

        # update degrees across graph
        for node in merged_graph:
            merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])

        # debug statement...
        logging.debug(f"After merge and edge merge: merged_graph node sample: {list(merged_graph.nodes())[:20]}")
        logging.debug(f"After merge and edge merge: {len(merged_graph.nodes())} nodes")

        # info statement...
        logging.info("Collapsing spurious paralogs...")

        # debug statement...
        logging.debug(f"Before collapse: {len(merged_graph.nodes())} nodes")

        # set context search parameters
        family_threshold = float(options.family_threshold)  # sequence identity threshold
        context_threshold = float(options.context_threshold)  # contextual similarity threshold 

        # one centroid to one sequence
        centroid_to_seq = {}
        for node, data in merged_graph.nodes(data=True):
            c = node

            seqs = merged_graph.nodes[node]["protein"]
            node_centroid_seq = max(seqs, key=len)

            centroid_to_seq[c] = node_centroid_seq

        # debug statement...
        logging.debug(f"centroid_to_seq: {list(centroid_to_seq.items())[:10]}")

        def write_fasta(seqs_dict, filename):
            with open(filename, "w") as f:
                for name, seq in seqs_dict.items():
                    f.write(f">{name}\n{seq}\n")

        # create temporary FASTA files
        query_fa = Path(options.outdir) / "centroids_query.fa"
        target_fa = Path(options.outdir) / "centroids_target.fa"

        write_fasta(centroid_to_seq, query_fa)
        write_fasta(centroid_to_seq, target_fa)

        # info statement
        logging.info("Computing pairwise identities...")

        # info statement...
        logging.info("Running MMSeqs2...")

        # run mmseqs on the two pangenome references
        run_mmseqs_easysearch(query=query_fa, target=target_fa, outdir=str(Path(options.outdir) / "mmseqs_clusters.m8"), tmpdir = str(Path(options.outdir) / "mmseqs_tmp"), threads=options.threads)
        
        # info statement...
        logging.info("MMSeqs2 complete. Reading and filtering results...")

        # read mmseqs results
        mmseqs = pd.read_csv(Path(options.outdir) / "mmseqs_clusters.m8", sep="\t")

        # ensure numeric columns
        for col in ["fident", "evalue", "tlen", "qlen", "nident"]:
            mmseqs[col] = pd.to_numeric(mmseqs[col], errors="coerce")

        # define length difference
        max_len = np.maximum(mmseqs["tlen"], mmseqs["qlen"])
        mmseqs["len_dif"] = 1 - (np.abs(mmseqs["tlen"] - mmseqs["qlen"]) / max_len)

        # filter for identity ≥ 70% and length difference ≥ 50%
        mmseqs = mmseqs[(mmseqs["fident"] >= family_threshold) & (mmseqs["len_dif"] >= 0.50)].copy()

        # remove self-matches (query == target)
        mmseqs = mmseqs[mmseqs["query"] != mmseqs["target"]]

        # remove rows where both have "_query"
        mmseqs = mmseqs[~((mmseqs["target"].str.contains("_query")) & (mmseqs["query"].str.contains("_query")))]

        # remove rows where NEITHER has "_query"
        mmseqs = mmseqs[
            ~((~mmseqs["target"].str.contains("_query")) & (~mmseqs["query"].str.contains("_query")))
        ]

        print(f"mmseqs filtered: {len(mmseqs)} hits remaining")

        # optional debugging
        logging.debug(mmseqs.head())

        ### compute contextual similarity

        # can still accidentally map together things from same genome by mapping a query node that's been merged into with a g2 node
        # will need to check that member sets for the nodes are disjoint (don't contain any of the same genomes)

        ident_lookup = build_ident_lookup(mmseqs)

        scores = []
        for row in mmseqs.itertuples(index=False):
            nA = row.query
            nB = row.target
            ident = row.fident

            s1 = context_similarity_seq(merged_graph, nA, nB, ident_lookup, depth=1)
            s2 = s1 if s1 >= 0.9 else context_similarity_seq(merged_graph, nA, nB, ident_lookup, depth=2)
            s3 = s2 if s2 >= 0.9 else context_similarity_seq(merged_graph, nA, nB, ident_lookup, depth=3)
            sims = [s1, s2, s3]

            scores.append((nA, nB, ident, sims))

        logging.debug(f"scores: {scores}")

        # sort dataframe by scores
        scores_sorted = sorted(
            scores,
            key=lambda x: (x[2], x[3][0], x[3][1], x[3][2]),
            reverse=True
        )

        # debug statement...
        logging.debug(f"scores_sorted: {scores_sorted}")

        # filter accepted pairs by identity + context thresholds
        accepted_pairs = []
        for nA, nB, ident, sims in scores_sorted:
            if (
                ident >= family_threshold
                and sims[0] >= context_threshold
                and (sims[1] >= context_threshold or sims[2] >= context_threshold)
                and set(merged_graph.nodes[nA]['members']).isdisjoint(set(merged_graph.nodes[nB]['members'])) # check they do not share any members (gene from within same genome will not be merged)
            ):
                accepted_pairs.append((nA, nB, ident, sims))

        # filter out any duplicates (in order, so best match kept)
        unique_pairs = []
        seen_nodes = set()
        for nA, nB, ident, sims in accepted_pairs:
            if nA not in seen_nodes and nB not in seen_nodes:
                unique_pairs.append((nA, nB, ident, sims))
                seen_nodes.add(nA)
                seen_nodes.add(nB)

        # reorder to ensure 'a' is always the node with '_query'
        reordered_pairs = []
        for a, b, ident, sims in unique_pairs:
            if "_query" in b and "_query" not in a:
                a, b = b, a
            reordered_pairs.append((a, b))

        # info statement
        logging.info("Copy graph file...")

        # copy to create new graph object
        collapsed_merged_graph = merged_graph.copy()

        # info statement...
        logging.info("Merging nodes and edges...")

        # create set of nodes to drop from the pangenome reference (since they're being merged)
        #to_drop = set()

        # merge the two sets of unique nodes into one set of unique nodes
        for a,b in reordered_pairs:

            # add metadata from second node

            # seqIDs
            merged_set = list(set(collapsed_merged_graph.nodes[a]["seqIDs"]) | set(collapsed_merged_graph.nodes[b]["seqIDs"]))
            collapsed_merged_graph.nodes[a]["seqIDs"] = merged_set

            # geneIDs
            merged_set = ";".join([collapsed_merged_graph.nodes[a]["geneIDs"], collapsed_merged_graph.nodes[b]["geneIDs"]])
            collapsed_merged_graph.nodes[a]["geneIDs"] = merged_set

            # members
            merged_set = list(set(collapsed_merged_graph.nodes[a]["members"]) | set(collapsed_merged_graph.nodes[b]["members"]))
            collapsed_merged_graph.nodes[a]["members"] = merged_set

            # genome IDs
            collapsed_merged_graph.nodes[a]["genomeIDs"] = ";".join([collapsed_merged_graph.nodes[a]["genomeIDs"], collapsed_merged_graph.nodes[b]["genomeIDs"]])

            # size
            size = len(collapsed_merged_graph.nodes[a]["members"])

            # lengths
            merged_set = collapsed_merged_graph.nodes[a]["lengths"] + collapsed_merged_graph.nodes[b]["lengths"]
            collapsed_merged_graph.nodes[a]["lengths"] = merged_set

            # note to remove node from pangenome reference
            #to_drop.add(b)

            # move edges from b to a before removing b
            for neighbor in list(collapsed_merged_graph.neighbors(b)):
                if neighbor == a:
                    continue
                if collapsed_merged_graph.has_edge(b, neighbor):
                    edge_attrs = dict(collapsed_merged_graph.get_edge_data(b, neighbor))
                else:
                    edge_attrs = {}

                if not collapsed_merged_graph.has_edge(a, neighbor):
                    collapsed_merged_graph.add_edge(a, neighbor, **edge_attrs)
                else:
                    merged_edge = collapsed_merged_graph.edges[a, neighbor]
                    merged_members = set(merged_edge.get("members", [])) | set(edge_attrs.get("members", []))
                    merged_edge["members"] = list(merged_members)
                    merged_edge["size"] = len(merged_members)
            
            # remove second node
            collapsed_merged_graph.remove_node(b)

            # (don't add centroid/longCentroidID/annotation/dna/protein/hasEnd/mergedDNA/paralog/maxLenId -- keep as original for now)

        # also need to remove pangenome reference centroids from new nodes that got merged during collapse
        #pan_genome_reference_merged.drop(
        #    pan_genome_reference_merged.index[pan_genome_reference_merged["id"].isin(to_drop)],
        #    inplace=True
        #)

        # update degrees across graph
        for node in collapsed_merged_graph:
            collapsed_merged_graph.nodes[node]["degrees"] = int(collapsed_merged_graph.degree[node])

        # debug statement...
        logging.debug(f"After collapse: {len(collapsed_merged_graph.nodes())} nodes")

        # info statement
        logging.info("Copy graph file...")

        # write graph back to merged_graph
        merged_graph = collapsed_merged_graph.copy()
        
        # calculate clustering performance (if test mode)
        if options.mode == 'test' and graph_count == (n_graphs-2):

            # info statement...
            logging.info("Calculating adjusted Rand index (ARI) and adjusted mutual information (AMI)...")

            ### gather seqIDs to enable calculation of clustering metrics
            
            cluster_dict_merged = get_seqIDs_in_nodes(merged_graph)
            cluster_dict_all = get_seqIDs_in_nodes(graph_all)

            rand_input_merged = dict_to_2d_array(cluster_dict_merged)
            rand_input_all = dict_to_2d_array(cluster_dict_all)

            # obtain shared seq_ids
            seq_ids_1 = []

            for node in merged_graph.nodes():
                seq_ids_1 += merged_graph.nodes[node]["seqIDs"]
                
            seq_ids_2 = []
            for node in graph_all.nodes():
                seq_ids_2 += graph_all.nodes[node]["seqIDs"]
                
            seq_ids_1 = set(seq_ids_1)
            seq_ids_2 = set(seq_ids_2)
                
            # take intersection
            common_seq_ids = seq_ids_1 & seq_ids_2 
            
            # print how many seq_ids were excluded
            only_in_graph_1 = seq_ids_1 - seq_ids_2
            only_in_graph_2 = seq_ids_2 - seq_ids_1
            logging.info(f"shared seqIDs: {len(common_seq_ids)}")
            logging.info(f"seqIDs only in merged (excluded): {len(only_in_graph_1)}")
            logging.info(f"seqIDs only in all (excluded): {len(only_in_graph_2)}")
            logging.debug(f"seqIDs only in merged (excluded): {only_in_graph_1}")
            logging.debug(f"seqIDs only in all (excluded): {only_in_graph_2}")

            rand_input_merged_filtered = rand_input_merged.loc[:, rand_input_merged.loc[0].isin(common_seq_ids)]
            rand_input_all_filtered = rand_input_all.loc[:, rand_input_all.loc[0].isin(common_seq_ids)]
            
            # get desired value order from row 0 of rand_input_all_filtered
            desired_order = list(rand_input_all_filtered.iloc[0])

            # create mapping from row 0 values in rand_input_merged_filtered to column names
            val_to_col = {val: col for col, val in zip(rand_input_merged_filtered.columns, rand_input_merged_filtered.iloc[0])}

            # reorder columns based on desired value order
            columns_in_order = [val_to_col[val] for val in desired_order if val in val_to_col]

            # apply the column reordering
            rand_input_merged_filtered = rand_input_merged_filtered[columns_in_order]
            
            # put sorted clusters into Rand index
            ri = rand_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            logging.info(f"Rand Index: {ri}")

            ari = adjusted_rand_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            logging.info(f"Adjusted Rand Index: {ari}")

            # put sorted clusters into mutual information
            mutual_info = mutual_info_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            logging.info(f"Mutual Information: {mutual_info}")

            adj_mutual_info = adjusted_mutual_info_score(rand_input_all_filtered.iloc[1], rand_input_merged_filtered.iloc[1])
            logging.info(f"Adjusted Mutual Information: {adj_mutual_info}")

        # info statement...
        logging.info("Merge complete. Preparing attribute metadata for export...")

        # CHECK IF ADDING GRAPH COUNT TO NODE NAME HERE IS RIGHT

        ### clean node names in merged graph
        if graph_count == 0:
            # remove _query suffix, add graph count 
            # (relabel node from graph_1 group_xxx_query to group_xxx_x)
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_query' in name:
                    new_name = re.sub(r'_query.*$', f'_g{graph_count+1}', name)
                    #new_name = name.replace('_query', f'_{graph_count+1}')
                    mapping[node_id] = new_name
                    logging.debug(f"Changed: {name} to {new_name}")
                if '_query' not in name:
                    logging.debug(f"Retained: {name}")
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)
        else:
            # remove _query suffix
            # (relabel query nodes node from group_xxx_x_query to group_xxx_x)
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_query' in name:
                    new_name = re.sub(r'_query.*$', "", name)
                    #new_name = name.replace('_query', "")
                    mapping[node_id] = new_name
                    logging.debug(f"Changed: {name} to {new_name}")
                if '_query' not in name:
                    logging.debug(f"Retained: {name}")
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)

        # debug statement...
        logging.debug("After updating node names:")
        for node in list(merged_graph.nodes())[:20]:
            logging.debug(f"  node: {node}")

        # ensure metadata written in correct format
        format_metadata_for_gml(merged_graph)

        # debug statement...
        logging.debug("After formatting metadata:")
        for node in list(merged_graph.nodes())[:20]:
            logging.debug(f"  node: {node}")
        
        # info statement...
        logging.info('Writing merged graph to outdir...')

        # write new graph to GML
        output_path = Path(options.outdir) / f"merged_graph_{graph_count+1}.gml"
        nx.write_gml(merged_graph, str(output_path))

        # write new pan-genome references to fasta
        reference_out = str(Path(options.outdir) / f"pan_genome_reference_{graph_count+1}.fa")

        ######## change so that pangenome reference is only based on final graph nodes!

        pan_genome_reference_merged = pd.DataFrame(columns=["id", "sequence"])

        for node in merged_graph.nodes():

            # get node centroid sequence
            seqs = merged_graph.nodes[node]["dna"].split(";")
            node_centroid_seq = max(seqs, key=len)

            # update merged pangenome reference
            pan_genome_reference_merged.loc[len(pan_genome_reference_merged)] = [node, node_centroid_seq]

        ########

        with open(reference_out, "w") as fasta_out:
            for _, row in pan_genome_reference_merged.iterrows():
                fasta_out.write(f">{row['id']}\n{row['sequence']}\n")

        # debug statement...
        graph_node_ids = set(merged_graph.nodes())
        ref_ids = set(pan_genome_reference_merged["id"])
        missing_in_ref = graph_node_ids - ref_ids
        extra_in_ref = ref_ids - graph_node_ids
        if missing_in_ref:
            logging.warning(f"{len(missing_in_ref)} nodes in graph but missing from reference")
            logging.warning(f"Examples: {list(missing_in_ref)[:10]}")
        if extra_in_ref:
            logging.warning(f"{len(extra_in_ref)} IDs in reference but missing in graph")
            logging.warning(f"Examples: {list(extra_in_ref)[:10]}")

        # add 1 to graph count
        graph_count += 1

        # print progress statement...
        logging.info(f"Iteration {graph_count} of {n_graphs-1} complete.")

    # info statement...
    logging.info('Finished successfully.')

    #return merge
if __name__ == "__main__":
    main()

