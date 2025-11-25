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
import gc
import multiprocessing as mp
import subprocess

# import custom functions
from .manipulate_seqids import indSID_to_allSID, get_seqIDs_in_nodes, dict_to_2d_array
from .run_mmseqs import run_mmseqs_search, mmseqs_createdb, mmseqs_concatdbs
from panaroo_functions.load_graphs import load_graphs
from panaroo_functions.write_gml_metadata import format_metadata_for_gml
from panaroo_functions.context_search import collapse_families, single_linkage
from panaroo_functions.merge_nodes import merge_node_cluster, gen_edge_iterables, gen_node_iterables, iter_del_dups, del_dups
from .relabel_nodes import relabel_nodes_preserve_attrs,sync_names
from .context_similarity import context_similarity_seq
from .context_similarity import build_ident_lookup, init_parallel, compute_scores_parallel

from .__init__ import __version__

# MUST USE FORK TO ENSURE PARALLEL COMPUTATION OF COLLAPSE SCORES DOESNT COPY GRAPH OBJECT -- LINUX DEFAULT; WINDOWS/MAC BEWARE!

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
                    default=0.7,
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
    if options.mode == 'test' and options.graph_all is None:
        logging.critical("Specifying --graph-all is required for test mode!")

    # set logging to 'debug' or 'info' (default)
    if options.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ### create outdir

    # first remove any existing files in mmseqs outdir (can cause problems)
    subprocess.run(f'rm -rf {str(options.outdir)}/mmseqs_tmp/*', shell=True, check=True, capture_output=True)

    mmseqs_dir = Path(options.outdir) / "mmseqs_tmp"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)

    # this will always point to the current combined pangenome db
    base_db = None

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

        # keep merged graph in memory instead of repeatedly reading in
        if graph_count == 0:
            graph_1, isolate_names, id_mapping = load_graphs([graph_file_1])
            graph_1 = graph_1[0]
        else:
            graph_1 = merged_graph

        graph_2, isolate_names, id_mapping = load_graphs([graph_file_2])
        graph_2 = graph_2[0]

        if options.mode == 'test':

            logging.info(f"Relabeling seqIDs to enable ARI/AMI calculation...")

            ### match clustering_ids from overall run to clustering_ids from individual runs using annotation_ids (test only)

            gene_data_all = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))
            gene_data_g2 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count+1)][0]) / "gene_data.csv"))

            if graph_count == 0:
                logging.debug("Applying gene data...")
                gene_data_g1 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count)][0]) / "gene_data.csv"))
            else:
                gene_data_g1 = None
                # not necessary because merged graph already has gene_all seqIDs mapped

            # rename column
            gene_data_all = gene_data_all.rename(columns={'clustering_id': 'clustering_id_all'})
            if graph_count == 0:
                logging.debug("Applying rename...")
                gene_data_g1 = gene_data_g1.rename(columns={'clustering_id': 'clustering_id_indiv'})
            
            gene_data_g2 = gene_data_g2.rename(columns={'clustering_id': 'clustering_id_indiv'})

            # first match by annotation ids:
            if graph_count == 0:
                logging.debug("Applying match...")
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
                logging.debug("Applying dropna...")
                matches_g1 = matches_g1.dropna()
            matches_g2 = matches_g2.dropna()

            # convert to dict for faster lookup than with loc
            if graph_count == 0:
                logging.debug("Applying gidmap...")
                gid_map_g1 = dict(zip(matches_g1['clustering_id_indiv'], matches_g1['clustering_id_all']))
            gid_map_g2 = dict(zip(matches_g2['clustering_id_indiv'], matches_g2['clustering_id_all']))

            # apply to graphs:
            if graph_count == 0:
                logging.debug("Applying ind...")
                graph_1 = indSID_to_allSID(graph_1, gid_map_g1)
            graph_2 = indSID_to_allSID(graph_2, gid_map_g2)

        # debug statement...
        logging.debug(f"--- MERGE {graph_count+1} ---")
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
        logging.info("Creating MMSeqs2 database(s)...")

        ### create mmseqs databases for faster search

        # define paths for new databases
        base_db = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+1}")
        temp_db = str(Path(options.outdir) / "mmseqs_tmp" / f"temp_db")
        
        # always create new AA database for new graph
        mmseqs_createdb(fasta=pangenome_reference_g2, outdb=temp_db, threads=options.threads, nt2aa=True)

        # create AA database for base graph on first iter only
        if graph_count == 0:
            mmseqs_createdb(fasta=pangenome_reference_g1, outdb=base_db, threads=options.threads, nt2aa=True)

        # info statement...
        logging.info("Running MMSeqs2...")

        ### run mmseqs on the two pangenome references
        run_mmseqs_search(
            targetdb=base_db,
            querydb=temp_db,
            resultdb = str(Path(options.outdir) / "mmseqs_tmp" / "resultdb"),
            resultm8 = str(Path(options.outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8"),
            tmpdir = str(Path(options.outdir) / "mmseqs_tmp"),
            threads=options.threads,
            fident=0.98,
            coverage=0.95
        )
        
        # info statement...
        logging.info("MMSeqs2 complete. Reading and filtering results...")

        # read mmseqs results
        # each "group_" refers to the centroid of that group in the pan_genomes_reference.fa
        mmseqs = pd.read_csv(str(Path(options.outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8"), sep='\t')

        ### match hits from mmseqs

        # change the second graph node names to the first graph node names for nodes that match according to mmseqs

        # make sure metrics are numeric
        mmseqs["fident"] = pd.to_numeric(mmseqs["fident"], errors='coerce')
        mmseqs["evalue"] = pd.to_numeric(mmseqs["evalue"], errors='coerce')
        mmseqs["tlen"] = pd.to_numeric(mmseqs["tlen"], errors='coerce')
        mmseqs["qlen"] = pd.to_numeric(mmseqs["qlen"], errors='coerce')

        # define length difference
        max_len = np.maximum(mmseqs['tlen'], mmseqs['qlen'])
        mmseqs["len_dif"] = 1 - (np.abs(mmseqs["tlen"] - mmseqs["qlen"]) / max_len)

        # filter for fraction nt identity >= 98% (global) and length difference <= 5%
        mmseqs = mmseqs[(mmseqs["fident"] >= 0.98) & (mmseqs["len_dif"] >= 0.95)].copy()

        ### iterate over query with each unique value of query, and pick the match with the highest fident, then highest len_dif (see calculation)
        # if still multiple matches, pick the first one

        # sort by fident (highest first), len_dif (highest first -- see calculation), and evalue (lowest first)
        mmseqs_sorted = mmseqs.sort_values(by=["fident", "len_dif", "evalue"], ascending=[False, False, True],)

        # debug statement...
        logging.debug(f" {len(mmseqs_sorted)} one-to-one hits.")
        logging.debug(f"{mmseqs_sorted}")

        # only keep the first occurrence per unique query (highest fident, lowest length difference, then smallest evalue if tie)
        mmseqs_filtered = mmseqs_sorted.drop_duplicates(subset=["query"], keep="first")
        mmseqs_filtered = mmseqs_filtered.drop_duplicates(subset=["target"], keep="first") # test if dropping target vs. query duplicates first changes results

        # debug statement...
        logging.debug(f"Filtered to {len(mmseqs_filtered)} one-to-one hits.")
        logging.debug(f"{mmseqs_filtered}")
        dups_query = mmseqs_filtered["query"].duplicated().sum()
        dups_target = mmseqs_filtered["target"].duplicated().sum()
        logging.debug(f"Remaining duplicates — query: {dups_query}, target: {dups_target}")

        # info statement...
        logging.info("Hits filtered. Mapping between graphs...")

        # in mmseqs, the first graph entered (in this case graph_1) is the target and the second entered (in this case graph_2) is the query
        # so graph_1 is our target in mmseqs and the basegraph in the tokenized merge

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

        # mapping format: dictionary with old labels (graph_2/query groups) as keys and new labels (graph_1/target) as values

        # convert df to dictionary with "query" as keys and "target" as values
        # this maps groups from graph_1 to groups from graph_2
        mapping = dict(zip(mmseqs_filtered["query"], mmseqs_filtered["target"]))

        ### to avoid matching nodes from query that have the same group_id but are not the same:
        # append all nodes in target graph with _target
        # append all target nodes in query graph with _target (for later matching)

        # this appends _target to values (graph_1/target groups)
        mapping = {key: f"{value}_target" for key, value in mapping.items()}

        # relabel query graph from old labels (keys) to new labels (values, the _target-appended graph_1 groups)
        # some of these will just be OG group_XXX (not target-appended) from graph 2; the rest will be group_XXX_target from graph 1
        relabeled_graph_2 = relabel_nodes_preserve_attrs(groupmapped_graph_2, mapping)
        relabeled_graph_2 = sync_names(relabeled_graph_2)

        # append _target to ALL nodes in target graph 
        mapping_target = dict(zip(groupmapped_graph_1.nodes, groupmapped_graph_1.nodes))
        mapping_target = {key: f"{value}_target" for key, value in mapping_target.items()}
        relabeled_graph_1 = relabel_nodes_preserve_attrs(groupmapped_graph_1, mapping_target)
        relabeled_graph_1 = sync_names(relabeled_graph_1)

        # reduce memory by removing intermediate files
        for name in [
            "mmseqs", "mmseqs_sorted", "mmseqs_filtered",
            "mapping_groups_1", "mapping_groups_2",
            "groupmapped_graph_1", "groupmapped_graph_2",
            "graph_1", "graph_2",
        ]:
            if name in locals():
                del locals()[name]
        gc.collect()

        # debug statement...
        logging.debug("MMseqs mapping (query to target_target):")
        for k,v in list(mapping.items())[:10]:
            logging.debug(f"  {k} to {v}")

        # debug statement...
        logging.debug(f"Relabeled graph_2 nodes sample: {list(relabeled_graph_2.nodes())[:10]}")
        logging.debug(f"Relabeled graph_1 (_target appended) nodes sample: {list(relabeled_graph_1.nodes())[:10]}")

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

        # rename base graph
        merged_graph = relabeled_graph_1
        del relabeled_graph_1 # delete variable name to ensure it can't be accidentally used in future

        # debug statement...
        logging.debug(f"Merging graphs. merged_graph currently has {len(merged_graph.nodes())} nodes.")
        logging.debug(f"Incoming relabeled_graph_2 has {len(relabeled_graph_2.nodes())} nodes.")

        # info statement...
        logging.info("Merging nodes...")

        # iterate, adding new node if node doesn't contain "_target"
        # and merging the nodes that both end in "_target"

        # create dictionary of nodes that will be added (not merged into existing nodes)
        mapping_groups_new = {}
        for node in relabeled_graph_2.nodes:
            if not merged_graph.has_node(node):
                mapping_groups_new[node] = f"{node}_g{graph_count+2}"

        # relabel nodes that will be added and sync their names...
        # (faster to do this just on smaller g2 instead of afterwards on merged_graph)

        if mapping_groups_new:  # only relabel if there is something to change
            relabeled_graph_2 = relabel_nodes_preserve_attrs(relabeled_graph_2, mapping_groups_new)
            relabeled_graph_2 = sync_names(relabeled_graph_2)

        # merge the two sets of unique nodes into one set of unique nodes
        for node in relabeled_graph_2.nodes:
            if merged_graph.has_node(node) == True:

                # add metadata from graph 2

                # (for centroids of nodes already in main graph, we leave them instead of updating with new centroids
                # to prevent centroids from drifting away over time, and instead maintain consistency)

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
                merged_set = merged_graph.nodes[node]["lengths"] + relabeled_graph_2.nodes[node]["lengths"]
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
                # e.g. group_XXX_g2 or group_YYY_target

                # we first found any edges that exist in the merged graph, e.g. group_XXX_target <-> group_YYY_target
                # we are now looking for group_XXX_target <-> group_YYY_g2 and group_XXX_g2 <-> group_YYY_g2 and any group_XXX_target <-> group_YYY_target only present in the new graph

                # this finds new group_XXX_target <-> group_YYY_target mappings only present in the g2 (since part of "else" statement)
                if edge[0] in merged_graph.nodes() and edge[1] in merged_graph.nodes():
                    merged_graph.add_edge(edge[0], edge[1]) # add edge
                    merged_graph.edges[edge].update(relabeled_graph_2.edges[edge]) # update with all metadata

                # these find group_XXX_target <-> group_YYY_g2
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

        # reduce memory by removing intermediate files
        if "relabeled_graph_2" in locals():
            del relabeled_graph_2
        gc.collect()

        # info statement...
        logging.info("Collapsing spurious paralogs...")

        # debug statement...
        logging.debug(f"Before collapse: {len(merged_graph.nodes())} nodes")

        # set context search parameters
        family_threshold = float(options.family_threshold)  # sequence identity threshold
        context_threshold = float(options.context_threshold)  # contextual similarity threshold 

        # write query centroid fasta (stream to reduce memory)
        def write_centroids_to_fasta(G, query_fa):
            with open(query_fa, "w") as ft:
                for node, data in G.nodes(data=True):
                    name = node
                    if name.endswith("_target") or "_target" in name:
                        # pre-existing nodes -- already in target db
                        continue
                    else:
                        # new nodes
                        seqs = data["protein"]
                        if isinstance(seqs, (list, tuple)):
                            seqs = max(seqs, key=len) # if list, pick longest sequence
                        if isinstance(seqs, str):
                            parts = seqs.split(";") # if string split on semicolon and pick longest
                            seqs = max(parts, key=len)
                        seqs = seqs.rstrip('*') # remove trailing stop
                        name = node
                        ft.write(f">{name}\n{seqs}\n")

        query_fa = Path(options.outdir) / "mmseqs_tmp" / "centroids_query.fa"
        write_centroids_to_fasta(merged_graph, query_fa)

        # info statement
        logging.info("Computing pairwise identities...")

        # info statement...
        logging.info("Creating MMSeqs2 database...")

        # create AA mmseqs database for query
        query_db = Path(options.outdir) / "mmseqs_tmp" / "query_db"
        mmseqs_createdb(fasta=query_fa, outdb=query_db, threads=options.threads, nt2aa=False)

        # info statement...
        logging.info("Running MMSeqs2...")

        # run mmseqs to get hits, keeping only those above the minimum useful threshold (family_threshold, which is LOWER than context threshold)
        run_mmseqs_search(
            targetdb=base_db,
            querydb=query_db,
            resultdb = str(Path(options.outdir) / "mmseqs_tmp" / "resultdb"),
            resultm8=str(Path(options.outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8"),
            tmpdir=str(Path(options.outdir) / "mmseqs_tmp"),
            threads=options.threads,
            fident=options.family_threshold,
            coverage=float(round((options.family_threshold * 0.95), 3))
        )

        # info statement...
        logging.info("MMSeqs2 complete. Reading and filtering results...")

        # read mmseqs results
        mmseqs = pd.read_csv(Path(options.outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8", sep="\t")

        # debugging statements...
        logging.debug(f"Unfiltered: {len(mmseqs)} one-to-one hits.")
        logging.debug(f"{mmseqs}")

        # ensure numeric columns
        for col in ["fident", "evalue", "tlen", "qlen"]:
            mmseqs[col] = pd.to_numeric(mmseqs[col], errors="coerce")

        # define length difference
        max_len = np.maximum(mmseqs["tlen"], mmseqs["qlen"])
        mmseqs["len_dif"] = 1 - (np.abs(mmseqs["tlen"] - mmseqs["qlen"]) / max_len)

        # filter for identity ≥ 70% and length difference ≥ 70%
        mmseqs = mmseqs[(mmseqs["fident"] >= family_threshold) & (mmseqs["len_dif"] >= family_threshold*0.95)].copy()

        # remove self-matches (target == query)
        mmseqs = mmseqs[mmseqs["target"] != mmseqs["query"]]

        # add _target to target node names
        # possibly pretty memory/time intensive for big dataframes, see if can make this more efficient later
        mmseqs["target"] += "_target"

        # debugging statements...
        logging.debug(f"mmseqs filtered: {len(mmseqs)} hits remaining")
        logging.debug(f"filtered mmseqs hits: {mmseqs.head()}")

        # info statement...
        logging.debug(f"Beginning context search...")

        ### compute contextual similarity

        # can still accidentally map together things from same genome by mapping a target node that's been merged into with a g2 node
        # thus we check that member sets for the nodes are disjoint (don't contain any of the same genomes)

        ident_lookup = build_ident_lookup(mmseqs)
        init_parallel(merged_graph, ident_lookup, context_threshold)
        scores = compute_scores_parallel(mmseqs, options.threads)

        # debug statement...
        logging.debug(f"scores: {scores[:5]}")

        # sort dataframe by scores
        scores_sorted = sorted(
            scores,
            key=lambda x: (x[2], x[3][0], x[3][1], x[3][2]),
            reverse=True
        )

        # debug statement...
        logging.debug(f"scores_sorted: {scores_sorted[:5]}")

        # filter accepted pairs by identity + context thresholds
        accepted_pairs = []
        for nA, nB, ident, sims in scores_sorted:
            if (
                ident >= family_threshold
                and sims[0] >= context_threshold
                and (sims[1] >= context_threshold or sims[2] >= context_threshold)
                and set(merged_graph.nodes[nA]['members']).isdisjoint(set(merged_graph.nodes[nB]['members'])) # check they do not share any members (genes within same genome will not be merged)
            ):
                accepted_pairs.append((nA, nB, ident, sims))

        # debug statement...
        logging.debug(f"accepted pairs (by context): {accepted_pairs[:10]}")

        # filter out any duplicates (in order, so best match kept)
        unique_pairs = []
        seen_nodes = set()
        for nA, nB, ident, sims in accepted_pairs:
            if nA not in seen_nodes and nB not in seen_nodes:
                unique_pairs.append((nA, nB, ident, sims))
                seen_nodes.add(nA)
                seen_nodes.add(nB)

        # debug statement...
        logging.debug(f"accepted pairs (duplicates removed): {accepted_pairs[:10]}")

        # reorder to ensure 'a' is always the node with '_target'
        reordered_pairs = []
        for a, b, ident, sims in unique_pairs:
            if "_target" in b and "_target" not in a:
                a, b = b, a
            reordered_pairs.append((a, b))
        
        # debug statement...
        logging.debug(f"accepted pairs (reordered): {accepted_pairs[:10]}")

        # reduce memory by removing intermediate files
        for name in ["mmseqs", "scores", "scores_sorted", "accepted_pairs", "unique_pairs"]:
            if name in locals():
                del locals()[name]
        gc.collect()

        # info statement...
        logging.info("Merging nodes and edges...")

        # merge the two sets of unique nodes into one set of unique nodes
        for a,b in reordered_pairs:

            # add metadata from second node

            # seqIDs
            merged_set = list(set(merged_graph.nodes[a]["seqIDs"]) | set(merged_graph.nodes[b]["seqIDs"]))
            merged_graph.nodes[a]["seqIDs"] = merged_set

            # geneIDs
            merged_set = ";".join([merged_graph.nodes[a]["geneIDs"], merged_graph.nodes[b]["geneIDs"]])
            merged_graph.nodes[a]["geneIDs"] = merged_set

            # members
            merged_set = list(set(merged_graph.nodes[a]["members"]) | set(merged_graph.nodes[b]["members"]))
            merged_graph.nodes[a]["members"] = merged_set

            # genome IDs
            merged_graph.nodes[a]["genomeIDs"] = ";".join([merged_graph.nodes[a]["genomeIDs"], merged_graph.nodes[b]["genomeIDs"]])

            # size
            size = len(merged_graph.nodes[a]["members"])

            # lengths
            merged_set = merged_graph.nodes[a]["lengths"] + merged_graph.nodes[b]["lengths"]
            merged_graph.nodes[a]["lengths"] = merged_set

            # move edges from b onto a before removing b
            for neighbor in list(merged_graph.neighbors(b)):
                #if neighbor == a:
                #    continue
                
                # get edge attributes of b
                edge_attrs = dict(merged_graph.get_edge_data(b, neighbor))
                
                # warn if don't have edge connecting neighbor
                if not merged_graph.has_edge(b, neighbor):
                    #edge_attrs = {}
                    logging.critical("neighbor not connected by edge -- this shouldn't happen!")

                if merged_graph.has_edge(a, neighbor):
                    # if the edge exists, merge metadata
                    merged_edge = merged_graph.edges[a, neighbor]
                    merged_members = set(merged_edge.get("members", [])) | set(edge_attrs.get("members", []))
                    merged_edge["members"] = list(merged_members)
                    merged_edge["size"] = len(merged_members)
                else:
                    # otherwise add the edge
                    merged_graph.add_edge(a, neighbor, **edge_attrs)
            
            # remove second node
            merged_graph.remove_node(b)

            # (don't add centroid/longCentroidID/annotation/dna/protein/hasEnd/mergedDNA/paralog/maxLenId -- keep as original for now)

        # update degrees across graph
        for node in merged_graph:
            merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])

        # debug statement...
        logging.debug(f"After collapse: {len(merged_graph.nodes())} nodes")
        
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

        ### clean node names in merged graph
        if graph_count == 0:
            # remove _target suffix, add graph count 
            # (relabel node from graph_1 group_xxx_target to group_xxx_gx)
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_target' in name:
                    new_name = re.sub(r'_target.*$', f'_g{graph_count+1}', name)
                    mapping[node_id] = new_name
                    #logging.debug(f"Changed: {name} to {new_name}")
                if '_target' not in name:
                    #logging.debug(f"Retained: {name}")
                    continue
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)
        else:
            # remove _target suffix
            # (relabel target nodes node from group_xxx_x_target to group_xxx_x)
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_target' in name:
                    new_name = re.sub(r'_target.*$', "", name)
                    mapping[node_id] = new_name
                    #logging.debug(f"Changed: {name} to {new_name}")
                if '_target' not in name:
                    #logging.debug(f"Retained: {name}")
                    continue
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)

        # debug statement...
        logging.debug("After updating node names:")
        for node in list(merged_graph.nodes())[:5]:
            logging.debug(f"  node: {node}")

        # ensure metadata written in correct format
        format_metadata_for_gml(merged_graph)

        # debug statement...
        logging.debug("After formatting metadata:")
        for node in list(merged_graph.nodes())[:5]:
            logging.debug(f"  node: {node}")

        # write new pan-genome reference to fasta (stream to reduce memory)
        #reference_out = Path(options.outdir) / f"pan_genome_reference_{graph_count+1}.fa"
        #with open(reference_out, "w") as fasta_out:
        #    for node in merged_graph.nodes():
        #        seqs = merged_graph.nodes[node]["dna"].split(";")
        #        node_centroid_seq = max(seqs, key=len)
        #        fasta_out.write(f">{node}\n{node_centroid_seq}\n")

        # after first iter, update base mmseqs database so first graph has _g1 appended node names
        if graph_count == 0:
            updated_node_names = Path(options.outdir) / "mmseqs_tmp" / f"tmp.fa"
            with open(updated_node_names, "w") as fasta_out:
                for node in merged_graph.nodes():
                    name = node
                    seqs = merged_graph.nodes[node]["protein"]
                    if isinstance(seqs, (list, tuple)):
                        seqs = max(seqs, key=len) # if list, pick longest sequence
                    if isinstance(seqs, str):
                        parts = seqs.split(";") # if string split on semicolon and pick longest
                        seqs = max(parts, key=len)
                    seqs = seqs.rstrip('*') # remove trailing stop
                    fasta_out.write(f">{node}\n{seqs}\n")

            outdb = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+2}")
            mmseqs_createdb(fasta=updated_node_names, outdb=outdb, threads=options.threads, nt2aa=False)
            base_db = outdb
            
        else:
            # write new nodes to fasta to update mmseqs db
            new_nodes_fasta = Path(options.outdir) / "mmseqs_tmp" / f"new_nodes_{graph_count+2}.fa"
            with open(new_nodes_fasta, "w") as fasta_out:
                for node in merged_graph.nodes():
                    name = node
                    if name.endswith(f'_g{graph_count+2}'):
                        seqs = merged_graph.nodes[node]["protein"]
                        if isinstance(seqs, (list, tuple)):
                            seqs = max(seqs, key=len) # if list, pick longest sequence
                        if isinstance(seqs, str):
                            parts = seqs.split(";") # if string split on semicolon and pick longest
                            seqs = max(parts, key=len)
                        seqs = seqs.rstrip('*') # remove trailing stop
                        fasta_out.write(f">{node}\n{seqs}\n")

            # update mmseqs database
            new_nodes_db = str(Path(options.outdir) / "mmseqs_tmp" / f"tmp_db")
            outdb = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+2}")

            mmseqs_createdb(fasta=new_nodes_fasta, outdb=new_nodes_db, threads=options.threads, nt2aa=False)
            mmseqs_concatdbs(db1=base_db, db2=new_nodes_db, outdb=outdb, tmpdir=str(Path(options.outdir) / "mmseqs_tmp"), threads=options.threads)
            base_db = outdb

        
        # info statement...
        logging.info('Writing merged graph to outdir...')

        # write new graph to GML
        output_path = Path(options.outdir) / f"merged_graph_{graph_count+1}.gml"
        #nx.write_gml(merged_graph, str(output_path))
        # temporarily: only write graphs w no metadata to allow for slow nx write speed
        for n in merged_graph.nodes():
            merged_graph.nodes[n].clear()
            merged_graph.nodes[n]["name"] = n
            merged_graph.nodes[n]["seqIDs"] = []
            merged_graph.nodes[n]["geneIDs"] = ''
            merged_graph.nodes[n]["members"] = []
            merged_graph.nodes[n]["genomeIDs"] = []
            merged_graph.nodes[n]["size"] = 1
            merged_graph.nodes[n]["lengths"] = []
            merged_graph.nodes[n]['longCentroidID'] = [] 
            merged_graph.nodes[n]['maxLenId'] = ''
            merged_graph.nodes[n]['centroid'] = []
            
        for u, v in merged_graph.edges():
            merged_graph[u][v].clear()
            merged_graph[u][v]["name"] = n
            merged_graph[u][v]["size"] = 1
            merged_graph[u][v]["members"] = []
            merged_graph[u][v]['genomeIDs'] = ''
        nx.write_gml(merged_graph, str(output_path))

        # write version without metadata for later visualization
        if graph_count == (n_graphs-2):
            output_path = Path(options.outdir) / f"merged_graph_{graph_count+1}_nometadata.gml"
            for n in merged_graph.nodes():
                merged_graph.nodes[n].clear()
            for u, v in merged_graph.edges():
                merged_graph[u][v].clear()
            nx.write_gml(merged_graph, str(output_path))

        # reduce memory by removing intermediate files
        for name in [
            "mapping", "mapping_target", "mapping_groups_new",
            "centroids_fa",
            "reordered_pairs",
        ]:
            if name in locals():
                del locals()[name]
        gc.collect()

        # add 1 to graph count
        graph_count += 1

        # print progress statement...
        logging.info(f"Iteration {graph_count} of {n_graphs-1} complete.")

    # info statement...
    logging.info('Finished successfully.')

    #return merge
if __name__ == "__main__":
    main()

