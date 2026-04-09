import subprocess
import tempfile
import os
from collections import defaultdict, Counter
import networkx as nx
from Bio.Seq import Seq
import edlib
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
import math
from tqdm import tqdm


def run_cdhit(
    input_file,
    output_file,
    id=0.95,
    n_cpu=1,
    s=0.0,  # length difference cutoff (%), default 0.0
    aL=0.0,  # alignment coverage for the longer sequence
    AL=99999999,  # alignment coverage control for the longer sequence
    aS=0.0,  # alignment coverage for the shorter sequence
    AS=99999999,  # alignment coverage control for the shorter sequence
    accurate=True,  # use the slower but more accurate options
    use_local=False,  #whether to use local or global sequence alignment
    word_length=None,
    min_length=None,
    quiet=False):

    cmd = "cd-hit"
    cmd += " -T " + str(n_cpu)
    cmd += " -i " + input_file
    cmd += " -o " + output_file
    cmd += " -c " + str(id)
    cmd += " -s " + str(s)
    cmd += " -aL " + str(aL)
    cmd += " -AL " + str(AL)
    cmd += " -aS " + str(aS)
    cmd += " -AS " + str(AS)
    cmd += " -M 0 -d 999"

    if use_local:
        cmd += " -G 0"

    if accurate:
        cmd += " -g 1 -n 2"

    if (word_length is not None) and (not accurate):
        cmd += " -n " + str(word_length)

    if min_length is not None:
        cmd += " -l " + str(min_length)

    if not quiet:
        print("running cmd: " + cmd)
    else:
        cmd += " > /dev/null"

    subprocess.run(cmd, shell=True, check=True)

    return


def run_cdhit_est(
    input_file,
    output_file,
    id=0.99,
    n_cpu=1,
    s=0.0,  # length difference cutoff (%), default 0.0
    aL=0.0,  # alignment coverage for the longer sequence
    AL=99999999,  # alignment coverage control for the longer sequence
    aS=0.0,  # alignment coverage for the shorter sequence
    AS=99999999,  # alignment coverage control for the shorter sequence
    accurate=True,  # use the slower but more accurate options
    use_local=False,  #whether to use local or global sequence alignment
    strand=1,  # default do both +/+ & +/- alignments if set to 0, only +/+
    print_aln=False,  # print alignment overlap in cluster file
    word_length=None,
    mask=True,
    quiet=False):

    cmd = "cd-hit-est"
    cmd += " -T " + str(n_cpu)
    cmd += " -i " + input_file
    cmd += " -o " + output_file
    cmd += " -c " + str(id)
    cmd += " -s " + str(s)
    cmd += " -aL " + str(aL)
    cmd += " -AL " + str(AL)
    cmd += " -aS " + str(aS)
    cmd += " -AS " + str(AS)
    cmd += " -r " + str(strand)
    cmd += " -M 0 -d 999"

    if mask:
        cmd += " -mask NX"

    if use_local:
        cmd += " -G 0"

    if accurate:
        cmd += " -g 1 -n 6"

    if (word_length is not None) and (not accurate):
        cmd += " -n " + str(word_length)

    if print_aln:
        cmd += " -p 1"

    if not quiet:
        print("running cmd: " + cmd)
    else:
        cmd += " > /dev/null"

    subprocess.run(cmd, shell=True, check=True)

    return


def is_valid(G, node, cluster):
    found = True
    for n in cluster:
        if len(G.nodes[node]['members'] & G.nodes[n]['members']) > 0:
            found = False
    return found


def iterative_cdhit(
    G,
    outdir,
    dna=False,
    s=0.0,  # length difference cutoff (%), default 0.0
    aL=0.0,  # alignment coverage for the longer sequence
    AL=99999999,  # alignment coverage control for the longer sequence
    aS=0.0,  # alignment coverage for the shorter sequence
    AS=99999999,  # alignment coverage control for the shorter sequence
    accurate=True,  # use the slower but more accurate options
    use_local=False,  #whether to use local or global sequence alignment
    strand=1,  # default do both +/+ & +/- alignments if set to 0, only +/+
    quiet=False,
    word_length=None,
    thresholds=[0.99, 0.95, 0.90, 0.85, 0.8, 0.75, 0.7],
    n_cpu=1):

    # create the files we will need
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, dir=outdir)
    temp_input_file.close()
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, dir=outdir)
    temp_output_file.close()

    centroid_to_seq = {}
    for node in G.nodes():
        if dna:
            for sid, seq in zip(G.nodes[node]["centroid"],
                                G.nodes[node]["dna"]):
                centroid_to_seq[sid] = seq
        else:
            for sid, seq in zip(G.nodes[node]["centroid"],
                                G.nodes[node]["protein"]):
                centroid_to_seq[sid] = seq

    clusters = []
    with open(temp_input_file.name, 'w') as outfile:
        for centroid in centroid_to_seq:
            clusters.append([centroid])
            outfile.write(">" + str(centroid) + "\n")
            outfile.write(centroid_to_seq[centroid] + "\n")

    for cid in thresholds:
        # run cd-hit
        if dna:
            run_cdhit_est(input_file=temp_input_file.name,
                          output_file=temp_output_file.name,
                          id=cid,
                          s=s,
                          aL=aL,
                          AL=AL,
                          aS=AS,
                          accurate=accurate,
                          use_local=use_local,
                          strand=strand,
                          quiet=quiet,
                          word_length=word_length,
                          n_cpu=n_cpu)
        else:
            run_cdhit(input_file=temp_input_file.name,
                      output_file=temp_output_file.name,
                      id=cid,
                      s=s,
                      aL=aL,
                      AL=AL,
                      aS=AS,
                      accurate=accurate,
                      use_local=use_local,
                      word_length=word_length,
                      quiet=quiet,
                      n_cpu=n_cpu)

        # process the output
        temp_clusters = []
        with open(temp_output_file.name + ".clstr", 'r') as infile:
            c = []
            for line in infile:
                if line[0] == ">":
                    temp_clusters.append(c)
                    c = []
                else:
                    c.append(line.split(">")[1].split("...")[0])
            temp_clusters.append(c)
        temp_clusters = temp_clusters[1:]

        # collapse previously clustered
        temp_clust_dict = {}
        for c, clust in enumerate(temp_clusters):
            for n in clust:
                temp_clust_dict[n] = c
        clust_dict = defaultdict(list)
        for clust in clusters:
            c = -1
            for n in clust:
                if n in temp_clust_dict:
                    c = temp_clust_dict[n]
            for n in clust:
                clust_dict[c].append(n)
        clusters = clust_dict.values()

        # cleanup and rename for next round
        os.remove(temp_input_file.name)
        os.remove(temp_output_file.name + ".clstr")
        temp_input_file.name = temp_output_file.name
        temp_output_file.name = temp_output_file.name + "t" + str(cid)

    return (clusters)


def pwdist_edlib(G, cdhit_clusters, threshold, dna=False, n_cpu=1):

    # Prepare sequences
    centroid_to_seq = {}
    for node in G.nodes():
        if dna:
            for sid, seq in zip(G.nodes[node]["centroid"],
                                G.nodes[node]["dna"]):
                centroid_to_seq[sid] = seq
        else:
            for sid, seq in zip(G.nodes[node]["centroid"],
                                G.nodes[node]["protein"]):
                centroid_to_seq[sid] = seq

    ncentroids = len(centroid_to_seq)

    # centroid to index
    centroid_to_index = {}
    for i, centroid in enumerate(centroid_to_seq):
        centroid_to_index[centroid] = i

    # get pairwise id between sequences in the same cdhit clusters
    all_distances = []
    for cluster in cdhit_clusters:
        all_distances += Parallel(n_jobs=n_cpu)(
            delayed(run_pw)(centroid_to_seq[c1], centroid_to_seq[c2],
                            centroid_to_index[c1], centroid_to_index[c2], dna)
            for c1, c2 in itertools.combinations(cluster, 2))

    data = []
    row_ind = []
    col_ind = []
    for d in all_distances:
        if d[2] >= threshold:
            data.append(1)
            row_ind.append(d[0])
            col_ind.append(d[1])

    distances_bwtn_centroids = csr_matrix((data, (row_ind, col_ind)),
                                          shape=(ncentroids, ncentroids))

    return distances_bwtn_centroids, centroid_to_index


def run_pw(seqA, seqB, n1, n2, dna):

    if len(seqA) > len(seqB):
        seqA, seqB = seqB, seqA

    if dna:
        pwid = 0.0
        for sA in [seqA, str(Seq(seqA).reverse_complement())]:
            aln = edlib.align(sA,
                              seqB,
                              mode="HW",
                              task='distance',
                              k=0.5 * len(seqA),
                              additionalEqualities=[('A', 'N'), ('C', 'N'),
                                                    ('G', 'N'), ('T', 'N')])
            if aln['editDistance'] == -1:
                pqid = max(pwid, 0.0)
            else:
                pwid = max(pwid, 1.0 - aln['editDistance'] / float(len(seqA)))
    else:
        aln = edlib.align(seqA,
                          seqB,
                          mode="HW",
                          task='distance',
                          k=0.5 * len(seqA),
                          additionalEqualities=[('*', 'X'), ('A', 'X'),
                                                ('C', 'X'), ('B', 'X'),
                                                ('E', 'X'), ('D', 'X'),
                                                ('G', 'X'), ('F', 'X'),
                                                ('I', 'X'), ('H', 'X'),
                                                ('K', 'X'), ('M', 'X'),
                                                ('L', 'X'), ('N', 'X'),
                                                ('Q', 'X'), ('P', 'X'),
                                                ('S', 'X'), ('R', 'X'),
                                                ('T', 'X'), ('W', 'X'),
                                                ('V', 'X'), ('Y', 'X'),
                                                ('X', 'X'), ('Z', 'X'),
                                                ('D', 'B'), ('N', 'B'),
                                                ('E', 'Z'), ('Q', 'Z')])
        if aln['editDistance'] == -1:
            pwid = 0.0
        else:
            pwid = 1.0 - aln['editDistance'] / float(len(seqA))

    return ((n1, n2, pwid))