<img alt="pangenomerge_logo" src="https://github.com/user-attachments/assets/64fb85f9-5697-4326-b2a5-4770923b3bc4" align="right" height="170" />

# pangenomerge 

Construct pangenome gene graphs for hundreds of thousands of bacterial genomes 🦠

# About

pangenomerge is a fast, accurate and reproducible tool to merge [Panaroo](https://github.com/gtonkinhill/panaroo) pangenome gene graphs or update them with individual genomes.

pangenomerge rapidly maps clusters of orthologous genes (COGs) between input graphs using [MMseqs2](https://github.com/soedinglab/MMseqs2) and scores hits by synteny, collapsing their orthologous genes and merging them into a single complete pangenome graph. An option to efficiently create component graphs from a very large (>100k-genome) population is available via Snakemake, which identifies strains within the population using PopPUNK and generates individual strain-level pangenome gene graphs using ggCaller and Panaroo before running pangenomerge to create a final complete graph. 

pangenomerge's runtime scales approximately linearly with the number of graphs to be merged, assuming each graph is comprised of a novel cluster of related isolates such as a strain within the population, while its maximum memory consumption increases asympotically with the number of graphs in rough proportion to the population's rarefaction curve. (Merging 426 component graphs containing, in total, 119k _Streptococcus pneumoniae_ isolates required 8hrs+3min of runtime and 7.8G maximum memory using 48 threads.) Thus, the practical limitation to the size of the pangenome graph that can be created lies primarily in the preceding step of generating component graphs.

> [!NOTE]
> pangenomerge only takes [Panaroo](https://github.com/gtonkinhill/panaroo) graphs as input

# Installation

## Dependencies

  - python
  - biopython
  - networkx
  - mmseqs2
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - edlib
  - tqdm
  - joblib
  - matplotlib-base
  - mafft (only for `pangenomerge-msa`)

## Installing with conda 

pangenomerge is available on the bioconda channel. Install in a new environment (recommended):

```
conda create -n pangenomerge -c bioconda pangenomerge
```

or into an existing environment:
```
conda install -c bioconda pangenomerge
```

## Installing with Snakemake

pangenomerge can be automatically installed and run via Snakemake (see [Workflow management and reproducibility for large analyses](https://github.com/qtoussaint/pangenome_merge#workflow-management-and-reproducibility-for-large-analyses)). To use this option, ensure you have installed Snakemake through micromamba.

First, clone the pangenomerge repository:
```
git clone https://github.com/qtoussaint/pangenome_merge
```

Next, create a config file for your project. An example `config.yaml` is available in `snakemake/config.yaml`.

In your `config.yaml`, you will need to provide a path to a [PopPUNK-format TSV](https://poppunk-docs.bacpop.org/query_assignment.html) containing the paths to your assemblies and their sample IDs (see the instructions for creating `qfile.txt` in the PopPUNK documentation).

Finally, run the Snakemake pipeline:
```
snakemake --executor slurm -j <maximum_concurrent_jobs> --group-components job_array=1 --default-resources slurm_account=<your_account> --snakefile </path/to/pangenome_merge/snakemake/Snakefile> --configfile </path/to/project_directory/config.yaml> --use-conda --latency-wait 60 --verbose
```
For more information about these options, consult `snakemake/example_slurm_run.sh` and the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/).

## Installing from source (not recommended)

You'll need to install the dependencies listed in `meta.yaml` manually. 

Clone the pangenomerge repository:
```
git clone https://github.com/qtoussaint/pangenome_merge
```
You can then run pangenomerge from the helper:
```
python3 /path/to/pangenome_merge/pangenomerge-runner.py --version
```
or install via pip:
```
cd /path/to/pangenome_merge
pip install .
pangenomerge --version
```

# Quickstart

To merge two or more Panaroo pangenome graphs, create a `.txt` file with one Panaroo output directory path per line, for example `paths.txt`. Then run:

```
pangenomerge --component-graphs paths.txt --outdir </path/to/outdir> --threads 16
```
> [!TIP]
> Make sure to provide paths to the Panaroo _directories_, not the `final_graph.gml` files they contain.

This will generate the following in your results directory:
  - `final_graph.gml`: the final merged pangenome gene graph
  - `intermediate_graphs/merged_graph_<index>.gml`: checkpoint graphs from each merge iteration
  - `pangenome_reference_aa/`: an MMseqs2 amino acid database containing representative protein sequences for each node (COG) in the final graph
  - `pangenome_metadata.sqlite`: an SQLite database containing all metadata for the final merged graph
  - `summary_statistics.txt`: core / soft-core / shell / cloud gene counts (with a strain-level breakdown if PopPUNK clusters were provided via `--include-clusters`)

## Generating additional outputs

### Panaroo-format outputs

> [!CAUTION]  
> Some Panaroo outputs, particularly `gene_data.csv`, can be >50GB for large datasets.
> 
> I have included them to allow for the use of tools requiring Panaroo-format outputs, but it's better to use the existing pangenomerge metadata files where possible.
> 
> If you need to use them, first test on a subset of your component graphs and decide if the filesize will be reasonable for your storage constraints and downstream usage before creating them for your entire dataset. 

After a merge, use `pangenomerge-postprocess` to generate Panaroo-format output files:

```
pangenomerge-postprocess --pangenomerge-results </path/to/pangenomerge_outdir> --component-graphs </path/to/paths.txt>
```

To specify pangenomerge files individually, such as for results with nonstandard filenames or locations, you can use individual flags instead:

```
pangenomerge-postprocess --sqlite </path/to/pangenome_metadata.sqlite> --gml </path/to/final_graph.gml> --component-graphs </path/to/paths.txt> --outdir </path/to/outdir>
```

This generates:
  - `gene_presence_absence_roary.csv`: Roary-format gene presence/absence matrix (14 metadata columns + one column per isolate)
  - `gene_presence_absence.csv`: simplified gene presence/absence matrix (3 metadata columns + one column per isolate)
  - `gene_presence_absence.Rtab`: binary presence/absence matrix
  - `pangenome_sequences.sqlite`: deduplicated, compressed per-gene DNA and protein sequences, queryable by node/COG
  - figures: COG frequency histograms (plus strain-level versions when `--include-clusters` was provided at merge time), multi-copy gene plots, `merge_statistics.csv`, and `pangenome_growth_curve.png`

`gene_data.csv` (Panaroo per-gene annotation) is **not** included by `--output all` because it can be very large; pass `--gene-data` to add it, or use `--output genedata` to generate it on its own.

You can generate specific outputs using `--output`:
```
pangenomerge-postprocess --sqlite db.sqlite --gml graph.gml --outdir out/ --output presenceabsence
pangenomerge-postprocess --sqlite db.sqlite --component-graphs paths.txt --outdir out/ --output genedata
pangenomerge-postprocess --sqlite db.sqlite --component-graphs paths.txt --outdir out/ --output sequences
pangenomerge-postprocess --sqlite db.sqlite --outdir out/ --output figures
```

### Multiple sequence alignments

`pangenomerge-msa` generates per-gene multiple sequence alignments, and a concatenated core genome alignment for phylogenetics, from a completed merge. It reads sequences from `pangenome_sequences.sqlite`, so run `pangenomerge-postprocess --output sequences` first:

```
pangenomerge --component-graphs paths.txt --outdir results/ --threads 16
pangenomerge-postprocess --pangenomerge-results results/ --output sequences --component-graphs paths.txt
pangenomerge-msa --outdir results/ --alignment core --aligner mafft --threads 16
```

This generates, in the alignment output directory:
  - `aligned_gene_sequences/`: one alignment per gene cluster
  - `alignment_resume_state.json`: run parameters, used by `--resume`

Pass `--core-alignment` to additionally write the concatenated core genome alignment
(`core_gene_alignment.aln`, `core_gene_alignment_filtered.aln`, `core_alignment_header.embl`,
`alignment_entropy.csv`). It is **off by default**: the concatenation step holds every core gene
alignment in memory simultaneously, which on a species-scale pangenome means tens of GB of RAM and
an output file to match. `--core_entropy_filter` only applies when this flag is given.

Gene names match the `Gene` column of `gene_presence_absence.csv`, so alignments can be joined to the presence/absence tables directly.

#### Aligners

`--aligner` accepts `mafft` (default), `muscle`, `muscle-super5`, `famsa`, or `none`.

  - `none` writes unaligned per-gene FASTA files to `unaligned_gene_sequences/` and skips alignment entirely. Useful for feeding another tool, or for a fast check that gene selection is behaving.
  - `famsa` is amino-acid only, so it requires `--codons` or `--strict-codons`.
  - `muscle` is not optimised beyond a few hundred isolates; use `muscle-super5` for larger datasets.

#### Codon alignments

`--codons` aligns amino-acid sequences first and then reverse-translates, which keeps the alignment in frame. Sequences that fail translation QC (length not divisible by 3, translation disagreeing with the stored protein, runs of `N`, or degenerate codons) are re-aligned back onto the codon alignment as DNA.

`--strict-codons` does the same but drops those sequences instead of re-aligning them, giving a cleaner alignment with fewer isolates per gene. On a test set where 20% of sequences failed QC, the `--codons` alignments carried 49-72 more records per gene than the `--strict-codons` ones.

The re-alignment step in `--codons` is a MAFFT profile alignment (`mafft --add`) of the rejected DNA onto the codon alignment, and it is memory-hungry on large gene clusters. If MAFFT is killed, the run now stops with the gene name rather than writing an empty alignment.

The two modes produce identical `aligned_protein_sequences/` and `unaligned_dna_sequences/` — they differ only in how genes with QC-failing DNA are finished. If you want both, pass the same `--shared-alignment-dir` to each run so the (expensive) protein alignment happens once:

```
pangenomerge-msa --outdir results/ --alignment-outdir msa/codon \
    --shared-alignment-dir msa/alignment --alignment pan --codons --aligner mafft --threads 16

pangenomerge-msa --outdir results/ --alignment-outdir msa/codon_strict \
    --shared-alignment-dir msa/alignment --alignment pan --strict-codons --aligner mafft --threads 16
```

The second run finds every protein alignment already present and skips straight to reverse translation. This is what the Snakemake workflow does.

The shared directory records the aligner it was built with in `shared_alignment_state.json`; a later run pointing at it with a different `--aligner` is refused rather than silently reusing incompatible alignments. Reuse of the shared directory does not require `--resume` — `--resume` governs only the per-mode output directory.

How much this saves depends on the data. On a 40-gene / 200-isolate test the second pass went from 396s to 265s (the reverse-translation step, not MAFFT, dominates at that size); on a 6-gene set of much larger clusters it went from 171s to 28s.

#### Memory

Peak memory is roughly `--threads` x the largest gene cluster: each cluster is read from SQLite,
aligned, written and freed as a self-contained unit, so nothing scales with the number of genes.
On a memory-constrained node, lower `--threads`.

For scale, on a 45,778-isolate *S. pyogenes* pangenome (14,247 clusters, 80M gene IDs) the graph
load alone peaks at under 400 MB. Note that it does have to scan the `node_geneids` and
`node_members` tables once to count sequences per cluster, which on a 30 GB SQLite database over
network storage takes several minutes before alignment begins.

The two large-pangenome traps to avoid are `--core-alignment` (see above) and `--alignment pan`,
which aligns all 14,247 clusters rather than only the core.

#### Core vs pan, and resuming

`--alignment core` (default) aligns only genes present in at least `--core_threshold` (default 0.95) of isolates. `--alignment pan` aligns every gene cluster and then concatenates the core — considerably more expensive on a species-scale pangenome.

Long runs can be resumed. If a run is interrupted, re-running the **identical** command with `--resume` picks up from the completed alignments; the run parameters recorded in `alignment_resume_state.json` must match, so you cannot resume a `--codons` run as `--strict-codons`. Without `--resume`, a run that finds an existing manifest refuses to start rather than mixing output from two different parameter sets.

### Complete gene sequence database

Individual per-gene DNA and protein sequences are stored in a separate SQLite database (`pangenome_sequences.sqlite`) with hash-based deduplication and zlib compression. Within a COG, many isolates will share identical alleles, so deduplication in combination with compression typically results in 5-15 GB for datasets that would otherwise require 100+ GB of raw FASTA storage. This replaces Panaroo's `combined_DNA_CDS.fasta` and `combined_protein_CDS.fasta`. 

Sequences can be queried programmatically using the `sequence_queries` module:

```python
from pangenomerge.custom_functions.sqlite import sqlite_connect
from pangenomerge.custom_functions.sequence_queries import attach_sequences, get_sequences_for_node

con = sqlite_connect("pangenome_metadata.sqlite", sqlite_cache=2000)
attach_sequences(con, "pangenome_sequences.sqlite")

# get all gene sequences within a COG
seqs = get_sequences_for_node(con, "node_123", seq_type="nt")

# get unique alleles within a COG (outputs geneIDs)
from pangenomerge.custom_functions.sequence_queries import get_unique_sequences_for_node
alleles = get_unique_sequences_for_node(con, "node_123", seq_type="aa")

# generate allele frequency spectrum
from pangenomerge.custom_functions.sequence_queries import get_sequence_counts_for_node
counts = get_sequence_counts_for_node(con, "node_123", seq_type="nt")

# export gene sequences within a COG to FASTA (e.g. to create an mmseqs2 database for that node)
from pangenomerge.custom_functions.sequence_queries import export_node_fasta
export_node_fasta(con, "node_123", "output.fasta", seq_type="nt", unique_only=True)
```

### Per-gene dN/dS

`snakemake/dn_ds/` fits a per-gene dN/dS (omega) to the pipeline's strict-codon alignments using [TOMBOMBADIL JAX](https://github.com/bacpop/TOMBOMBADIL_jax) on the `scalar_omega_pi` branch, which estimates a single alignment-wide omega per gene. It runs as a standalone SLURM job array once the `msa` rule has finished, and is not part of `rule all`.

```bash
snakemake/dn_ds/run_dnds.sh --results-dir /path/to/project/results
```

This writes `<results>/dnds/gene_omega.csv` with two columns, `node,omega`, covering every gene in the pangenome; genes with no usable fit have a blank omega. Per-gene detail — sequence count, how much of the alignment survived filtering, whether the optimiser converged, and why a gene was skipped — goes to `<results>/dnds/dnds_stats.tsv`. Genes whose `fit_status` reads `reached max steps` rather than `converged` did not settle, and their omega should be treated as unreliable; this is most common for genes with few sequences.

The alignments are always taken from `<results>/msa/codon_strict/aligned_gene_sequences`, i.e. the `--strict-codons` output. Strict mode drops sequences that fail translation QC rather than re-aligning them, so every sequence in the alignment is in frame, free of ambiguous bases and internal stops, and gapped only in whole codons. `--msa-dir` can point elsewhere but refuses a directory outside `codon_strict/` unless `--force-msa-dir` is given, and each gene's alignment is spot-checked for the properties strict mode guarantees.

#### Gap filtering

Codon columns present in fewer than `--min-occupancy` (default 0.5) of sequences are dropped **before** the fit. TOMBOMBADIL excludes gap codons from its counts but keeps the column at a reduced sample size, so a column present in 2% of sequences still contributes to both the objective and the F3x4 equilibrium frequencies. Because omega is a single scalar per gene there is no per-site estimate to filter afterwards, so this is the only point at which those columns can be excluded. Occupancy is sharply bimodal in practice — columns are either almost fully occupied or nearly empty — so the threshold is not a sensitive parameter. Genes with fewer than `--min-codons` (default 30) surviving columns are skipped.

#### Scheduling

Genes are processed in chunks of `--chunk-size` (default 50) by a single job array, throttled to `--max-concurrent` (default 100) tasks, rather than one job per gene: a species-scale pangenome has O(10^4) genes at ~2-3 min each, which one job per gene would swamp with scheduling overhead. Defaults of 8 CPUs, 8 GB and 5 h per task come from measured usage (~1.1 GB peak, 1.5-2.5 min per gene of ~46k sequences). A gene's fit is skipped if its output already exists, so a task killed on walltime can be resubmitted as-is. Run with `--dry-run` to see the `sbatch` commands without submitting, or `--help` for all options.


### Per-gene diversity (pi)

`snakemake/pi/` measures standing diversity rather than selection: one amino-acid pi and one nucleotide pi per gene, computed directly from the alignments with no model and no external tool. Like the dN/dS stage it runs as a standalone SLURM job array once the `msa` rule has finished, and is not part of `rule all`.

```bash
snakemake/pi/run_pi.sh --results-dir /path/to/project/results
```

This writes `<results>/pi/gene_pi.csv` with three columns, `node,pi_aa,pi_nt`, covering every gene in the pangenome; genes with no usable measurement have blank values. Per-gene detail goes to `<results>/pi/pi_stats.tsv`.

Both values are for the whole gene, not a per-site series. Each column contributes

```
n_j     = sequences with a real state at column j
p_i     = count of state i / n_j
pi_j    = n_j/(n_j - 1) * (1 - sum_i p_i^2)
pi_gene = mean of pi_j over retained columns with n_j >= 2
```

which is Nei and Li's estimator with pairwise deletion of missing data. Counting states rather than comparing sequence pairs is what makes this tractable — a 46k-sequence gene has ~1.05e9 pairs — and the two definitions agree when every pair has complete data. The number of retained columns is reported alongside each value so it can be renormalised.

#### Which alignments, and why

The alignments are taken from `<results>/msa/codon/aligned_gene_sequences` — the **non-strict** `--codons` output, the opposite of the choice the dN/dS stage makes. Diversity should be measured over every isolate, including the ones `--strict-codons` drops for failing translation QC, and nothing in this calculation relies on the guarantees strict mode provides: `N`, degenerate bases and part-gapped codons are simply missing data. `--msa-dir` can point elsewhere but refuses a directory outside `codon/` unless `--force-msa-dir` is given.

pi_aa comes from the protein alignment (`aligned_protein_sequences/`, found beside the codon alignments or under `<results>/msa/alignment/`, or set with `--protein-dir`) and pi_nt from the codon alignment, so the two are read from what the aligner actually produced rather than one being derived from the other.

Amino-acid columns count only the 20 standard residues. A gap, an `X`, or a stop is missing data and does not contribute to the column or to its occupancy — an internal stop in a non-strict alignment comes from `mafft --add` re-aligning a broken ORF, so counting it as a 21st state would let a handful of frameshifted records dominate a column. The corresponding *nucleotides* are still real bases and do count toward pi_nt; the stop rule is an amino-acid-level judgement, not a claim that the DNA is unreadable.

#### Gap filtering

Amino-acid columns present in fewer than `--min-occupancy` (default 0.5) of sequences are dropped, and pi_nt is measured over the codon triplets of exactly the columns that survived. Using one filter for both means pi_aa and pi_nt describe the same region of the gene, so they can be compared to each other. Genes with fewer than `--min-sites` (default 30) surviving columns are skipped.

That mapping is only valid if the two files line up, and in non-strict mode they need not: the protein alignment is shared with `--strict-codons` and is built *before* QC, after which `mafft --add` can insert columns into the codon alignment. Each gene is therefore checked — equal record counts, IDs matching in order over the first `--check-sequences` (default 100) records, and a codon alignment exactly three times the protein's width — and a gene that fails has its nucleotide sites filtered on their own occupancy instead, recorded as `filter_mode=independent` in `pi_stats.tsv`. This is per-gene and deliberately not fatal, but a run reporting many `independent` genes means the two measures no longer cover the same region and the comparison between them is no longer clean; `collect_pi.py` prints the tally.

#### Interpreting the two values

pi_aa is not expected to be smaller than pi_nt. A codon carries three nucleotide sites but one amino-acid site, so a wholly nonsynonymous change contributes three times as much per amino-acid site as per nucleotide site: `pi_aa / (3 * pi_nt)` is the ratio bounded near 0 (all variation synonymous) and 1 (all of it nonsynonymous), and it is what should be compared with omega from the dN/dS stage. On the seven GAC genes the two agree closely — 0.47, 0.51, 0.41, 0.51 and 0.55 against fitted omegas of 0.51, 0.64, 0.61, 0.56 and 0.62 — with one outlier, `group_1953_g1`, whose protein sequences are nearly invariant (pi_aa 6.1e-4) while its synonymous diversity is not (pi_nt 2.7e-3).

#### Scheduling

Genes are processed in chunks of `--chunk-size` (default 500) by a single job array, throttled to `--max-concurrent` (default 50) tasks. Defaults of 1 CPU, 4 GB and 2 h per task come from measured usage: the largest GAC gene (45,691 sequences x 1,986 nt, 93 MB) takes ~1.5 s, of which 0.4 s is reading the file and 0.5 s is pi_nt on top of pi_aa. That is ~100x cheaper per gene than the dN/dS fit, hence the larger chunks and smaller resources. Counting is done a block at a time with numpy, sized by `--block-cells` (default 4,000,000 alignment cells) rather than by a fixed number of records, so peak memory does not grow with gene length. A killed task simply recomputes its chunk — there is no intermediate artefact to cache. Run with `--dry-run` to see the `sbatch` commands without submitting, or `--help` for all options.


# Running pangenomerge

### What is the difference between the 'run' and 'test' modes?

'Run' mode merges two or more Panaroo pangenome gene graphs, or iteratively updates an existing graph with single genomes.

'Test' mode creates a merged graph and provides clustering accuracy metrics based on a ground truth graph; this mode is considerably slower than run mode and is not intended for use with large datasets (>3k samples).

### What are the family and context thresholds?

These thresholds represent the fraction of identical amino acids between two aligned COGs, expressed as floats (e.g. 98% identity = 0.98).

The **family threshold** is _the minimum amino acid identity required for two COGs to be examined as potential orthologs_. This doesn't mean that all genes with this identity will be merged, but rather that context search will be performed to evaluate their syntenic similarity. Regardless of the family threshold specified, COGs that have AA identity>=98% are always merged (unless one of the COGs has a higher-identity match with a different node).

The **context threshold** is _the minimum amino acid identity required for neighboring genes to be considered a 'match' (orthologous) during context search_. In other words, any neighboring genes with AA identity above this value will count as support towards the hypothesis that the pair of COGs are orthologous and should be merged.

### What is the most principled method to choose a family and context threshold for my dataset?

You can perform tests of clustering accuracy across different threshold values by using 'test' mode on a subset of your data. For instance, an example analysis might look like:
- Use PopPUNK to separate your population into strains
- Use ggCaller to call genes within each strain population
- Use Panaroo to create graphs of three strains individually
- Use Panaroo to create a graph of all isolates from the three strains combined
- Run pangenomerge in test mode with different threshold values; this will compare the graph resulting from pangenomerge merging the three strain graphs to the "ground truth" graph created from all isolates using Panaroo
- Use the threshold values that result in the best adjusted Rand index and adjusted mutual information scores

You can additionally compare the level of collapse between genes that you know should be collapsed into one COG or kept separate, and/or the number of new COGs added to the merged graph with each iteration, and adjust the thresholds up or down accordingly.
  
We have tested various default settings for these thresholds on several bacterial species and obtained the highest clustering accuracy using the current defaults; when in doubt, these are a good baseline. An important caveat is that Panaroo is not intended for use on highly diverse populations, such as some mixed-strain datasets; while considering clustering accuracy metrics can help us understand how similar a pangenomerge graph is to a Panaroo graph created from the same isolates, they cannot distinguish which graph is more 'correct'. We nonetheless use Panaroo graphs as a ground truth because Panaroo, as a gold-standard graphing method, provides us the closest estimate to the true graph we can realistically obtain.

### Workflow management and reproducibility for large analyses

Many people running pangenomerge will be interested in creating pangenomes with hundreds of thousands of genomes. This involves substantial large-scale data analysis prior to running pangenomerge, including clustering genomes into strains by genetic relatedness, calling genes on strain-level populations, and creating hundreds or thousands of strain-level Panaroo gene graphs. To reduce the burden of this upstream analysis and improve its reproducibility, a Snakemake pipeline with Slurm capability is available in the Snakemake folder.

To run pangenomerge from Snakemake, follow the steps in `snakemake/example_slurm_run.sh`. This option takes a TSV of sample IDs and assembly paths as input, and runs the recommended workflow of PopPUNK, ggCaller, panaroo, and pangenomerge with your chosen parameters, spreading compute across your HPC cluster. All software is automatically installed and managed by Snakemake via preconfigured conda YAMLs.

# Reference Library

## pangenomerge

```
usage: pangenomerge [-h] [--mode {run,test}] --outdir OUTDIR
                    [--component-graphs COMPONENT_GRAPHS]
                    [--iterative ITERATIVE] [--graph-all GRAPH_ALL]
                    [--metadata-in-graph KEEP_METADATA_IN_GRAPH]
                    [--include-clusters INCLUDE_CLUSTERS]
                    [--family-threshold FAMILY_THRESHOLD]
                    [--context-threshold CONTEXT_THRESHOLD]
                    [--frameshift-coverage FRAMESHIFT_COVERAGE]
                    [--frameshift-identity FRAMESHIFT_IDENTITY]
                    [--context-search-iterations CONTEXT_SEARCH_ITERATIONS]
                    [--threads THREADS] [--sqlite-cache SQLITE_CACHE]
                    [--debug] [--version]

Merges two or more Panaroo pangenome gene graphs, or iteratively updates an
existing graph.

options:
  -h, --help            show this help message and exit

Input and output options:
  --mode {run,test}     Run pangenome gene graph merge ("run") or calculate
                        clustering accuracy metrics for merge ("test").
                        [Default = Run]
  --outdir OUTDIR       Output directory.
  --component-graphs COMPONENT_GRAPHS
                        Path to a text file with one Panaroo output directory
                        path per line. Each directory must contain
                        final_graph.gml and pan_genome_reference.fa. If
                        running in test mode, must also contain gene_data.csv.
                        Graphs will be merged in the order presented in the
                        file.
  --iterative ITERATIVE
                        Tab-separated list of GFFs and their sample IDs for
                        iterative updating of the graph. Use only for single
                        samples or sets of samples too diverse to create an
                        initial pangenome. Samples will be merged in the order
                        presented in the file.
  --graph-all GRAPH_ALL
                        Path to Panaroo output directory of pangenome gene
                        graph created from all samples in component-graphs.
                        Only required for the test case, where it is used as
                        the ground truth.
  --metadata-in-graph KEEP_METADATA_IN_GRAPH
                        Retains metadata in the final graph GML (in addition
                        to the SQLite database). Dramatically increases
                        runtime and memory consumption. Not recommended with
                        >10k isolates.
  --include-clusters INCLUDE_CLUSTERS
                        Path to a PopPUNK clusters CSV (header:
                        Taxon,Cluster). If provided, populates
                        isolate_names.poppunk_cluster in the SQLite. Sample
                        names in the CSV must match those in the input graphs;
                        isolates without a match are warned and stored as
                        NULL.

Parameters:
  --family-threshold FAMILY_THRESHOLD
                        Sequence identity threshold for putative spurious
                        paralogs. Default: 0.7
  --context-threshold CONTEXT_THRESHOLD
                        Sequence identity threshold for neighbors of putative
                        spurious paralogs. Default: 0.7
  --frameshift-coverage FRAMESHIFT_COVERAGE
                        Coverage threshold for frameshift second-pass merge
                        (catches truncated/frameshifted variants). Default:
                        0.25
  --frameshift-identity FRAMESHIFT_IDENTITY
                        Sequence identity threshold for frameshift second-pass
                        merge. Default: 0.90
  --context-search-iterations CONTEXT_SEARCH_ITERATIONS
                        Max outer rounds of the alternating context/frameshift
                        merge loop (family inner loop always runs to its own
                        fixed point). -1 = run until no new pairs merge.
                        Default: -1

Other options:
  --threads THREADS     Number of threads
  --sqlite-cache SQLITE_CACHE
                        Desired size of SQLite cache expressed in KB.
                        Diminishing returns above 1 GB (1048576 KB). Defaults
                        to 2000 KB.
  --debug               Set logging to 'debug' instead of 'info' (default)
  --version             show program's version number and exit
```

## pangenomerge-postprocess

```
usage: pangenomerge-postprocess [-h]
                                [--pangenomerge-results PANGENOMERGE_RESULTS]
                                [--sqlite SQLITE] [--gml GML]
                                [--outdir OUTDIR]
                                [--output {all,presenceabsence,genedata,sequences,figures}]
                                [--gene-data]
                                [--component-graphs COMPONENT_GRAPHS]
                                [--sequences-sqlite SEQUENCES_SQLITE]
                                [--sqlite-cache SQLITE_CACHE]

Generate Panaroo-format output files from pangenomerge output

options:
  -h, --help            show this help message and exit
  --pangenomerge-results PANGENOMERGE_RESULTS
                        Path to a pangenomerge output directory containing the
                        --sqlite, --gml, and --sequences-sqlite inputs.
                        Mutually exclusive with those flags. When set,
                        --outdir defaults to <pangenomerge-
                        results>/postprocessing.
  --sqlite SQLITE       Path to pangenome_metadata.sqlite
  --gml GML             Path to final_graph.gml (required for gene presence-
                        absence output)
  --outdir OUTDIR       Output directory for generated files (default:
                        <pangenomerge-results>/postprocessing when
                        --pangenomerge-results is set)
  --output {all,presenceabsence,genedata,sequences,figures}
                        Which outputs to generate: presenceabsence (Panaroo-
                        format gene presence-absence tables), genedata
                        (Panaroo-format gene_data.csv), sequences
                        (pangenome_sequences.sqlite), figures (COG frequency
                        histograms, multi-copy gene plots,
                        merge_statistics.csv, pangenome_growth_curve.png;
                        strain-level histograms additionally generated when
                        --include-clusters was passed at merge time). 'all'
                        generates everything except gene_data.csv; pass
                        --gene-data to include it (default: all)
  --gene-data           Also generate gene_data.csv alongside the outputs
                        selected by --output.
  --component-graphs COMPONENT_GRAPHS
                        Path to text file with one Panaroo output directory
                        path per line (same file used for pangenomerge
                        --component-graphs). Required for all outputs except
                        --output 'figures' without --gene-data.
  --sequences-sqlite SEQUENCES_SQLITE
                        Path to pangenome_sequences.sqlite (default:
                        pangenome_sequences.sqlite in same dir as --sqlite)
  --sqlite-cache SQLITE_CACHE
                        SQLite cache size in KB (default: 2000)
```

## pangenomerge-msa

```
usage: pangenomerge-msa [-h] [-o PANGENOMERGE_RESULTS]
                        [--alignment-outdir ALIGNMENT_OUTDIR]
                        [--shared-alignment-dir SHARED_ALIGNMENT_DIR]
                        [--alignment {core,pan}]
                        [-a {mafft,muscle,muscle-super5,famsa,none}]
                        [--codons] [--strict-codons]
                        [--core_threshold CORE_THRESHOLD]
                        [--core_subset CORE_SUBSET]
                        [--core_entropy_filter CORE_ENTROPY_FILTER] [--resume]
                        [-t THREADS] [--verbose] [--version]

Generate Panaroo-style MSAs from Pangenomerge output

options:
  -h, --help            show this help message and exit
  -t THREADS, --threads THREADS
                        Number of worker threads to use. (default: 1)
  --verbose             Print additional progress information. (default: False)
  --version             show program's version number and exit

Input/output:
  -o PANGENOMERGE_RESULTS, --outdir PANGENOMERGE_RESULTS
                        Pangenomerge results directory. Defaults --gml,
                        --sqlite, --sequences-sqlite to files in this
                        directory. Optionally, supply them separately.
  --alignment-outdir ALIGNMENT_OUTDIR
                        Output directory. Defaults to the --outdir directory.
  --shared-alignment-dir SHARED_ALIGNMENT_DIR
                        Directory for intermediates that do not depend on
                        --strict-codons (aligned_protein_sequences/,
                        unaligned_dna_sequences/). Point a --codons and a
                        --strict-codons run at the same directory to align the
                        proteins once instead of twice. Defaults to the
                        alignment output directory.

Gene alignment:
  --alignment {core,pan}
                        Generate core or pan genome per-gene alignments.
                        (default: core)
  -a, --aligner {mafft,muscle,muscle-super5,famsa,none}
                        External aligner to use, or 'none' to write unaligned
                        FASTA files. (default: mafft)
  --codons              Generate codon alignments by aligning amino-acid
                        sequences first. (default: False)
  --strict-codons       Only generate codon alignments with well-formed protein
                        sequences, no DNA re-alignment of pseudogenes or
                        frameshifts. (default: False)
  --core_threshold CORE_THRESHOLD
                        Core-genome frequency threshold. (default: 0.95)
  --core_subset CORE_SUBSET
                        Randomly subset the core genome to this many genes.
  --core_entropy_filter CORE_ENTROPY_FILTER
                        Manual Block Mapping and Gathering with Entropy filter.
                        If omitted, the Tukey outlier rule is used.
  --resume              Resume a previously incomplete gene alignment run.
                        (default: False)
```

# Example Analysis

<img width="1266" height="925" alt="pangenome gene graph" src="https://github.com/user-attachments/assets/6dd0e0d1-6a77-4385-aa9e-950fd80caef1" />

*A pangenome gene graph of a large Streptococcus pneumoniae population (119k isolates, sourced from the [AllTheBacteria](https://allthebacteria.org/) project), with capsule genes highlighted in red. Produced using PopPUNK, ggCaller, panaroo, and pangenomerge and visualized using Gephi.*

# Citations

Pangenomerge is based on several tools, including:

- Panaroo: Tonkin-Hill G, MacAlasdair N, Ruis C, Weimann A, Horesh G, Lees JA, Gladstone RA, Lo S, Beaudoin C, Floto RA, Frost SDW, Corander J, Bentley SD, Parkhill J. 2020. Producing polished prokaryotic pangenomes with the Panaroo pipeline. Genome Biol 21:180.
- MMseqs2: Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026–1028 (2017). https://doi.org/10.1038/nbt.3988

