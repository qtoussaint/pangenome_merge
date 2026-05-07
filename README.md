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
  - matplotlib-base

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
                        0.5
  --frameshift-identity FRAMESHIFT_IDENTITY
                        Sequence identity threshold for frameshift second-pass
                        merge. Default: 0.95
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

# Example Analysis

<img width="1266" height="925" alt="pangenome gene graph" src="https://github.com/user-attachments/assets/6dd0e0d1-6a77-4385-aa9e-950fd80caef1" />

*A pangenome gene graph of a large Streptococcus pneumoniae population (119k isolates, sourced from the [AllTheBacteria](https://allthebacteria.org/) project), with capsule genes highlighted in red. Produced using PopPUNK, ggCaller, panaroo, and pangenomerge and visualized using Gephi.*

# Citations

Pangenomerge is based on several tools, including:

- Panaroo: Tonkin-Hill G, MacAlasdair N, Ruis C, Weimann A, Horesh G, Lees JA, Gladstone RA, Lo S, Beaudoin C, Floto RA, Frost SDW, Corander J, Bentley SD, Parkhill J. 2020. Producing polished prokaryotic pangenomes with the Panaroo pipeline. Genome Biol 21:180.
- MMseqs2: Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026–1028 (2017). https://doi.org/10.1038/nbt.3988

