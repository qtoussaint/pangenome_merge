<img alt="pangenomerge_logo" src="https://github.com/user-attachments/assets/75ad835c-0719-462c-b3d3-3065689b7733" align="right" height="250" />

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

  - python >=3.10
  - biopython >=1.80
  - networkx >=3.4.2
  - mmseqs2
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - edlib
  - tqdm

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

pangenomerge can be automatically installed and run via Snakemake (see "Workflow management and reproducibility for large analyses"). To use this option, ensure you have installed Snakemake through micromamba.

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

To merge two or more Panaroo pangenome graphs, create a TSV containing the paths to each Panaroo output directory, for example `paths.tsv`. Then run:

```
pangenomerge --component-graphs paths.tsv --outdir </path/to/outdir> --threads 16
```
> [!TIP]
> Make sure to provide paths to the Panaroo _directories_, not the `final_graph.gml` files they contain.

This will generate the following in your results directory:
  - Graphs titled `merged_graph_<index>.gml`: an updated graph is output every time a new graph is merged into the base graph (e.g. when merging 15 graphs, 13 intermediary graphs and one final graph will be output)
  - `mmseqs_tmp/pan_genome_db_<index>`: an MMseqs2 database containing representative sequences for each node (COG) in the graph
  - `pangenome_metadata.sqlite`: an SQLite database containing all metadata for the final merged graph

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

```
usage: pangenomerge [-h] [--mode {run,test}] --outdir OUTDIR [--component-graphs COMPONENT_GRAPHS] [--iterative ITERATIVE] [--graph-all GRAPH_ALL] [--metadata-in-graph KEEP_METADATA_IN_GRAPH] [--family-threshold FAMILY_THRESHOLD] [--context-threshold CONTEXT_THRESHOLD] [--threads THREADS] [--sqlite-cache SQLITE_CACHE]
                    [--debug] [--version]

Merges two or more Panaroo pangenome gene graphs, or iteratively updates an existing graph.

options:
  -h, --help            show this help message and exit

Input and output options:
  --mode {run,test}     Run pangenome gene graph merge ("run") or calculate clustering accuracy metrics for merge ("test"). [Default = Run]
  --outdir OUTDIR       Output directory.
  --component-graphs COMPONENT_GRAPHS
                        Tab-separated list of paths to Panaroo output directories of component subgraphs. Each directory must contain final_graph.gml and pan_genome_reference.fa. If running in test mode, must also contain gene_data.csv. Graphs will be merged in the order presented in the file.
  --iterative ITERATIVE
                        Tab-separated list of GFFs and their sample IDs for iterative updating of the graph. Use only for single samples or sets of samples too diverse to create an initial pangenome. Samples will be merged in the order presented in the file.
  --graph-all GRAPH_ALL
                        Path to Panaroo output directory of pangenome gene graph created from all samples in component-graphs. Only required for the test case, where it is used as the ground truth.
  --metadata-in-graph KEEP_METADATA_IN_GRAPH
                        Retains metadata in the final graph GML (in addition to the SQLite database). Dramatically increases runtime and memory consumption. Not recommended with >10k isolates.

Parameters:
  --family-threshold FAMILY_THRESHOLD
                        Sequence identity threshold for putative spurious paralogs. Default: 0.7
  --context-threshold CONTEXT_THRESHOLD
                        Sequence identity threshold for neighbors of putative spurious paralogs. Default: 0.9

Other options:
  --threads THREADS     Number of threads
  --sqlite-cache SQLITE_CACHE
                        Desired size of SQLite cache expressed in KB. Diminishing returns above 1 GB (1048576 KB). Defaults to 2000 KB.
  --debug               Set logging to 'debug' instead of 'info' (default)
  --version             show program's version number and exit
```

# Example Analysis

<img width="1266" height="925" alt="pangenome gene graph" src="https://github.com/user-attachments/assets/6dd0e0d1-6a77-4385-aa9e-950fd80caef1" />

*A pangenome gene graph of a large Streptococcus pneumoniae population (119k isolates, sourced from the [AllTheBacteria](https://allthebacteria.org/) project), with capsule genes highlighted in red. Produced using PopPUNK, ggCaller, panaroo, and pangenomerge and visualized using Gephi.*

# Citations

Pangenomerge is based on several tools, including:

- Panaroo: Tonkin-Hill G, MacAlasdair N, Ruis C, Weimann A, Horesh G, Lees JA, Gladstone RA, Lo S, Beaudoin C, Floto RA, Frost SDW, Corander J, Bentley SD, Parkhill J. 2020. Producing polished prokaryotic pangenomes with the Panaroo pipeline. Genome Biol 21:180.
- MMseqs2: Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026–1028 (2017). https://doi.org/10.1038/nbt.3988

