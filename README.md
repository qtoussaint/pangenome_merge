# pangenomerge

🚧🚧🚧 Pangenomerge is still in beta and is subject to change! 🚧🚧🚧

The full documentation and a tutorial will be added mid-March; in the meantime, feel free to contact me at lilyjacqueline [at] ebi [dot] ac [dot] uk with any questions or features you'd like to see.

### Documentation

'Run' mode merges two or more Panaroo pangenome gene graphs, or iteratively updates an existing graph with single genomes.

'Test' mode creates a merged graph and provides clustering accuracy metrics based on a ground truth graph; this mode is considerably slower than run mode and is not intended for use with large datasets (>3k samples). 

A Snakemake pipeline with Slurm capability is available in the Snakemake folder. To run pangenomerge from Snakemake, follow the steps in `snakemake/example_slurm_run.sh`. This option takes a TSV of sample IDs and assembly paths as input, and runs PopPUNK, ggCaller, panaroo, and pangenomerge with your chosen parameters (spreading compute across your HPC cluster). All software is automatically installed and managed by Snakemake via preconfigured conda YAMLs.

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

### Example Analyses

<img width="1266" height="925" alt="pangenome gene graph" src="https://github.com/user-attachments/assets/6dd0e0d1-6a77-4385-aa9e-950fd80caef1" />

*A pangenome gene graph of a large Streptococcus pneumoniae population (119k isolates, sourced from the [AllTheBacteria](https://allthebacteria.org/) project), with capsule genes highlighted in red. Produced using PopPUNK, ggCaller, panaroo, and pangenomerge.*
