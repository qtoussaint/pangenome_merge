# pangenome_merge

🚧🚧🚧 Under construction -- use with caution 🚧🚧🚧

'Run' mode merges two or more Panaroo pan-genome gene graphs (or iteratively updates an existing graph). 'Test' mode creates a merged graph and provides clustering accuracy metrics based on a ground truth graph; this mode is considerably slower than run mode and is not intended for use with large datasets (>3k samples). 

```
usage: pangenomerge [-h] [--mode {run,test}] --outdir OUTDIR [--component-graphs COMPONENT_GRAPHS] [--iterative ITERATIVE] [--graph-all GRAPH_ALL] [--metadata-in-graph KEEP_METADATA_IN_GRAPH] [--family-threshold FAMILY_THRESHOLD] [--context-threshold CONTEXT_THRESHOLD] [--threads THREADS] [--sqlite-cache SQLITE_CACHE] [--debug] [--version]

Merges two or more Panaroo pan-genome gene graphs, or iteratively updates an existing graph.

options:
  -h, --help            show this help message and exit

Input and output options:
  --mode {run,test}     Run pan-genome gene graph merge ("run") or calculate clustering accuracy metrics for merge ("test"). [Default = Run]
  --outdir OUTDIR       Output directory.
  --component-graphs COMPONENT_GRAPHS
                        Tab-separated list of paths to Panaroo output directories of component subgraphs. Each directory must contain final_graph.gml and pan_genome_reference.fa. If running in test mode, must also contain gene_data.csv. Graphs will be merged in the order presented in the file.
  --iterative ITERATIVE
                        Tab-separated list of GFFs and their sample IDs for iterative updating of the graph. Use only for single samples or sets of samples too diverse to create an initial pan-genome. Samples will be merged in the order presented in the file.
  --graph-all GRAPH_ALL
                        Path to Panaroo output directory of pan-genome gene graph created from all samples in component-graphs. Only required for the test case, where it is used as the ground truth.
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
