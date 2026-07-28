# Pipeline figure

`pangenomerge_pipeline.dot` / `.mmd` are the source for a detailed flowchart of the
pangenomerge pipeline, intended as the basis for a paper figure. Both render the **same**
flow as **one figure with a shared backbone**: the run-mode pipeline is the backbone (blue),
and the two **test-mode-only** additions are highlighted in orange (seqID remap to the
all-isolates ground truth, and the RI/ARI/MI/AMI scoring on the final iteration). The
green inner box is the `collapse_spurious_paralogs` refinement engine; yellow notes call out
the performance **optimizations** at the step where each one applies. Two loops are drawn
with labelled back-edges: the outer sequential pairwise-merge loop and the inner fixed-point
refinement loop.

- **`pangenomerge_pipeline.dot`** — Graphviz source, the authoritative figure (vector output
  for Illustrator-style touch-up).
- **`pangenomerge_pipeline.pdf` / `.svg`** — rendered from the DOT.
- **`pangenomerge_pipeline.mmd`** — Mermaid mirror, for quick iteration / GitHub viewing
  (paste into <https://mermaid.live> or use the VS Code Mermaid preview).

## Render

```bash
# Graphviz (authoritative)
dot -Tpdf pangenomerge_pipeline.dot -o pangenomerge_pipeline.pdf
dot -Tsvg pangenomerge_pipeline.dot -o pangenomerge_pipeline.svg

# Mermaid (optional, needs @mermaid-js/mermaid-cli)
mmdc -i pangenomerge_pipeline.mmd -o pangenomerge_pipeline_mermaid.svg
```
