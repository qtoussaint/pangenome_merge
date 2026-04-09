# Local fix notes

## Missing-node guard in context similarity

- File: `pangenomerge/custom_functions/context_similarity.py`
- Function: `context_similarity_seq`
- Change: added `if nA not in G or nB not in G: return 0.0` before neighbor lookups.
- Why: MMSeqs candidate pairs can reference node ids that are no longer present after relabel/merge steps.
- Effect: prevents node lookup failures during context similarity scoring and skips only invalid pairs.
