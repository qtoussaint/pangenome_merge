#!/bin/bash
#SBATCH --job-name=dnds
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# One array task = one chunk of genes, fitted serially by tombombadil_omega.py.
#
# Sizing comes from the gac run (sacct job 27853877, 7 genes of ~46k sequences
# x 270-662 codons on 8 cpus): 1:34-2:32 elapsed and 0.91-1.11 GB MaxRSS per
# gene. The runtime floor is JAX-compile-bound - the 11-sequence gene still took
# 2:26 - so budget ~3 min per gene regardless of size. The defaults above give
# a 50-gene chunk (~2.5 h) roughly 2x walltime headroom and ~7x memory headroom.
# All of them are overridden from the sbatch command line in run_dnds.sh.
#
# Usage: sbatch --array=0-N%C dnds_array.sh <outdir>/dnds_params.sh

set -euo pipefail

PARAMS="${1:?usage: dnds_array.sh <dnds_params.sh>}"
# shellcheck source=/dev/null
source "$PARAMS"

echo "[dnds] task ${SLURM_ARRAY_TASK_ID} start: $(date -Is) on $(hostname)"
echo "[dnds] params: $PARAMS"

# -u: keep per-gene progress visible in the log while the task is still
# running, instead of block-buffered until it exits
exec "$DNDS_PYTHON" -u "$DNDS_DIR/tombombadil_omega.py" \
    --alignments "$DNDS_OUTDIR/alignments.tsv" \
    --outdir "$DNDS_OUTDIR" \
    --chunk-index "$SLURM_ARRAY_TASK_ID" \
    --chunk-size "$DNDS_CHUNK_SIZE" \
    --tombombadil-repo "$DNDS_REPO" \
    --python "$DNDS_PYTHON" \
    --min-occupancy "$DNDS_MIN_OCCUPANCY" \
    --min-seqs "$DNDS_MIN_SEQS" \
    --min-codons "$DNDS_MIN_CODONS" \
    --fit-replicates "$DNDS_FIT_REPLICATES" \
    --gene-timeout "$DNDS_GENE_TIMEOUT" \
    --check-sequences "$DNDS_CHECK_SEQUENCES"
