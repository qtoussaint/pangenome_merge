#!/bin/bash
#SBATCH --job-name=pi
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# One array task = one chunk of genes, measured serially by gene_pi.py.
#
# Sizing comes from the gac alignments (group_1953_g1, 45691 sequences x 1986
# nt, 93 MB): 0.4 s to read and parse, 0.7 s to count amino acids, 0.5 s more
# for nucleotides - ~1.5 s for the largest gene, essentially all I/O. That is
# ~100x cheaper per gene than the dN/dS fit, hence chunks 10x larger and a
# tenth of the resources. numpy releases the GIL but nothing here is threaded,
# so one cpu is the right ask. Peak memory is set by --block-cells, not by gene
# size: 4e6 cells is ~32 MB of int64 index, so 4G is mostly headroom for the
# file buffer. All of these are overridden from the sbatch command line in
# run_pi.sh.
#
# Usage: sbatch --array=0-N%C pi_array.sh <outdir>/pi_params.sh

set -euo pipefail

PARAMS="${1:?usage: pi_array.sh <pi_params.sh>}"
# shellcheck source=/dev/null
source "$PARAMS"

echo "[pi] task ${SLURM_ARRAY_TASK_ID} start: $(date -Is) on $(hostname)"
echo "[pi] params: $PARAMS"

# -u: keep per-gene progress visible in the log while the task is still
# running, instead of block-buffered until it exits
exec "$PI_PYTHON" -u "$PI_DIR/gene_pi.py" \
    --alignments "$PI_OUTDIR/alignments.tsv" \
    --outdir "$PI_OUTDIR" \
    --chunk-index "$SLURM_ARRAY_TASK_ID" \
    --chunk-size "$PI_CHUNK_SIZE" \
    --min-occupancy "$PI_MIN_OCCUPANCY" \
    --min-seqs "$PI_MIN_SEQS" \
    --min-sites "$PI_MIN_SITES" \
    --block-cells "$PI_BLOCK_CELLS" \
    --check-sequences "$PI_CHECK_SEQUENCES"
