#!/bin/bash
# Submit per-gene nucleotide and amino-acid diversity (pi) for a finished
# pangenomerge run, and write <outdir>/gene_pi.csv with columns
# node,pi_aa,pi_nt.
#
# Run this on a login node once the `msa` rule has completed; it submits a SLURM
# job array over chunks of genes plus a dependent collect job, then returns.
#
#   ./run_pi.sh --results-dir /path/to/project/results
#
# A job array is used rather than one job per gene: a merged species pangenome
# has O(10^4) genes, and at ~1.5 s each the per-job scheduling overhead would
# dwarf the work itself. One array (MaxArraySize is 80000 here) with
# --chunk-size genes per task amortises that, throttles with a single %N, and
# lets a failed chunk be resubmitted by index.

set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR=""
MSA_DIR=""
PROTEIN_DIR=""
FORCE_MSA_DIR=0
OUTDIR=""
CHUNK_SIZE=500
MAX_CONCURRENT=50
MIN_OCCUPANCY=0.2
MIN_SEQS=2
MIN_SITES=30
BLOCK_CELLS=4000000
CHECK_SEQUENCES=100
CPUS=1
MEM=4G
TIME=02:00:00
PARTITION=standard
SLURM_ACCOUNT=""
PYTHON="$(command -v python3 || true)"
DRY_RUN=0

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<USAGE

Options:
  --results-dir DIR    pipeline results directory (contains msa/); the codon
                       alignments are taken from
                       DIR/msa/codon/aligned_gene_sequences and the protein
                       alignments from DIR/msa/alignment/aligned_protein_sequences
  --msa-dir DIR        use this codon alignment directory instead. Refused
                       unless it sits under codon/, or --force-msa-dir is given
  --protein-dir DIR    use this protein alignment directory instead
  --force-msa-dir      allow a --msa-dir outside codon/
  --outdir DIR         output directory (default: <results-dir>/pi)
  --chunk-size N       genes per array task (default: $CHUNK_SIZE)
  --max-concurrent N   concurrent array tasks (default: $MAX_CONCURRENT)
  --min-occupancy F    keep columns present in >= F of sequences (default: $MIN_OCCUPANCY)
  --min-seqs N         skip genes with fewer sequences (default: $MIN_SEQS)
  --min-sites N        skip genes with fewer retained amino-acid sites (default: $MIN_SITES)
  --block-cells N      alignment cells counted per numpy block (default: $BLOCK_CELLS)
  --check-sequences N  records per gene whose IDs are compared between the
                       protein and dna alignments, -1 for all (default: $CHECK_SEQUENCES)
  --cpus N             cpus per array task (default: $CPUS)
  --mem SIZE           memory per array task (default: $MEM)
  --time HH:MM:SS      walltime per array task (default: $TIME)
  --partition NAME     SLURM partition (default: $PARTITION)
  --slurm-account NAME SLURM account
  --python PATH        interpreter with numpy (default: $PYTHON)
  --dry-run            write the inputs and print the sbatch commands only
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
        --msa-dir)          MSA_DIR="$2"; shift 2 ;;
        --protein-dir)      PROTEIN_DIR="$2"; shift 2 ;;
        --force-msa-dir)    FORCE_MSA_DIR=1; shift ;;
        --outdir)           OUTDIR="$2"; shift 2 ;;
        --chunk-size)       CHUNK_SIZE="$2"; shift 2 ;;
        --max-concurrent)   MAX_CONCURRENT="$2"; shift 2 ;;
        --min-occupancy)    MIN_OCCUPANCY="$2"; shift 2 ;;
        --min-seqs)         MIN_SEQS="$2"; shift 2 ;;
        --min-sites)        MIN_SITES="$2"; shift 2 ;;
        --block-cells)      BLOCK_CELLS="$2"; shift 2 ;;
        --check-sequences)  CHECK_SEQUENCES="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --mem)              MEM="$2"; shift 2 ;;
        --time)             TIME="$2"; shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --slurm-account)    SLURM_ACCOUNT="$2"; shift 2 ;;
        --python)           PYTHON="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  usage >&2; die "unknown argument: $1" ;;
    esac
done

### resolve the alignment directories

# The non-strict tree is the whole point here, and it is the opposite of the
# choice dN/dS makes: pi measures standing diversity, so it wants every isolate,
# including the ones --strict-codons drops for failing translation QC. Nothing
# in gene_pi.py relies on strict mode's guarantees - N, degenerate bases and
# part-gapped codons are all just missing data.
if [ -n "$MSA_DIR" ]; then
    # codon/ is what the `msa` rule writes; codons/ is the older hand-run
    # layout. Anything else is assumed to be the strict tree.
    case "$(basename "$(dirname "$MSA_DIR")")" in
        codon|codons) ;;
        *)
            [ "$FORCE_MSA_DIR" -eq 1 ] || die \
"--msa-dir is not under codon/: $MSA_DIR
       This pipeline wants the non-strict --codons alignments, which keep the
       sequences codon_strict/ drops. Pass --force-msa-dir if you are certain
       this is what you want."
            ;;
    esac
else
    [ -n "$RESULTS_DIR" ] || { usage >&2; die "--results-dir or --msa-dir is required"; }
    # matches the `msa` rule: mode=codon, FLAG=--codons
    MSA_DIR="$RESULTS_DIR/msa/codon/aligned_gene_sequences"
fi
[ -d "$MSA_DIR" ] || die "codon alignment directory not found: $MSA_DIR"

# The protein alignment is shared between the two codon modes. In the hand-run
# layout it sits beside aligned_gene_sequences/ inside the mode directory; the
# Snakemake `msa` rule puts it in --shared-alignment-dir instead.
if [ -z "$PROTEIN_DIR" ]; then
    for candidate in \
        "$(dirname "$MSA_DIR")/aligned_protein_sequences" \
        "${RESULTS_DIR:+$RESULTS_DIR/msa/alignment/aligned_protein_sequences}"
    do
        [ -n "$candidate" ] && [ -d "$candidate" ] && { PROTEIN_DIR="$candidate"; break; }
    done
fi
[ -n "$PROTEIN_DIR" ] || die \
"no protein alignment directory found next to $MSA_DIR
       Looked for aligned_protein_sequences/ beside it and under
       <results-dir>/msa/alignment/. Pass --protein-dir explicitly."
[ -d "$PROTEIN_DIR" ] || die "protein alignment directory not found: $PROTEIN_DIR"

if [ -z "$OUTDIR" ]; then
    [ -n "$RESULTS_DIR" ] || die "--outdir is required when --msa-dir is used"
    OUTDIR="$RESULTS_DIR/pi"
fi

### check the interpreter

[ -n "$PYTHON" ] && [ -x "$PYTHON" ] || die "python not executable: ${PYTHON:-<none found>}"
"$PYTHON" -c "import numpy" 2>/dev/null || die \
"$PYTHON cannot import numpy, which gene_pi.py requires.
       Pass --python with an interpreter that has it."

### build the gene list

mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/stats"
ALIGNMENTS="$OUTDIR/alignments.tsv"

# node name is the alignment basename, per get_alignment_basename() in
# pangenomerge/alignment_functions/generate_alignments.py, and is the same on
# both sides. A gene with no protein alignment keeps an empty third field rather
# than being dropped, so it still appears in gene_pi.csv as a blank row.
# -L, unlike the dN/dS stage's plain find: a symlinked alignment (or a
# symlinked directory of them) is a normal way to assemble an ad-hoc gene set,
# and -type f would silently skip every one of them
find -L "$MSA_DIR" -maxdepth 1 -name '*.aln.fas' -type f -size +0 -print \
    | sort \
    | awk -F/ -v prot="$PROTEIN_DIR" '{
          name = $NF; sub(/\.aln\.fas$/, "", name)
          p = prot "/" name ".aln.fas"
          ok = ((getline line < p) > 0)
          close(p)
          print name "\t" $0 "\t" (ok ? p : "")
      }' \
    > "$ALIGNMENTS"

NGENES=$(wc -l < "$ALIGNMENTS")
[ "$NGENES" -gt 0 ] || die "no non-empty *.aln.fas files in $MSA_DIR"
NOPROT=$(awk -F'\t' '$3 == ""' "$ALIGNMENTS" | wc -l)
NCHUNKS=$(( (NGENES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

### record every setting, so the array (and any resubmit) is reproducible

PARAMS="$OUTDIR/pi_params.sh"
cat > "$PARAMS" <<PARAMEOF
# written by run_pi.sh on $(date -Is)
# source of codon alignments  : $MSA_DIR
# source of protein alignments: $PROTEIN_DIR
PI_DIR=$(printf '%q' "$PI_DIR")
PI_OUTDIR=$(printf '%q' "$OUTDIR")
PI_PYTHON=$(printf '%q' "$PYTHON")
PI_CHUNK_SIZE=$(printf '%q' "$CHUNK_SIZE")
PI_MIN_OCCUPANCY=$(printf '%q' "$MIN_OCCUPANCY")
PI_MIN_SEQS=$(printf '%q' "$MIN_SEQS")
PI_MIN_SITES=$(printf '%q' "$MIN_SITES")
PI_BLOCK_CELLS=$(printf '%q' "$BLOCK_CELLS")
PI_CHECK_SEQUENCES=$(printf '%q' "$CHECK_SEQUENCES")
PARAMEOF

echo "codon alns  : $MSA_DIR"
echo "protein alns: $PROTEIN_DIR"
echo "genes       : $NGENES ($NOPROT with no protein alignment)"
echo "outdir      : $OUTDIR"
echo "array       : 0-$((NCHUNKS - 1))%$MAX_CONCURRENT ($CHUNK_SIZE genes per task)"
echo "resources   : $CPUS cpus, $MEM, $TIME, partition $PARTITION"
echo "python      : $PYTHON"

### submit

acct_args=()
[ -n "$SLURM_ACCOUNT" ] && acct_args=(--account "$SLURM_ACCOUNT")

array_args=(
    --array="0-$((NCHUNKS - 1))%$MAX_CONCURRENT"
    --job-name=pi
    --partition="$PARTITION"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --output="$OUTDIR/logs/pi_%A_%a.out"
    --error="$OUTDIR/logs/pi_%A_%a.err"
    "${acct_args[@]}"
    --parsable
    "$PI_DIR/pi_array.sh" "$PARAMS"
)

collect_cmd="$(printf '%q ' "$PYTHON" "$PI_DIR/collect_pi.py" --outdir "$OUTDIR")"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "dry run, would submit:"
    printf '  sbatch'; printf ' %q' "${array_args[@]}"; printf '\n'
    printf '  sbatch --dependency=afterany:<jobid> ... --wrap %q\n' "$collect_cmd"
    exit 0
fi

ARRAY_JOB=$(sbatch "${array_args[@]}")

# afterany, not afterok: genes that fail are meant to land in the CSV as blanks,
# so the table should still be written when some chunks die
COLLECT_JOB=$(sbatch --parsable \
    --dependency=afterany:"$ARRAY_JOB" \
    --job-name=pi_collect \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem=2G \
    --time=00:30:00 \
    --output="$OUTDIR/logs/pi_collect_%j.out" \
    --error="$OUTDIR/logs/pi_collect_%j.err" \
    "${acct_args[@]}" \
    --wrap "$collect_cmd")

echo
echo "submitted array   : $ARRAY_JOB"
echo "submitted collect : $COLLECT_JOB (afterany:$ARRAY_JOB)"
echo "result            : $OUTDIR/gene_pi.csv"
echo "per-gene detail   : $OUTDIR/pi_stats.tsv"
