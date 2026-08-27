#!/bin/bash
# Submit per-gene dN/dS (TOMBOMBADIL JAX, branch scalar_omega_pi) for a finished
# pangenomerge run, and write <outdir>/gene_omega.csv with columns node,omega.
#
# Run this on a login node once the `msa` rule has completed; it submits a SLURM
# job array over chunks of genes plus a dependent collect job, then returns.
#
#   ./run_dnds.sh --results-dir /path/to/project/results
#
# A job array is used rather than one job per gene: a merged species pangenome
# has O(10^4) genes, each taking ~2-3 min, so per-job scheduling overhead and
# submission limits would dominate. One array (MaxArraySize is 80000 here) with
# --chunk-size genes per task amortises that, throttles with a single %N, and
# lets a failed chunk be resubmitted by index.

set -euo pipefail

DNDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# defaults: the known-good paths from the validated gac run
RESULTS_DIR=""
MSA_DIR=""
FORCE_MSA_DIR=0
OUTDIR=""
CHUNK_SIZE=50
MAX_CONCURRENT=100
MIN_OCCUPANCY=0.2
MIN_SEQS=20
MIN_CODONS=30
FIT_REPLICATES=4
GENE_TIMEOUT=1800
CHECK_SEQUENCES=100
CPUS=8
MEM=8G
TIME=05:00:00
PARTITION=standard
SLURM_ACCOUNT=""
REPO=/hps/software/users/jlees/jacqueline/TOMBOMBADIL_jax/scalar_omega_pi
PYTHON=/hps/software/users/jlees/jacqueline/envs/tombombadil/bin/python
BRANCH=scalar_omega_pi
DRY_RUN=0

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<USAGE

Options:
  --results-dir DIR    pipeline results directory (contains msa/); the strict
                       alignments are taken from
                       DIR/msa/codon_strict/aligned_gene_sequences
  --msa-dir DIR        use this alignment directory instead. Refused unless it
                       sits under codon_strict/, or --force-msa-dir is given
  --force-msa-dir      allow a --msa-dir outside codon_strict/
  --outdir DIR         output directory (default: <results-dir>/dnds)
  --chunk-size N       genes per array task (default: $CHUNK_SIZE)
  --max-concurrent N   concurrent array tasks (default: $MAX_CONCURRENT)
  --min-occupancy F    keep codons present in >= F of sequences (default: $MIN_OCCUPANCY)
  --min-seqs N         skip genes with fewer sequences (default: $MIN_SEQS)
  --min-codons N       skip genes with fewer retained codons (default: $MIN_CODONS)
  --fit-replicates N   tombombadil --fit-replicates (default: $FIT_REPLICATES)
  --gene-timeout S     abandon a gene after S seconds (default: $GENE_TIMEOUT)
  --check-sequences N  records per gene given the full strict-format check,
                       -1 for all (default: $CHECK_SEQUENCES)
  --cpus N             cpus per array task (default: $CPUS)
  --mem SIZE           memory per array task (default: $MEM)
  --time HH:MM:SS      walltime per array task (default: $TIME)
  --partition NAME     SLURM partition (default: $PARTITION)
  --slurm-account NAME SLURM account
  --tombombadil-repo D TOMBOMBADIL_jax checkout (default: $REPO)
  --python PATH        interpreter with tombombadil's dependencies
  --dry-run            write the inputs and print the sbatch commands only
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
        --msa-dir)          MSA_DIR="$2"; shift 2 ;;
        --force-msa-dir)    FORCE_MSA_DIR=1; shift ;;
        --outdir)           OUTDIR="$2"; shift 2 ;;
        --chunk-size)       CHUNK_SIZE="$2"; shift 2 ;;
        --max-concurrent)   MAX_CONCURRENT="$2"; shift 2 ;;
        --min-occupancy)    MIN_OCCUPANCY="$2"; shift 2 ;;
        --min-seqs)         MIN_SEQS="$2"; shift 2 ;;
        --min-codons)       MIN_CODONS="$2"; shift 2 ;;
        --fit-replicates)   FIT_REPLICATES="$2"; shift 2 ;;
        --gene-timeout)     GENE_TIMEOUT="$2"; shift 2 ;;
        --check-sequences)  CHECK_SEQUENCES="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --mem)              MEM="$2"; shift 2 ;;
        --time)             TIME="$2"; shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --slurm-account)    SLURM_ACCOUNT="$2"; shift 2 ;;
        --tombombadil-repo) REPO="$2"; shift 2 ;;
        --python)           PYTHON="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  usage >&2; die "unknown argument: $1" ;;
    esac
done

### resolve the alignment directory

# --strict-codons is the whole point: the non-strict codon/ tree realigns
# QC-failing DNA with MAFFT, so it can contain N and part-gapped codons that the
# occupancy filter's one-base-per-codon shortcut assumes are absent.
if [ -n "$MSA_DIR" ]; then
    # codon_strict/ is what the `msa` rule writes; strict_codons/ is the older
    # hand-run layout. Anything else is assumed to be the non-strict tree.
    case "$(basename "$(dirname "$MSA_DIR")")" in
        codon_strict|strict_codons) ;;
        *)
            [ "$FORCE_MSA_DIR" -eq 1 ] || die \
"--msa-dir is not under codon_strict/: $MSA_DIR
       This pipeline requires --strict-codons alignments. Pass --force-msa-dir
       if you are certain the directory holds strict codon alignments."
            ;;
    esac
else
    [ -n "$RESULTS_DIR" ] || { usage >&2; die "--results-dir or --msa-dir is required"; }
    # matches the `msa` rule: mode=codon_strict, FLAG=--strict-codons
    MSA_DIR="$RESULTS_DIR/msa/codon_strict/aligned_gene_sequences"
fi
[ -d "$MSA_DIR" ] || die "alignment directory not found: $MSA_DIR"

if [ -z "$OUTDIR" ]; then
    [ -n "$RESULTS_DIR" ] || die "--outdir is required when --msa-dir is used"
    OUTDIR="$RESULTS_DIR/dnds"
fi

### check the tombombadil checkout

[ -d "$REPO/tombombadil" ] || die "not a TOMBOMBADIL_jax checkout: $REPO"
[ -x "$PYTHON" ] || die "python not executable: $PYTHON"
repo_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "$repo_branch" != "$BRANCH" ]; then
    die "$REPO is on branch '$repo_branch', expected '$BRANCH'.
       Scalar omega is only produced by the $BRANCH branch:
       git -C $REPO checkout $BRANCH"
fi

### build the gene list

mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/raw" "$OUTDIR/stats"
ALIGNMENTS="$OUTDIR/alignments.tsv"

# node name is the alignment basename, per get_alignment_basename() in
# pangenomerge/alignment_functions/generate_alignments.py
find "$MSA_DIR" -maxdepth 1 -name '*.aln.fas' -type f -size +0 -print \
    | sort \
    | awk -F/ '{name=$NF; sub(/\.aln\.fas$/, "", name); print name "\t" $0}' \
    > "$ALIGNMENTS"

NGENES=$(wc -l < "$ALIGNMENTS")
[ "$NGENES" -gt 0 ] || die "no non-empty *.aln.fas files in $MSA_DIR"
NCHUNKS=$(( (NGENES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

### record every setting, so the array (and any resubmit) is reproducible

PARAMS="$OUTDIR/dnds_params.sh"
cat > "$PARAMS" <<PARAMEOF
# written by run_dnds.sh on $(date -Is)
# source of alignments: $MSA_DIR
DNDS_DIR=$(printf '%q' "$DNDS_DIR")
DNDS_OUTDIR=$(printf '%q' "$OUTDIR")
DNDS_REPO=$(printf '%q' "$REPO")
DNDS_PYTHON=$(printf '%q' "$PYTHON")
DNDS_CHUNK_SIZE=$(printf '%q' "$CHUNK_SIZE")
DNDS_MIN_OCCUPANCY=$(printf '%q' "$MIN_OCCUPANCY")
DNDS_MIN_SEQS=$(printf '%q' "$MIN_SEQS")
DNDS_MIN_CODONS=$(printf '%q' "$MIN_CODONS")
DNDS_FIT_REPLICATES=$(printf '%q' "$FIT_REPLICATES")
DNDS_GENE_TIMEOUT=$(printf '%q' "$GENE_TIMEOUT")
DNDS_CHECK_SEQUENCES=$(printf '%q' "$CHECK_SEQUENCES")
PARAMEOF

echo "alignments : $MSA_DIR"
echo "genes      : $NGENES"
echo "outdir     : $OUTDIR"
echo "array      : 0-$((NCHUNKS - 1))%$MAX_CONCURRENT ($CHUNK_SIZE genes per task)"
echo "resources  : $CPUS cpus, $MEM, $TIME, partition $PARTITION"
echo "tombombadil: $REPO ($repo_branch)"

### submit

acct_args=()
[ -n "$SLURM_ACCOUNT" ] && acct_args=(--account "$SLURM_ACCOUNT")

array_args=(
    --array="0-$((NCHUNKS - 1))%$MAX_CONCURRENT"
    --job-name=dnds
    --partition="$PARTITION"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --output="$OUTDIR/logs/dnds_%A_%a.out"
    --error="$OUTDIR/logs/dnds_%A_%a.err"
    "${acct_args[@]}"
    --parsable
    "$DNDS_DIR/dnds_array.sh" "$PARAMS"
)

collect_cmd="$(printf '%q ' "$PYTHON" "$DNDS_DIR/collect_omega.py" --outdir "$OUTDIR")"

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
    --job-name=dnds_collect \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem=2G \
    --time=00:30:00 \
    --output="$OUTDIR/logs/dnds_collect_%j.out" \
    --error="$OUTDIR/logs/dnds_collect_%j.err" \
    "${acct_args[@]}" \
    --wrap "$collect_cmd")

echo
echo "submitted array   : $ARRAY_JOB"
echo "submitted collect : $COLLECT_JOB (afterany:$ARRAY_JOB)"
echo "result            : $OUTDIR/gene_omega.csv"
echo "per-gene detail   : $OUTDIR/dnds_stats.tsv"
