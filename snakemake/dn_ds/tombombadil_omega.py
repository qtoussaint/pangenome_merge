#!/usr/bin/env python3
"""Fit a scalar dN/dS (omega) to one chunk of strict-codon gene alignments.

Run by dnds_array.sh, once per SLURM array task. Each task takes a contiguous
slice of the alignment list written by run_dnds.sh and, for every gene in it:

  1. drops codon columns occupied by fewer than --min-occupancy of the
     sequences, and
  2. runs TOMBOMBADIL JAX (branch scalar_omega_pi) on what is left.

Why filter, and why before the fit: omega on that branch is a single
alignment-wide scalar (sample.py:make_fn vmaps the per-column model with omega
broadcast), so there is no per-site omega to filter afterwards. TOMBOMBADIL
drops gap and ambiguous codons from its counts but keeps the column, at a
reduced sample size, in both the objective and the F3x4 pi estimate - so a
column present in 2% of sequences still shapes the genome-wide omega. Removing
those columns first is the only point at which it can be done.

What this deliberately does NOT do: re-run any QC that --strict-codons already
performed. Sequences reaching aligned_gene_sequences/ under strict mode have
been checked to be a multiple of 3, to translate exactly to their stored
protein (so no internal stops), and to be free of N runs and degenerate codons;
gaps are written as whole '---' codons. Occupancy is therefore just a '-' test
on the first base of each triplet, and that count is exactly the per-column
sample size TOMBOMBADIL's count_codons() would compute. A cheap format guard
over the first --check-sequences records confirms the file really is a strict
alignment, so a run pointed at the non-strict codon/ tree fails loudly instead
of silently relying on a broken shortcut.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# strict alignments are ACGT plus '-'; accept either case
ALLOWED = np.frombuffer(b"-ACGTacgt", dtype="S1")
GAP = b"-"

STATS_HEADER = [
    "node",
    "n_seqs",
    "n_codons_total",
    "n_codons_kept",
    "frac_kept",
    "status",
    "fit_status",
    "seconds",
    "message",
]


class StrictFormatError(Exception):
    """The alignment does not look like --strict-codons output."""


def iter_fasta(path):
    """Yield (header, sequence) pairs, sequence lines joined."""
    name = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].rstrip("\n")
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def check_strict_format(seq_bytes, n_codons, node):
    """Confirm one record has the shape --strict-codons guarantees.

    Guards the first-base-per-codon shortcut used by count_occupancy: the
    non-strict codon/ tree realigns QC-failing DNA with MAFFT and so can carry
    N and part-gapped codons, neither of which the shortcut would notice.
    """
    if seq_bytes.size != n_codons * 3:
        raise StrictFormatError(
            f"{node}: records have unequal lengths ({seq_bytes.size} vs "
            f"{n_codons * 3}); expected a --strict-codons alignment")
    if not np.isin(seq_bytes, ALLOWED).all():
        bad = sorted({c.decode() for c in seq_bytes[~np.isin(seq_bytes, ALLOWED)]})
        raise StrictFormatError(
            f"{node}: alignment contains {bad}, but --strict-codons output is "
            "only ACGT and '-'. Is this the non-strict codon/ directory rather "
            "than codon_strict/?")
    gaps = (seq_bytes == GAP).reshape(-1, 3)
    if (gaps.any(axis=1) & ~gaps.all(axis=1)).any():
        raise StrictFormatError(
            f"{node}: alignment contains part-gapped codons, but "
            "--strict-codons writes every gap as a whole '---' codon. Is this "
            "the non-strict codon/ directory rather than codon_strict/?")


def count_occupancy(path, node, check_sequences):
    """First pass: per-codon occupancy, without holding the alignment.

    Returns (present_counts, n_seqs, n_codons). present_counts[i] is the number
    of sequences with a real codon at column i, which equals the column sample
    size N that TOMBOMBADIL will compute.
    """
    counts = None
    n_codons = 0
    n_seqs = 0
    for header, seq in iter_fasta(path):
        raw = np.frombuffer(seq.encode(), dtype="S1")
        if counts is None:
            if raw.size == 0 or raw.size % 3:
                raise StrictFormatError(
                    f"{node}: alignment length {raw.size} is not a non-zero "
                    "multiple of 3; expected a codon alignment")
            n_codons = raw.size // 3
            counts = np.zeros(n_codons, dtype=np.int64)
        if check_sequences < 0 or n_seqs < check_sequences:
            check_strict_format(raw, n_codons, node)
        elif raw.size != n_codons * 3:
            raise StrictFormatError(
                f"{node}: record {header!r} has length {raw.size}, expected "
                f"{n_codons * 3}")
        # gaps are whole codons, so the first base decides the whole codon
        counts += raw[::3] != GAP
        n_seqs += 1
    if counts is None:
        raise StrictFormatError(f"{node}: alignment file is empty")
    return counts, n_seqs, n_codons


def write_filtered(path, out_path, keep):
    """Second pass: rewrite the alignment keeping only `keep` codon columns."""
    keep_bases = np.repeat(keep, 3)
    with open(out_path, "w") as out:
        for header, seq in iter_fasta(path):
            raw = np.frombuffer(seq.encode(), dtype="S1")
            out.write(">" + header + "\n")
            out.write(raw[keep_bases].tobytes().decode() + "\n")


def run_tombombadil(alignment, stem, args):
    """Invoke the scalar_omega_pi fit, mirroring the validated gac run."""
    command = [
        args.python, "-m", "tombombadil",
        "--alignment", alignment,
        "--pi", "F3x4",
        "--fit-replicates", str(args.fit_replicates),
        "--fit-until-convergence",
        "--cpus", str(args.cpus),
        "--output-jax", stem,
    ]
    env = dict(os.environ)
    # sample.py calls plt.show(); keep it headless on compute nodes
    env["MPLBACKEND"] = "Agg"
    # the checkout is not pip-installed, so -m tombombadil needs it as cwd
    return subprocess.run(command, cwd=args.tombombadil_repo, env=env,
                          capture_output=True, text=True,
                          timeout=args.gene_timeout)


def process_gene(node, path, args, raw_dir, tmpdir):
    """Filter and fit one gene. Returns a stats row dict."""
    started = time.time()
    row = {
        "node": node,
        "n_seqs": "",
        "n_codons_total": "",
        "n_codons_kept": "",
        "frac_kept": "",
        "status": "",
        "fit_status": "",
        "seconds": "",
        "message": "",
    }
    stem = os.path.join(raw_dir, node)
    scalar_csv = stem + "_scalar.csv"

    def finish(status, message=""):
        row["status"] = status
        row["message"] = message
        row["seconds"] = f"{time.time() - started:.1f}"
        return row

    if os.path.exists(scalar_csv) and os.path.getsize(scalar_csv) > 0:
        return finish("cached", "existing fit reused")

    try:
        counts, n_seqs, n_codons = count_occupancy(path, node,
                                                   args.check_sequences)
    except (OSError, ValueError) as err:
        return finish("read_error", str(err))
    # StrictFormatError is deliberately not caught: a mis-pointed --msa-dir is a
    # run-wide mistake and should kill the task, not become one NA row

    keep = counts >= args.min_occupancy * n_seqs
    n_kept = int(keep.sum())
    row["n_seqs"] = n_seqs
    row["n_codons_total"] = n_codons
    row["n_codons_kept"] = n_kept
    row["frac_kept"] = f"{n_kept / n_codons:.4f}"

    if n_seqs < args.min_seqs:
        return finish("too_few_sequences",
                      f"{n_seqs} sequences < --min-seqs {args.min_seqs}")
    if n_kept < args.min_codons:
        return finish("too_few_codons",
                      f"{n_kept} codons at occupancy >= {args.min_occupancy} "
                      f"< --min-codons {args.min_codons}")

    if n_kept == n_codons:
        # nothing to drop: fit the original file rather than copying it
        alignment = path
        filtered = None
    else:
        filtered = os.path.join(tmpdir, node + ".filt.aln.fas")
        write_filtered(path, filtered, keep)
        alignment = filtered

    try:
        result = run_tombombadil(alignment, stem, args)
    except subprocess.TimeoutExpired:
        return finish("timeout", f"exceeded --gene-timeout {args.gene_timeout}s")
    finally:
        if filtered is not None and os.path.exists(filtered):
            # ~50 MB per large gene; keeping these would dwarf the results
            os.remove(filtered)

    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-15:]
        print(f"[{node}] tombombadil failed (exit {result.returncode}):",
              file=sys.stderr)
        for line in tail:
            print(f"[{node}]   {line}", file=sys.stderr)
        return finish("fit_error", f"exit {result.returncode}")

    if not (os.path.exists(scalar_csv) and os.path.getsize(scalar_csv) > 0):
        return finish("no_output", f"{scalar_csv} not written")

    for line in (result.stdout or "").splitlines():
        if line.startswith("Optimization status:"):
            # e.g. "converged after 240 step(s)" vs "reached max steps after
            # 500 step(s)" - the latter means the estimate is not trustworthy
            row["fit_status"] = line.split(":", 1)[1].strip()
        if line.startswith("Final likelihood") or line.startswith("Optimization status"):
            print(f"[{node}] {line.strip()}")
    return finish("fitted")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alignments", required=True,
                        help="TSV of node<TAB>path written by run_dnds.sh")
    parser.add_argument("--outdir", required=True,
                        help="dN/dS output directory")
    parser.add_argument("--chunk-index", type=int, required=True,
                        help="0-based chunk to process (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--chunk-size", type=int, required=True,
                        help="number of genes per chunk")
    parser.add_argument("--tombombadil-repo", required=True,
                        help="TOMBOMBADIL_jax checkout on branch scalar_omega_pi")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter with the tombombadil dependencies")
    parser.add_argument("--min-occupancy", type=float, default=0.2,
                        help="keep codon columns present in at least this "
                             "fraction of sequences (default: %(default)s)")
    parser.add_argument("--min-seqs", type=int, default=20,
                        help="skip genes with fewer sequences; a scalar omega "
                             "fitted to a handful of sequences is too noisy to "
                             "interpret (default: %(default)s)")
    parser.add_argument("--min-codons", type=int, default=30,
                        help="skip genes with fewer retained codons "
                             "(default: %(default)s)")
    parser.add_argument("--fit-replicates", type=int, default=4,
                        help="tombombadil --fit-replicates (default: %(default)s)")
    parser.add_argument("--cpus", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                        help="tombombadil --cpus (default: SLURM_CPUS_PER_TASK)")
    parser.add_argument("--gene-timeout", type=int, default=1800,
                        help="seconds before abandoning one gene "
                             "(default: %(default)s)")
    parser.add_argument("--check-sequences", type=int, default=100,
                        help="records per gene given the full strict-format "
                             "check; -1 for all (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with open(args.alignments) as handle:
        genes = [line.rstrip("\n").split("\t") for line in handle if line.strip()]

    start = args.chunk_index * args.chunk_size
    chunk = genes[start:start + args.chunk_size]
    if not chunk:
        print(f"chunk {args.chunk_index} is empty ({len(genes)} genes total)")
        return 0

    raw_dir = os.path.join(args.outdir, "raw")
    stats_dir = os.path.join(args.outdir, "stats")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"chunk_{args.chunk_index}.tsv")

    print(f"chunk {args.chunk_index}: genes {start}-{start + len(chunk) - 1} "
          f"of {len(genes)}, {args.cpus} cpus, "
          f"min-occupancy {args.min_occupancy}")

    # a resubmitted chunk refits nothing, so carry over what the first run
    # measured rather than rewriting the stats file with blanks
    prior = {}
    if os.path.exists(stats_path):
        with open(stats_path, newline="") as handle:
            for old_row in csv.DictReader(handle, delimiter="\t"):
                prior[old_row["node"]] = old_row

    tallies = {}
    with tempfile.TemporaryDirectory(
            dir=os.environ.get("TMPDIR")) as tmpdir, \
            open(stats_path, "w", newline="") as stats_handle:
        writer = csv.DictWriter(stats_handle, fieldnames=STATS_HEADER,
                                delimiter="\t")
        writer.writeheader()
        for node, path in chunk:
            row = process_gene(node, path, args, raw_dir, tmpdir)
            if row["status"] == "cached" and node in prior:
                for key in ("n_seqs", "n_codons_total", "n_codons_kept",
                            "frac_kept", "fit_status"):
                    row[key] = prior[node].get(key, "")
            writer.writerow(row)
            # flush per gene so a walltime kill still leaves a usable record
            stats_handle.flush()
            tallies[row["status"]] = tallies.get(row["status"], 0) + 1
            print(f"[{node}] {row['status']} ({row['seconds']}s) "
                  f"{row['n_codons_kept']}/{row['n_codons_total']} codons, "
                  f"{row['n_seqs']} seqs")

    summary = ", ".join(f"{k}={v}" for k, v in sorted(tallies.items()))
    print(f"chunk {args.chunk_index} done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
