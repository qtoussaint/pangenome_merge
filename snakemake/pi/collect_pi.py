#!/usr/bin/env python3
"""Gather the per-chunk pi measurements into one node,pi_aa,pi_nt table.

Runs once, after the array finishes, as a dependent job submitted by run_pi.sh.
The node list comes from alignments.tsv rather than from whatever happens to be
in stats/, so every gene that went in comes out: genes with no usable
measurement get blank values. Per-gene detail - sequence counts, how much of the
alignment survived the occupancy filter, which filter the nucleotides were
measured under, why a gene was skipped - goes to the companion stats table.
"""

import argparse
import csv
import glob
import os
import sys

STATS_HEADER = [
    "node",
    "n_seqs_aa",
    "n_seqs_nt",
    "n_aa_sites_total",
    "n_aa_sites_kept",
    "frac_aa_kept",
    "n_nt_sites_total",
    "n_nt_sites_kept",
    "frac_nt_kept",
    "pi_aa",
    "pi_nt",
    "filter_mode",
    "status",
    "seconds",
    "message",
]


def read_stats(stats_dir):
    """Merge the per-chunk stats files, keyed by node."""
    stats = {}
    for path in sorted(glob.glob(os.path.join(stats_dir, "chunk_*.tsv"))):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                stats[row["node"]] = row
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", required=True,
                        help="pi output directory used by the array")
    parser.add_argument("--pi-csv", default=None,
                        help="output CSV (default: <outdir>/gene_pi.csv)")
    parser.add_argument("--stats-tsv", default=None,
                        help="output stats TSV (default: <outdir>/pi_stats.tsv)")
    args = parser.parse_args(argv)

    alignments = os.path.join(args.outdir, "alignments.tsv")
    pi_csv = args.pi_csv or os.path.join(args.outdir, "gene_pi.csv")
    stats_tsv = args.stats_tsv or os.path.join(args.outdir, "pi_stats.tsv")

    with open(alignments) as handle:
        nodes = [line.rstrip("\n").split("\t")[0]
                 for line in handle if line.strip()]

    stats = read_stats(os.path.join(args.outdir, "stats"))

    tallies = {}
    filter_modes = {}
    n_with_pi = 0
    with open(pi_csv, "w", newline="") as pi_handle, \
            open(stats_tsv, "w", newline="") as stats_handle:
        pi_writer = csv.writer(pi_handle)
        pi_writer.writerow(["node", "pi_aa", "pi_nt"])
        stats_writer = csv.DictWriter(stats_handle, fieldnames=STATS_HEADER,
                                      delimiter="\t")
        stats_writer.writeheader()

        for node in nodes:
            row = dict(stats.get(node, {"node": node}))
            if not row.get("status"):
                # no stats row: the chunk never ran, or was killed before it
                # reached this gene
                row["status"] = "not_run"
            pi_writer.writerow([node, row.get("pi_aa", ""),
                                row.get("pi_nt", "")])
            if row.get("pi_aa"):
                n_with_pi += 1
            stats_writer.writerow({k: row.get(k, "") for k in STATS_HEADER})
            tallies[row["status"]] = tallies.get(row["status"], 0) + 1
            mode = row.get("filter_mode")
            if mode:
                filter_modes[mode] = filter_modes.get(mode, 0) + 1

    print(f"nodes: {len(nodes)}")
    if nodes:
        print(f"pi measured: {n_with_pi} "
              f"({100 * n_with_pi / len(nodes):.1f}%)")
    for status, count in sorted(tallies.items()):
        print(f"  {status}: {count}")
    # a run with many "independent" genes means `mafft --add` shifted columns
    # often enough that pi_aa and pi_nt are no longer measured over the same
    # region, which is worth knowing before the numbers are compared
    for mode, count in sorted(filter_modes.items()):
        print(f"  filter_mode {mode}: {count}")
    print(f"wrote {pi_csv}")
    print(f"wrote {stats_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
