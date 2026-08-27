#!/usr/bin/env python3
"""Gather per-gene TOMBOMBADIL fits into one node,omega table.

Runs once, after the array finishes, as a dependent job submitted by
run_dnds.sh. The node list comes from alignments.tsv rather than from whatever
happens to be in raw/, so every gene that went in comes out: genes with no
usable fit get an empty omega. Per-gene detail (sequence count, how much of the
alignment survived the occupancy filter, why a gene was skipped) goes to the
companion stats table.
"""

import argparse
import csv
import glob
import os
import sys

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


def read_omega(scalar_csv):
    """Pull the omega row out of one {stem}_scalar.csv (variable,value)."""
    try:
        with open(scalar_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("variable") == "omega":
                    return row.get("value", "")
    except (OSError, csv.Error):
        return None
    return None


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
                        help="dN/dS output directory used by the array")
    parser.add_argument("--omega-csv", default=None,
                        help="output CSV (default: <outdir>/gene_omega.csv)")
    parser.add_argument("--stats-tsv", default=None,
                        help="output stats TSV (default: <outdir>/dnds_stats.tsv)")
    args = parser.parse_args(argv)

    alignments = os.path.join(args.outdir, "alignments.tsv")
    raw_dir = os.path.join(args.outdir, "raw")
    omega_csv = args.omega_csv or os.path.join(args.outdir, "gene_omega.csv")
    stats_tsv = args.stats_tsv or os.path.join(args.outdir, "dnds_stats.tsv")

    with open(alignments) as handle:
        nodes = [line.rstrip("\n").split("\t")[0]
                 for line in handle if line.strip()]

    stats = read_stats(os.path.join(args.outdir, "stats"))

    tallies = {}
    n_with_omega = 0
    with open(omega_csv, "w", newline="") as omega_handle, \
            open(stats_tsv, "w", newline="") as stats_handle:
        omega_writer = csv.writer(omega_handle)
        omega_writer.writerow(["node", "omega"])
        stats_writer = csv.DictWriter(stats_handle,
                                      fieldnames=STATS_HEADER + ["omega"],
                                      delimiter="\t")
        stats_writer.writeheader()

        for node in nodes:
            omega = read_omega(os.path.join(raw_dir, node + "_scalar.csv"))
            omega_writer.writerow([node, "" if omega is None else omega])
            if omega is not None:
                n_with_omega += 1

            row = dict(stats.get(node, {"node": node}))
            if not row.get("status"):
                # no stats row: either the chunk never ran, or its stats file
                # was lost while the fit itself survived
                row["status"] = "not_run" if omega is None else "fitted_no_stats"
            row["omega"] = "" if omega is None else omega
            stats_writer.writerow({k: row.get(k, "") for k in
                                   STATS_HEADER + ["omega"]})
            tallies[row["status"]] = tallies.get(row["status"], 0) + 1

    print(f"nodes: {len(nodes)}")
    if nodes:
        print(f"omega estimated: {n_with_omega} "
              f"({100 * n_with_omega / len(nodes):.1f}%)")
    for status, count in sorted(tallies.items()):
        print(f"  {status}: {count}")
    print(f"wrote {omega_csv}")
    print(f"wrote {stats_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
