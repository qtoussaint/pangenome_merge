#!/usr/bin/env python3
"""Measure per-gene nucleotide and amino-acid diversity for one chunk of genes.

Run by pi_array.sh, once per SLURM array task. Each task takes a contiguous
slice of the alignment list written by run_pi.sh and, for every gene in it,
reports two numbers: pi_aa from the protein alignment and pi_nt from the
non-strict codon alignment.

Both are whole-gene values, not per-site series. Per column, with pairwise
deletion of missing data,

    n_j     = sequences with a real state at column j
    p_i     = count of state i / n_j
    pi_j    = n_j/(n_j - 1) * (1 - sum_i p_i^2)
    pi_gene = mean of pi_j over retained columns with n_j >= 2

Counting states rather than comparing sequence pairs is what makes this
tractable: a 46k-sequence gene has ~1.05e9 pairs. The two definitions agree
when every pair has complete data, and the retained column count is reported so
the value can be renormalised.

Why the non-strict alignments, when the dN/dS stage refuses them: pi measures
standing diversity, so it wants every isolate, including the ones --strict-codons
drops for failing translation QC. Nothing here relies on the guarantees strict
mode provides - N, degenerate bases and part-gapped codons are simply missing
data - so the looser alignment costs nothing and covers more sequences.

The one thing the non-strict tree does cost is column correspondence. The
protein alignment is shared between --codons and --strict-codons and is built
before QC; non-strict mode then adds the QC-failing DNA with `mafft --add`,
which can insert columns into the codon alignment. len(dna) == 3 * len(protein)
therefore holds only when nothing failed QC, and is checked per gene rather than
assumed - see decide_nt_filter().
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# state codes: real states occupy 0..N_REAL-1, everything else is missing
N_BASES = 4
N_AMINO = 20

BASE_LUT = np.full(256, N_BASES, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    BASE_LUT[ord(_b)] = _i
    BASE_LUT[ord(_b.lower())] = _i

# '-', '*', 'X' and the ambiguity codes B/Z/J/U/O all fall through to missing.
# '*' among them is the stop-codon decision: an internal stop in a non-strict
# alignment comes from `mafft --add` re-aligning a broken ORF, so counting it as
# a 21st state would let a handful of frameshifted records dominate a column.
AMINO_LUT = np.full(256, N_AMINO, dtype=np.uint8)
for _i, _a in enumerate("ACDEFGHIKLMNPQRSTVWY"):
    AMINO_LUT[ord(_a)] = _i
    AMINO_LUT[ord(_a.lower())] = _i

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


class AlignmentError(Exception):
    """The file is not a usable alignment (ragged, empty, unreadable)."""


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


def count_states(path, lut, n_states, block_cells, keep_headers=0):
    """One pass over an alignment, returning per-column state counts.

    Returns (counts, n_seqs, headers) where counts has shape
    (n_columns, n_states + 1); the last column is the missing-data tally.

    Records are counted a block at a time rather than one by one: reshaping a
    block into an (n_records, n_columns) byte matrix and reducing it with a
    single np.bincount is an order of magnitude faster than per-record work, and
    the whole cost of this stage is one file read plus these two reductions.

    Block size is derived from block_cells rather than fixed, because bincount
    needs an int64 index array the size of the block: a fixed record count that
    is comfortable on a 2 kb gene would build a multi-gigabyte intermediate on a
    30 kb one.
    """
    counts = None
    width = 0
    offsets = None
    rows_per_block = 0
    n_seqs = 0
    headers = []
    buf = []

    def flush():
        block = np.frombuffer(b"".join(buf), dtype=np.uint8).reshape(-1, width)
        codes = lut[block].astype(np.int64)
        codes += offsets
        counts[...] += np.bincount(
            codes.ravel(), minlength=width * (n_states + 1)
        ).reshape(width, n_states + 1)
        buf.clear()

    for header, seq in iter_fasta(path):
        if counts is None:
            width = len(seq)
            if width == 0:
                raise AlignmentError(f"{path}: first record is empty")
            counts = np.zeros((width, n_states + 1), dtype=np.int64)
            offsets = np.arange(width, dtype=np.int64) * (n_states + 1)
            rows_per_block = max(1, block_cells // width)
        elif len(seq) != width:
            raise AlignmentError(
                f"{path}: record {header!r} has length {len(seq)}, expected "
                f"{width}; this is not an alignment")
        if keep_headers < 0 or n_seqs < keep_headers:
            headers.append(header)
        buf.append(seq.encode())
        n_seqs += 1
        if len(buf) == rows_per_block:
            flush()
    if buf:
        flush()
    if counts is None:
        raise AlignmentError(f"{path}: file contains no records")
    return counts, n_seqs, headers


def column_pi(counts, n_states, keep):
    """Mean per-column pi over the kept columns, and how many were used.

    Columns with fewer than two observed sequences are dropped on top of `keep`:
    the n/(n-1) correction is undefined at n = 1, and a column seen once carries
    no information about diversity either way.
    """
    real = counts[:, :n_states]
    n = real.sum(axis=1)
    usable = keep & (n >= 2)
    n_used = int(usable.sum())
    if n_used == 0:
        return None, 0
    obs = real[usable].astype(np.float64)
    n_obs = n[usable].astype(np.float64)
    freq = obs / n_obs[:, None]
    homozygosity = (freq ** 2).sum(axis=1)
    pi = (n_obs / (n_obs - 1.0)) * (1.0 - homozygosity)
    return float(pi.mean()), n_used


def decide_nt_filter(keep_aa, nt_counts, n_seqs_aa, n_seqs_nt,
                     aa_headers, nt_headers, min_occupancy):
    """Choose which codon columns pi_nt is measured over.

    Preferred: the protein alignment's retained columns, mapped to their codon
    triplets, so pi_aa and pi_nt describe exactly the same region of the gene and
    their ratio means something.

    That mapping is only valid if the two files line up - equal record counts in
    the same order, and a DNA alignment exactly three times the protein's width.
    `mafft --add` can break both. When it has, the nucleotide sites are filtered
    on their own occupancy instead and the caller records filter_mode
    "independent": one gene measured over a slightly different region is a far
    better outcome than one measured over mis-paired columns.
    """
    width_nt = nt_counts.shape[0]
    expected = keep_aa.size * 3
    if width_nt != expected:
        reason = (f"dna alignment is {width_nt} columns, expected 3 x "
                  f"{keep_aa.size} = {expected}")
    elif n_seqs_nt != n_seqs_aa:
        reason = (f"{n_seqs_nt} dna records vs {n_seqs_aa} protein records")
    else:
        mismatch = next((i for i, (a, n) in enumerate(zip(aa_headers, nt_headers))
                         if a != n), None)
        reason = None if mismatch is None else (
            f"record {mismatch} is {nt_headers[mismatch]!r} in the dna "
            f"alignment but {aa_headers[mismatch]!r} in the protein alignment")

    if reason is None:
        return np.repeat(keep_aa, 3), "codon", ""

    occupancy = nt_counts[:, :N_BASES].sum(axis=1)
    keep_nt = occupancy >= min_occupancy * n_seqs_nt
    return keep_nt, "independent", (
        "protein and dna alignments do not correspond (" + reason +
        "); nucleotide sites filtered independently")


def process_gene(node, dna_path, protein_path, args):
    """Measure one gene. Returns a stats row dict."""
    started = time.time()
    row = dict.fromkeys(STATS_HEADER, "")
    row["node"] = node

    def finish(status, message=""):
        row["status"] = status
        row["message"] = message
        row["seconds"] = f"{time.time() - started:.1f}"
        return row

    if not protein_path:
        return finish("no_protein", "no protein alignment for this gene")

    try:
        aa_counts, n_seqs_aa, aa_headers = count_states(
            protein_path, AMINO_LUT, N_AMINO, args.block_cells,
            args.check_sequences)
        nt_counts, n_seqs_nt, nt_headers = count_states(
            dna_path, BASE_LUT, N_BASES, args.block_cells,
            args.check_sequences)
    except AlignmentError as err:
        return finish("length_error", str(err))
    except (OSError, ValueError) as err:
        return finish("read_error", str(err))

    row["n_seqs_aa"] = n_seqs_aa
    row["n_seqs_nt"] = n_seqs_nt
    row["n_aa_sites_total"] = aa_counts.shape[0]
    row["n_nt_sites_total"] = nt_counts.shape[0]

    occupancy_aa = aa_counts[:, :N_AMINO].sum(axis=1)
    keep_aa = occupancy_aa >= args.min_occupancy * n_seqs_aa
    keep_nt, filter_mode, note = decide_nt_filter(
        keep_aa, nt_counts, n_seqs_aa, n_seqs_nt, aa_headers, nt_headers,
        args.min_occupancy)
    row["filter_mode"] = filter_mode

    pi_aa, used_aa = column_pi(aa_counts, N_AMINO, keep_aa)
    pi_nt, used_nt = column_pi(nt_counts, N_BASES, keep_nt)
    row["n_aa_sites_kept"] = used_aa
    row["n_nt_sites_kept"] = used_nt
    row["frac_aa_kept"] = f"{used_aa / aa_counts.shape[0]:.4f}"
    row["frac_nt_kept"] = f"{used_nt / nt_counts.shape[0]:.4f}"

    if min(n_seqs_aa, n_seqs_nt) < args.min_seqs:
        return finish("too_few_sequences",
                      f"{min(n_seqs_aa, n_seqs_nt)} sequences < --min-seqs "
                      f"{args.min_seqs}")
    if used_aa < args.min_sites:
        return finish("too_few_sites",
                      f"{used_aa} amino-acid sites at occupancy >= "
                      f"{args.min_occupancy} < --min-sites {args.min_sites}")

    row["pi_aa"] = "" if pi_aa is None else repr(pi_aa)
    row["pi_nt"] = "" if pi_nt is None else repr(pi_nt)
    return finish("computed", note)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alignments", required=True,
                        help="TSV of node<TAB>dna_path<TAB>protein_path "
                             "written by run_pi.sh")
    parser.add_argument("--outdir", required=True, help="pi output directory")
    parser.add_argument("--chunk-index", type=int, required=True,
                        help="0-based chunk to process (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--chunk-size", type=int, required=True,
                        help="number of genes per chunk")
    parser.add_argument("--min-occupancy", type=float, default=0.2,
                        help="keep columns present in at least this fraction "
                             "of sequences (default: %(default)s)")
    parser.add_argument("--min-seqs", type=int, default=2,
                        help="skip genes with fewer sequences; pi is "
                             "undefined for one sequence (default: %(default)s)")
    parser.add_argument("--min-sites", type=int, default=30,
                        help="skip genes with fewer retained amino-acid sites "
                             "(default: %(default)s)")
    parser.add_argument("--block-cells", type=int, default=4_000_000,
                        help="alignment cells counted per numpy block; caps "
                             "peak memory independently of gene length "
                             "(default: %(default)s)")
    parser.add_argument("--check-sequences", type=int, default=100,
                        help="records per gene whose IDs are compared between "
                             "the protein and dna alignments, -1 for all "
                             "(default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with open(args.alignments) as handle:
        genes = [line.rstrip("\n").split("\t")
                 for line in handle if line.strip()]

    start = args.chunk_index * args.chunk_size
    chunk = genes[start:start + args.chunk_size]
    if not chunk:
        print(f"chunk {args.chunk_index} is empty ({len(genes)} genes total)")
        return 0

    stats_dir = os.path.join(args.outdir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"chunk_{args.chunk_index}.tsv")

    print(f"chunk {args.chunk_index}: genes {start}-{start + len(chunk) - 1} "
          f"of {len(genes)}, min-occupancy {args.min_occupancy}")

    tallies = {}
    with open(stats_path, "w", newline="") as stats_handle:
        writer = csv.DictWriter(stats_handle, fieldnames=STATS_HEADER,
                                delimiter="\t")
        writer.writeheader()
        for fields in chunk:
            node, dna_path = fields[0], fields[1]
            protein_path = fields[2] if len(fields) > 2 else ""
            row = process_gene(node, dna_path, protein_path, args)
            writer.writerow(row)
            # flush per gene so a walltime kill still leaves a usable record
            stats_handle.flush()
            tallies[row["status"]] = tallies.get(row["status"], 0) + 1
            print(f"[{node}] {row['status']} ({row['seconds']}s) "
                  f"pi_aa={row['pi_aa'] or 'NA'} pi_nt={row['pi_nt'] or 'NA'} "
                  f"{row['n_aa_sites_kept']}/{row['n_aa_sites_total']} aa sites, "
                  f"{row['n_seqs_aa']} seqs")
            if row["message"]:
                print(f"[{node}]   {row['message']}")

    summary = ", ".join(f"{k}={v}" for k, v in sorted(tallies.items()))
    print(f"chunk {args.chunk_index} done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
