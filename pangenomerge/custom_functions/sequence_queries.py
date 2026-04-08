import zlib
import sqlite3
from typing import Iterator


def get_sequences_for_node(con: sqlite3.Connection, node_id: str,
                           seq_type: str = "nt") -> list[tuple[str, str]]:
    """Return [(geneid, sequence), ...] for all genes in a node.

    Parameters
    ----------
    con : sqlite3.Connection
    node_id : str
        The merged node/COG identifier.
    seq_type : str
        'nt' for nucleotide or 'aa' for amino acid.
    """
    hash_col = "gs.nt_hash" if seq_type == "nt" else "gs.aa_hash"
    cur = con.execute(f"""
        SELECT ng.geneid, us.seq_data
        FROM node_geneids ng
        JOIN gene_sequences gs ON gs.geneid = ng.geneid
        JOIN unique_sequences us
          ON us.seq_hash = {hash_col}
          AND us.seq_type = ?
        WHERE ng.node_id = ?
    """, (seq_type, node_id))

    return [(geneid, zlib.decompress(blob).decode()) for geneid, blob in cur]


def iter_sequences_for_node(con: sqlite3.Connection, node_id: str,
                            seq_type: str = "nt") -> Iterator[tuple[str, str]]:
    """Memory-efficient iterator yielding (geneid, sequence) for a node."""
    hash_col = "gs.nt_hash" if seq_type == "nt" else "gs.aa_hash"
    cur = con.execute(f"""
        SELECT ng.geneid, us.seq_data
        FROM node_geneids ng
        JOIN gene_sequences gs ON gs.geneid = ng.geneid
        JOIN unique_sequences us
          ON us.seq_hash = {hash_col}
          AND us.seq_type = ?
        WHERE ng.node_id = ?
    """, (seq_type, node_id))

    for geneid, blob in cur:
        yield geneid, zlib.decompress(blob).decode()


def get_unique_sequences_for_node(con: sqlite3.Connection, node_id: str,
                                  seq_type: str = "nt") -> dict[str, list[str]]:
    """Return {sequence: [geneid1, geneid2, ...]} — deduplicated view.

    The number of keys equals the number of unique alleles for this node.
    """
    hash_col = "gs.nt_hash" if seq_type == "nt" else "gs.aa_hash"
    cur = con.execute(f"""
        SELECT us.seq_data, GROUP_CONCAT(ng.geneid, ';')
        FROM node_geneids ng
        JOIN gene_sequences gs ON gs.geneid = ng.geneid
        JOIN unique_sequences us
          ON us.seq_hash = {hash_col}
          AND us.seq_type = ?
        WHERE ng.node_id = ?
        GROUP BY us.seq_hash
    """, (seq_type, node_id))

    return {
        zlib.decompress(blob).decode(): geneids.split(";")
        for blob, geneids in cur
    }


def get_sequence_counts_for_node(con: sqlite3.Connection, node_id: str,
                                 seq_type: str = "nt") -> dict[str, int]:
    """Return {sequence: count} — allele frequency spectrum for a node."""
    hash_col = "gs.nt_hash" if seq_type == "nt" else "gs.aa_hash"
    cur = con.execute(f"""
        SELECT us.seq_data, COUNT(*) as cnt
        FROM node_geneids ng
        JOIN gene_sequences gs ON gs.geneid = ng.geneid
        JOIN unique_sequences us
          ON us.seq_hash = {hash_col}
          AND us.seq_type = ?
        WHERE ng.node_id = ?
        GROUP BY us.seq_hash
    """, (seq_type, node_id))

    return {
        zlib.decompress(blob).decode(): cnt
        for blob, cnt in cur
    }


def export_node_fasta(con: sqlite3.Connection, node_id: str, output_path: str,
                      seq_type: str = "nt", unique_only: bool = False) -> int:
    """Write a FASTA file containing sequences for a node.

    Parameters
    ----------
    con : sqlite3.Connection
    node_id : str
        The merged node/COG identifier.
    output_path : str
        Path to write the FASTA file.
    seq_type : str
        'nt' for nucleotide or 'aa' for amino acid.
    unique_only : bool
        If True, write each unique sequence once (header = first geneid).

    Returns
    -------
    int
        Number of sequences written.
    """
    n_written = 0
    with open(output_path, "w") as f:
        if unique_only:
            for seq, geneids in get_unique_sequences_for_node(con, node_id, seq_type).items():
                f.write(f">{geneids[0]}\n{seq}\n")
                n_written += 1
        else:
            for geneid, seq in iter_sequences_for_node(con, node_id, seq_type):
                f.write(f">{geneid}\n{seq}\n")
                n_written += 1
    return n_written
