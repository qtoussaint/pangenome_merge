import argparse
import logging
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from pangenomerge.custom_functions.sqlite import sqlite_connect, sqlite_connect_sequences, sqlite_create_sequence_indexes, ingest_gene_sequences, add_gene_annotations_to_sqlite

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_GN_SUFFIX_RE = re.compile(r'^(.+)_g(\d+)$')


def ingest_gene_annotations(con, component_graphs_tsv):
    """Batch-ingest gene annotations from all component graph directories."""
    graph_files = pd.read_csv(component_graphs_tsv, sep='\t', header=None)
    n_graphs = len(graph_files)
    for idx in range(n_graphs):
        graph_id = idx + 1
        graph_dir = str(graph_files.iloc[idx][0])
        add_gene_annotations_to_sqlite(con, graph_id=graph_id, graph_dir=graph_dir)
    logging.info(f"Ingested gene annotations from {n_graphs} component graphs")


def _derive_gene_name(annotation, used_gene_names, unique_id_counter):
    """Derive a unique gene name from annotation, matching Panaroo's logic."""
    if annotation:
        name = "~~~".join(
            gn for gn in annotation.strip().strip(";").split(";") if gn != ""
        )
        name = "".join(e for e in name if e.isalnum() or e in ["_", "~"])
    else:
        name = ""

    if name and name.lower() not in used_gene_names:
        used_gene_names.add(name.lower())
        return name, unique_id_counter

    gen_name = f"group_{unique_id_counter}"
    unique_id_counter += 1
    used_gene_names.add(gen_name.lower())
    return gen_name, unique_id_counter


def generate_gene_presence_absence(sqlite_path, gml_path, output_dir,
                                   sqlite_cache=2000):
    """Generate Panaroo-format gene presence/absence files from pangenomerge output."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Build isolate mapping and orig_ids from SQLite ---
    con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
    cur = con.cursor()

    cur.execute("SELECT graph_id, member_index, sample_name FROM isolate_names ORDER BY graph_id, member_index")
    isolates = []
    member_to_col = {}
    for graph_id, member_index, sample_name in cur:
        col = len(isolates)
        member_key = f"{member_index}_g{graph_id}"
        member_to_col[member_key] = col
        isolates.append(sample_name)

    n_isolates = len(isolates)
    logging.info(f"Loaded {n_isolates} isolates")

    # --- 2. Build orig_ids from gene_annotations table ---
    cur.execute("SELECT geneid, annotation_id FROM gene_annotations")
    orig_ids = dict(cur.fetchall())
    logging.info(f"Loaded {len(orig_ids)} gene ID -> annotation ID mappings from SQLite")

    # --- 3. Read GML for connected component numbering ---
    G = nx.read_gml(str(gml_path))
    # Map node labels to their connected component and order within it
    node_fragment = {}  # node_id -> (fragment_num, order_within_fragment)
    frag = 0
    for component in nx.connected_components(G):
        frag += 1
        for count, node in enumerate(component, 1):
            node_fragment[str(node)] = (frag, count)
    del G
    logging.info(f"Identified {frag} genome fragments from GML")

    # --- 4. Batch-load node metadata with aggregated lengths ---
    cur.execute("""
        SELECT n.node_id, n.name, n.size, n.annotation, n.description,
               MIN(nl.length), MAX(nl.length),
               CASE WHEN SUM(nl.count) > 0
                    THEN SUM(nl.length * nl.count) * 1.0 / SUM(nl.count)
                    ELSE 0 END
        FROM nodes n
        LEFT JOIN node_lengths nl ON n.node_id = nl.node_id
        GROUP BY n.node_id
    """)
    node_meta = {}
    for row in cur:
        node_id = row[0]
        node_meta[node_id] = {
            'name': row[1],
            'size': row[2] or 0,
            'annotation': row[3],
            'description': row[4],
            'min_len': row[5] or 0,
            'max_len': row[6] or 0,
            'avg_len': row[7] or 0,
        }
    logging.info(f"Loaded metadata for {len(node_meta)} nodes")

    # --- 5. Batch-load members and geneIDs grouped by node ---
    # Members: node_id -> set of column indices
    cur.execute("SELECT node_id, member FROM node_members ORDER BY node_id")
    node_members_map = defaultdict(set)
    for node_id, member in cur:
        col = member_to_col.get(member)
        if col is not None:
            node_members_map[node_id].add(col)

    # GeneIDs: node_id -> list of (geneid, col_index)
    cur.execute("SELECT node_id, geneid, member FROM node_geneids ORDER BY node_id")
    node_geneids_map = defaultdict(list)
    for node_id, geneid, member in cur:
        col = member_to_col.get(member)
        if col is not None:
            node_geneids_map[node_id].append((geneid, col))

    # Count seqIDs per node
    cur.execute("SELECT node_id, COUNT(*) FROM node_seqids GROUP BY node_id")
    node_seqid_counts = dict(cur.fetchall())

    con.close()

    # --- 6. Build entries for each node ---
    used_gene_names = set([""])
    unique_id_counter = 0
    entries = []  # list of (entry_size, entry_roary, entry_simple, pres_abs)

    for node_id, meta in node_meta.items():
        # Derive gene name
        gene_name, unique_id_counter = _derive_gene_name(
            meta['annotation'], used_gene_names, unique_id_counter
        )

        # Fragment info
        frag_num, frag_order = node_fragment.get(node_id, (0, 0))

        # Number of sequences
        n_seqids = node_seqid_counts.get(node_id, 0)
        size = meta['size']
        avg_seqs = (1.0 * n_seqids / size) if size > 0 else 0

        # Build presence/absence array
        pres_abs = [""] * n_isolates
        present_cols = node_members_map.get(node_id, set())

        # Fill in Prokka annotation IDs from geneIDs
        for geneid, col in node_geneids_map.get(node_id, []):
            annot_id = orig_ids.get(geneid, geneid)
            if pres_abs[col] == "":
                pres_abs[col] = annot_id
            else:
                pres_abs[col] += ";" + annot_id

        # For members with no geneID mapping, mark as present with gene name
        for col in present_cols:
            if pres_abs[col] == "":
                pres_abs[col] = gene_name

        entry_size = sum(1 for v in pres_abs if v != "")

        # Roary CSV row: 14 metadata columns + isolate columns
        entry_roary = [
            gene_name,
            meta['annotation'] or "",
            meta['description'] or "",
            str(size),
            str(n_seqids),
            str(avg_seqs),
            str(frag_num),
            str(frag_order),
            "", "", "",  # Accessory Fragment, Accessory Order, QC
            str(meta['min_len']),
            str(meta['max_len']),
            str(meta['avg_len']),
        ] + pres_abs

        # Simple CSV row: 3 metadata columns + isolate columns
        entry_simple = [
            gene_name,
            meta['annotation'] or "",
            meta['description'] or "",
        ] + pres_abs

        entries.append((entry_size, entry_roary, entry_simple, pres_abs, gene_name))

    # --- 7. Sort by prevalence descending ---
    entries.sort(key=lambda x: x[0], reverse=True)

    # --- 8. Write output files ---
    roary_header = [
        "Gene", "Non-unique Gene name", "Annotation",
        "No. isolates", "No. sequences", "Avg sequences per isolate",
        "Genome Fragment", "Order within Fragment",
        "Accessory Fragment", "Accessory Order with Fragment", "QC",
        "Min group size nuc", "Max group size nuc", "Avg group size nuc",
    ] + isolates
    simple_header = ["Gene", "Non-unique Gene name", "Annotation"] + isolates
    rtab_header = ["Gene"] + isolates

    roary_path = output_dir / "gene_presence_absence_roary.csv"
    simple_path = output_dir / "gene_presence_absence.csv"
    rtab_path = output_dir / "gene_presence_absence.Rtab"

    with open(roary_path, "w", newline="") as roary_f, \
         open(simple_path, "w", newline="") as simple_f, \
         open(rtab_path, "w") as rtab_f:

        roary_w = csv.writer(roary_f)
        simple_w = csv.writer(simple_f)
        roary_w.writerow(roary_header)
        simple_w.writerow(simple_header)
        rtab_f.write("\t".join(rtab_header) + "\n")

        for entry_size, entry_roary, entry_simple, pres_abs, gene_name in entries:
            roary_w.writerow(entry_roary)
            simple_w.writerow(entry_simple)
            rtab_f.write(gene_name + "\t")
            rtab_f.write("\t".join("0" if e == "" else "1" for e in pres_abs) + "\n")

    logging.info(f"Wrote {len(entries)} gene clusters to {output_dir}")
    logging.info(f"  {roary_path.name} ({len(roary_header)} columns)")
    logging.info(f"  {simple_path.name} ({len(simple_header)} columns)")
    logging.info(f"  {rtab_path.name} ({len(rtab_header)} columns)")


def generate_gene_data(sqlite_path, output_dir, sqlite_cache=2000):
    """Generate Panaroo-format gene_data.csv from pangenomerge SQLite database."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
    cur = con.cursor()

    # Build member -> sample_name lookup from isolate_names
    cur.execute("SELECT graph_id, member_index, sample_name FROM isolate_names")
    member_to_sample = {}
    for graph_id, member_index, sample_name in cur:
        member_key = f"{member_index}_g{graph_id}"
        member_to_sample[member_key] = sample_name

    # Query per-gene data: join gene_annotations with node_geneids and node_sequences
    cur.execute("""
        SELECT ga.geneid, ga.annotation_id, ga.scaffold_name,
               ga.gene_name, ga.description,
               ng.member, ng.graph_id,
               ns.dna, ns.protein
        FROM gene_annotations ga
        JOIN node_geneids ng ON ng.geneid = ga.geneid
        LEFT JOIN node_sequences ns ON ns.node_id = ng.node_id
    """)

    gene_data_path = output_dir / "gene_data.csv"
    n_rows = 0
    with open(gene_data_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gff_file", "scaffold_name", "clustering_id", "annotation_id",
            "prot_sequence", "dna_sequence", "gene_name", "description"
        ])

        for row in cur:
            geneid, annotation_id, scaffold_name, gene_name, description, \
                member, graph_id, dna, protein = row

            # Derive gff_file from sample_name
            gff_file = member_to_sample.get(member, "")

            # Strip _g{N} suffix to recover original clustering_id
            m = _GN_SUFFIX_RE.match(geneid)
            clustering_id = m.group(1) if m else geneid

            writer.writerow([
                gff_file,
                scaffold_name or "",
                clustering_id,
                annotation_id or "",
                protein or "",
                dna or "",
                gene_name or "",
                description or "",
            ])
            n_rows += 1

    con.close()
    logging.info(f"Wrote {n_rows} gene entries to {gene_data_path}")


def generate_merge_figures(sqlite_path, output_dir, sqlite_cache=2000):
    """Generate per-graph merge statistics CSV and pangenome growth curve plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
    cur = con.cursor()

    # Count new nodes per graph_id by parsing the _g{N} suffix from node_id
    cur.execute("SELECT node_id FROM nodes")
    nodes_per_graph = Counter()
    for (node_id,) in cur:
        m = _GN_SUFFIX_RE.match(node_id)
        if m:
            nodes_per_graph[int(m.group(2))] += 1

    if not nodes_per_graph:
        logging.warning("No nodes found in database; skipping figures")
        con.close()
        return

    # Count samples per graph_id
    cur.execute("SELECT graph_id, COUNT(*) FROM isolate_names GROUP BY graph_id")
    samples_per_graph = dict(cur.fetchall())

    # --- Per-COG isolate prevalence for summary stats and histogram ---
    cur.execute("SELECT COUNT(*) FROM isolate_names")
    n_isolates = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM node_members GROUP BY node_id")
    member_counts = [row[0] for row in cur.fetchall()]

    con.close()

    # Compute percentage of isolates each COG is found in
    pct_isolates = [100.0 * c / n_isolates for c in member_counts]

    # Classify into Panaroo-style bins
    core = sum(1 for p in pct_isolates if p >= 99)
    soft_core = sum(1 for p in pct_isolates if 95 <= p < 99)
    shell = sum(1 for p in pct_isolates if 15 <= p < 95)
    cloud = sum(1 for p in pct_isolates if p < 15)
    total = len(pct_isolates)

    stats_path = output_dir / "summary_statistics.txt"
    with open(stats_path, "w") as f:
        f.write(f"Core genes\t(99% <= isolates <= 100%)\t{core}\n")
        f.write(f"Soft core genes\t(95% <= isolates < 99%)\t{soft_core}\n")
        f.write(f"Shell genes\t(15% <= isolates < 95%)\t{shell}\n")
        f.write(f"Cloud genes\t(0% <= isolates < 15%)\t{cloud}\n")
        f.write(f"Total genes\t(0% <= isolates <= 100%)\t{total}\n")
    logging.info(f"Wrote summary statistics to {stats_path}")

    # Plot U-shaped COG frequency histogram
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(pct_isolates, bins=100, range=(0, 100),
                 color='#2171b5', edgecolor='white', linewidth=0.3)
    ax_hist.set_xlabel('percentage of isolates')
    ax_hist.set_ylabel('number of COGs')
    ax_hist.set_title('COG Frequency Distribution')
    ax_hist.grid(True, alpha=0.3, axis='y')
    fig_hist.tight_layout()

    hist_path = output_dir / "cog_frequency_histogram.png"
    fig_hist.savefig(hist_path, dpi=150)
    plt.close(fig_hist)
    logging.info(f"Wrote COG frequency histogram to {hist_path}")

    # Build dataframe sorted by graph_id
    max_graph = max(nodes_per_graph.keys())
    rows = []
    cumulative_nodes = 0
    cumulative_samples = 0
    for gid in range(1, max_graph + 1):
        n_new_nodes = nodes_per_graph.get(gid, 0)
        n_samples = samples_per_graph.get(gid, 0)
        cumulative_nodes += n_new_nodes
        cumulative_samples += n_samples
        rows.append({
            'graph_id': gid,
            'n_new_nodes': n_new_nodes,
            'n_samples': n_samples,
            'cumulative_nodes': cumulative_nodes,
            'cumulative_samples': cumulative_samples,
        })

    df = pd.DataFrame(rows)

    # Write CSV
    csv_path = output_dir / "merge_statistics.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Wrote merge statistics to {csv_path}")

    # Plot pangenome growth curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['cumulative_samples'], df['cumulative_nodes'],
            marker='o', linewidth=1.5, markersize=4, color='#2171b5')
    ax.set_xlabel('number of isolates')
    ax.set_ylabel('clusters of orthologous genes (COGs)')
    ax.set_title('Pangenome Growth Curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "pangenome_growth_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info(f"Wrote pangenome growth curve to {plot_path}")


def cli_main():
    parser = argparse.ArgumentParser(
        description='Generate Panaroo-format output files from pangenomerge output'
    )
    parser.add_argument('--sqlite', required=True,
                        help='Path to pangenome_metadata.sqlite')
    parser.add_argument('--gml', default=None,
                        help='Path to merged_graph_N.gml (required for gene presence-absence output)')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for generated files')
    parser.add_argument('--output', choices=['all', 'presenceabsence', 'genedata', 'sequences', 'figures'],
                        default='all',
                        help='Which outputs to generate: presenceabsence (Panaroo-format gene presence-absence tables), '
                             'genedata (Panaroo-format gene_data.csv), sequences (pangenome_sequences.sqlite), '
                             'figures (merge statistics CSV and pangenome growth curve plot) (default: all)')
    parser.add_argument('--component-graphs', required=False, dest='component_graphs',
                        default=None,
                        help='Path to component graphs TSV (required for all outputs except figures)')
    parser.add_argument('--sequences-sqlite', default=None, dest='sequences_sqlite',
                        help='Path to pangenome_sequences.sqlite (default: pangenome_sequences.sqlite in same dir as --sqlite)')
    parser.add_argument('--sqlite-cache', type=int, default=2000,
                        dest='sqlite_cache',
                        help='SQLite cache size in KB (default: 2000)')
    args = parser.parse_args()

    if args.output in ('all', 'presenceabsence') and args.gml is None:
        parser.error("--gml is required when --output is 'all' or 'presenceabsence'")

    if args.output not in ('figures',) and args.component_graphs is None:
        parser.error("--component-graphs is required when --output is not 'figures'")

    # Ingest gene annotations from component graphs (deferred from merge step)
    if args.component_graphs is not None and args.output not in ('figures',):
        meta_con = sqlite_connect(database=args.sqlite, sqlite_cache=args.sqlite_cache)
        ingest_gene_annotations(meta_con, args.component_graphs)
        meta_con.close()

    if args.output in ('all', 'presenceabsence'):
        generate_gene_presence_absence(
            sqlite_path=args.sqlite,
            gml_path=args.gml,
            output_dir=args.outdir,
            sqlite_cache=args.sqlite_cache,
        )

    if args.output in ('all', 'genedata'):
        generate_gene_data(
            sqlite_path=args.sqlite,
            output_dir=args.outdir,
            sqlite_cache=args.sqlite_cache,
        )

    if args.output in ('all', 'sequences'):
        seq_db_path = args.sequences_sqlite
        if seq_db_path is None:
            seq_db_path = str(Path(args.sqlite).parent / "pangenome_sequences.sqlite")
        seq_con = sqlite_connect_sequences(database=seq_db_path, sqlite_cache=args.sqlite_cache)
        ingest_gene_sequences(seq_con, args.component_graphs)
        sqlite_create_sequence_indexes(seq_con)
        seq_con.close()
        logging.info(f"Gene sequence ingestion complete → {seq_db_path}")

    if args.output in ('all', 'figures'):
        generate_merge_figures(
            sqlite_path=args.sqlite,
            output_dir=args.outdir,
            sqlite_cache=args.sqlite_cache,
        )


if __name__ == "__main__":
    cli_main()
