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


def generate_gene_presence_absence(sqlite_path=None, gml_path=None, output_dir=None,
                                   sqlite_cache=2000, con=None):
    """Generate Panaroo-format gene presence/absence files from pangenomerge output."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Build isolate mapping and orig_ids from SQLite ---
    owns_con = con is None
    if owns_con:
        con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
        con.execute("PRAGMA query_only=ON;")
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

    if owns_con:
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


def generate_gene_data(sqlite_path=None, output_dir=None, sqlite_cache=2000, con=None):
    """Generate Panaroo-format gene_data.csv from pangenomerge SQLite database."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_con = con is None
    if owns_con:
        con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
        con.execute("PRAGMA query_only=ON;")
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

    if owns_con:
        con.close()
    logging.info(f"Wrote {n_rows} gene entries to {gene_data_path}")


def generate_summary_statistics(sqlite_path=None, output_dir=None, sqlite_cache=2000, con=None):
    """Write summary_statistics.txt: core / soft-core / shell / cloud gene counts.

    Appends a strain-level (PopPUNK cluster) section when clusters are present.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_con = con is None
    if owns_con:
        con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
        con.execute("PRAGMA query_only=ON;")
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM isolate_names")
    n_isolates = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM node_members GROUP BY node_id")
    member_counts = [row[0] for row in cur.fetchall()]

    if not member_counts:
        logging.warning("No nodes found in database; skipping summary statistics")
        if owns_con:
            con.close()
        return

    pct_isolates = [100.0 * c / n_isolates for c in member_counts]

    core = sum(1 for p in pct_isolates if p >= 99)
    soft_core = sum(1 for p in pct_isolates if 95 <= p < 99)
    shell = sum(1 for p in pct_isolates if 15 <= p < 95)
    cloud = sum(1 for p in pct_isolates if p < 15)
    total = len(pct_isolates)

    core_coarse = sum(1 for p in pct_isolates if p >= 95)
    intermediate = sum(1 for p in pct_isolates if 15 <= p < 95)
    rare = sum(1 for p in pct_isolates if p < 15)

    stats_path = output_dir / "summary_statistics.txt"
    with open(stats_path, "w") as f:
        f.write(f"Core genes\t(99% <= isolates <= 100%)\t{core}\n")
        f.write(f"Soft core genes\t(95% <= isolates < 99%)\t{soft_core}\n")
        f.write(f"Shell genes\t(15% <= isolates < 95%)\t{shell}\n")
        f.write(f"Cloud genes\t(0% <= isolates < 15%)\t{cloud}\n")
        f.write(f"Total genes\t(0% <= isolates <= 100%)\t{total}\n")
        f.write(f"\n# Coarse categories (rare / intermediate / core)\n")
        f.write(f"Core genes\t(95% <= isolates <= 100%)\t{core_coarse}\n")
        f.write(f"Intermediate genes\t(15% <= isolates < 95%)\t{intermediate}\n")
        f.write(f"Rare genes\t(0% <= isolates < 15%)\t{rare}\n")
    logging.info(f"Wrote summary statistics to {stats_path}")

    cur.execute("""
        SELECT graph_id, member_index, poppunk_cluster
        FROM isolate_names
        WHERE poppunk_cluster IS NOT NULL
    """)
    member_to_cluster = {f"{mi}_g{gid}": cl for (gid, mi, cl) in cur.fetchall()}
    n_strains = len(set(member_to_cluster.values()))

    if n_strains > 0:
        node_to_strains = defaultdict(set)
        for node_id, member in cur.execute("SELECT node_id, member FROM node_members"):
            cl = member_to_cluster.get(member)
            if cl is not None:
                node_to_strains[node_id].add(cl)

        strain_counts = [len(s) for s in node_to_strains.values()]
        pct_strains = [100.0 * c / n_strains for c in strain_counts]

        s_core = sum(1 for p in pct_strains if p >= 99)
        s_soft = sum(1 for p in pct_strains if 95 <= p < 99)
        s_shell = sum(1 for p in pct_strains if 15 <= p < 95)
        s_cloud = sum(1 for p in pct_strains if p < 15)
        s_core_coarse = sum(1 for p in pct_strains if p >= 95)
        s_intermediate = sum(1 for p in pct_strains if 15 <= p < 95)
        s_rare = sum(1 for p in pct_strains if p < 15)
        with open(stats_path, "a") as f:
            f.write(f"\n# Strain-level (PopPUNK cluster, n={n_strains})\n")
            f.write(f"Core genes\t(99% <= strains <= 100%)\t{s_core}\n")
            f.write(f"Soft core genes\t(95% <= strains < 99%)\t{s_soft}\n")
            f.write(f"Shell genes\t(15% <= strains < 95%)\t{s_shell}\n")
            f.write(f"Cloud genes\t(0% <= strains < 15%)\t{s_cloud}\n")
            f.write(f"\n# Coarse categories (rare / intermediate / core)\n")
            f.write(f"Core genes\t(95% <= strains <= 100%)\t{s_core_coarse}\n")
            f.write(f"Intermediate genes\t(15% <= strains < 95%)\t{s_intermediate}\n")
            f.write(f"Rare genes\t(0% <= strains < 15%)\t{s_rare}\n")

    if owns_con:
        con.close()


def generate_merge_figures(sqlite_path=None, output_dir=None, sqlite_cache=2000, con=None):
    """Generate per-graph merge statistics CSV and pangenome growth curve plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_con = con is None
    if owns_con:
        con = sqlite_connect(database=sqlite_path, sqlite_cache=sqlite_cache)
        con.execute("PRAGMA query_only=ON;")
    cur = con.cursor()

    # Count new nodes per graph_id using SQL-side suffix extraction
    cur.execute("""
        SELECT CAST(SUBSTR(node_id, INSTR(node_id, '_g') + 2) AS INTEGER) AS graph_id,
               COUNT(*)
        FROM nodes
        WHERE INSTR(node_id, '_g') > 0
        GROUP BY 1
    """)
    nodes_per_graph = dict(cur.fetchall())

    if not nodes_per_graph:
        logging.warning("No nodes found in database; skipping figures")
        if owns_con:
            con.close()
        return

    # Count samples per graph_id
    cur.execute("SELECT graph_id, COUNT(*) FROM isolate_names GROUP BY graph_id")
    samples_per_graph = dict(cur.fetchall())

    # --- Per-COG isolate prevalence for histogram ---
    cur.execute("SELECT COUNT(*) FROM isolate_names")
    n_isolates = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM node_members GROUP BY node_id")
    member_counts = [row[0] for row in cur.fetchall()]

    # Compute percentage of isolates each COG is found in
    pct_isolates = [100.0 * c / n_isolates for c in member_counts]

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

    fig_ric, ax_ric = plt.subplots(figsize=(8, 5))
    ax_ric.hist(pct_isolates, bins=[0, 15, 95, 100],
                color='#2171b5', edgecolor='white', linewidth=0.3)
    ax_ric.set_xlabel('percentage of isolates')
    ax_ric.set_ylabel('number of COGs')
    ax_ric.set_title('COG Frequency Distribution (rare / intermediate / core)')
    ax_ric.grid(True, alpha=0.3, axis='y')
    fig_ric.tight_layout()
    ric_path = output_dir / "cog_frequency_ric_histogram.png"
    fig_ric.savefig(ric_path, dpi=150)
    plt.close(fig_ric)
    logging.info(f"Wrote rare/intermediate/core COG frequency histogram to {ric_path}")

    # --- Per-COG strain (PopPUNK cluster) prevalence ---
    cur.execute("""
        SELECT graph_id, member_index, poppunk_cluster
        FROM isolate_names
        WHERE poppunk_cluster IS NOT NULL
    """)
    member_to_cluster = {f"{mi}_g{gid}": cl for (gid, mi, cl) in cur.fetchall()}
    n_strains = len(set(member_to_cluster.values()))

    if n_strains == 0:
        logging.info("No PopPUNK clusters found in isolate_names; skipping strain-level histograms")
    else:
        node_to_strains = defaultdict(set)
        for node_id, member in cur.execute("SELECT node_id, member FROM node_members"):
            cl = member_to_cluster.get(member)
            if cl is not None:
                node_to_strains[node_id].add(cl)

        strain_counts = [len(s) for s in node_to_strains.values()]
        pct_strains = [100.0 * c / n_strains for c in strain_counts]

        fig_sc, ax_sc = plt.subplots(figsize=(8, 5))
        ax_sc.hist(strain_counts, bins=range(1, n_strains + 2),
                   color='#2171b5', edgecolor='white', linewidth=0.3, align='left')
        ax_sc.set_xlabel('number of strains')
        ax_sc.set_ylabel('number of COGs')
        ax_sc.set_title(f'COG Strain-Count Distribution (n_strains={n_strains})')
        ax_sc.grid(True, alpha=0.3, axis='y')
        fig_sc.tight_layout()
        sc_path = output_dir / "cog_strain_count_histogram.png"
        fig_sc.savefig(sc_path, dpi=150)
        plt.close(fig_sc)
        logging.info(f"Wrote COG strain-count histogram to {sc_path}")

        fig_sp, ax_sp = plt.subplots(figsize=(8, 5))
        ax_sp.hist(pct_strains, bins=100, range=(0, 100),
                   color='#2171b5', edgecolor='white', linewidth=0.3)
        ax_sp.set_xlabel('percentage of strains')
        ax_sp.set_ylabel('number of COGs')
        ax_sp.set_title('COG Strain Frequency Distribution')
        ax_sp.grid(True, alpha=0.3, axis='y')
        fig_sp.tight_layout()
        sp_path = output_dir / "cog_strain_frequency_histogram.png"
        fig_sp.savefig(sp_path, dpi=150)
        plt.close(fig_sp)
        logging.info(f"Wrote COG strain frequency histogram to {sp_path}")

        fig_ric_s, ax_ric_s = plt.subplots(figsize=(8, 5))
        ax_ric_s.hist(pct_strains, bins=[0, 15, 95, 100],
                      color='#2171b5', edgecolor='white', linewidth=0.3)
        ax_ric_s.set_xlabel('percentage of strains')
        ax_ric_s.set_ylabel('number of COGs')
        ax_ric_s.set_title('COG Strain Frequency Distribution (rare / intermediate / core)')
        ax_ric_s.grid(True, alpha=0.3, axis='y')
        fig_ric_s.tight_layout()
        ric_s_path = output_dir / "cog_strain_frequency_ric_histogram.png"
        fig_ric_s.savefig(ric_s_path, dpi=150)
        plt.close(fig_ric_s)
        logging.info(f"Wrote rare/intermediate/core COG strain frequency histogram to {ric_s_path}")

    # --- Multi-copy genes diagnostic: n_geneids vs. n_members per COG ---
    cur.execute("SELECT node_id, COUNT(*) FROM node_geneids GROUP BY node_id")
    geneids_per_node = dict(cur.fetchall())
    cur.execute("SELECT node_id, COUNT(*) FROM node_members GROUP BY node_id")
    members_per_node = dict(cur.fetchall())

    mc_x = []
    mc_y = []
    for node_id, n_g in geneids_per_node.items():
        n_m = members_per_node.get(node_id, 0)
        if n_m > 0:
            mc_x.append(n_m)
            mc_y.append(n_g)

    if mc_x:
        fig_mc, ax_mc = plt.subplots(figsize=(8, 5))
        ax_mc.scatter(mc_x, mc_y, alpha=0.3, color='#2171b5',
                      s=12, edgecolor='none')
        max_v = max(max(mc_x), max(mc_y))
        ax_mc.plot([0, max_v], [0, max_v], color='red', linewidth=1)
        ax_mc.set_xlim(0, max_v)
        ax_mc.set_ylim(0, max_v)
        ax_mc.set_xlabel('number of members (COG size)')
        ax_mc.set_ylabel('number of geneids')
        ax_mc.set_title('multi-copy genes')
        ax_mc.grid(True, alpha=0.3)
        fig_mc.tight_layout()
        mc_path = output_dir / "multi_copy_genes.png"
        fig_mc.savefig(mc_path, dpi=150)
        plt.close(fig_mc)
        logging.info(f"Wrote multi-copy genes plot to {mc_path}")

    if owns_con:
        con.close()

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
    parser.add_argument('--pangenomerge-results', default=None, dest='pangenomerge_results',
                        help='Path to a pangenomerge output directory containing the --sqlite, '
                             '--gml, and --sequences-sqlite inputs. Mutually exclusive with those '
                             'flags. When set, --outdir defaults to <pangenomerge-results>/postprocessing.')
    parser.add_argument('--sqlite', default=None,
                        help='Path to pangenome_metadata.sqlite')
    parser.add_argument('--gml', default=None,
                        help='Path to final_graph.gml (required for gene presence-absence output)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for generated files '
                             '(default: <pangenomerge-results>/postprocessing when --pangenomerge-results is set)')
    parser.add_argument('--output', choices=['all', 'presenceabsence', 'genedata', 'sequences', 'figures'],
                        default='all',
                        help='Which outputs to generate: presenceabsence (Panaroo-format gene presence-absence tables), '
                             'genedata (Panaroo-format gene_data.csv), sequences (pangenome_sequences.sqlite), '
                             'figures (merge statistics CSV and pangenome growth curve plot). '
                             "'all' generates everything except gene_data.csv; pass --gene-data to include it (default: all)")
    parser.add_argument('--gene-data', action='store_true', dest='gene_data',
                        help='Also generate gene_data.csv alongside the outputs selected by --output.')
    parser.add_argument('--component-graphs', required=False, dest='component_graphs',
                        default=None,
                        help='Path to component graphs TSV (required for all outputs except figures)')
    parser.add_argument('--sequences-sqlite', default=None, dest='sequences_sqlite',
                        help='Path to pangenome_sequences.sqlite (default: pangenome_sequences.sqlite in same dir as --sqlite)')
    parser.add_argument('--sqlite-cache', type=int, default=2000,
                        dest='sqlite_cache',
                        help='SQLite cache size in KB (default: 2000)')
    args = parser.parse_args()

    if args.pangenomerge_results is not None:
        conflicting = [name for name, val in
                       (('--sqlite', args.sqlite),
                        ('--gml', args.gml),
                        ('--sequences-sqlite', args.sequences_sqlite))
                       if val is not None]
        if conflicting:
            parser.error(
                "--pangenomerge-results is mutually exclusive with "
                + " / ".join(conflicting))
        results_dir = Path(args.pangenomerge_results)
        args.sqlite = str(results_dir / "pangenome_metadata.sqlite")
        args.gml = str(results_dir / "final_graph.gml")
        if args.outdir is None:
            args.outdir = str(results_dir / "postprocessing")
    elif args.sqlite is None:
        parser.error("must specify either --pangenomerge-results or --sqlite")

    if args.outdir is None:
        parser.error("--outdir is required when --pangenomerge-results is not set")

    if args.output in ('all', 'presenceabsence') and args.gml is None:
        parser.error("--gml is required when --output is 'all' or 'presenceabsence'")

    needs_component_graphs = args.output != 'figures' or args.gene_data
    if needs_component_graphs and args.component_graphs is None:
        parser.error("--component-graphs is required unless --output is 'figures' without --gene-data")

    # Ingest gene annotations from component graphs (deferred from merge step)
    if args.component_graphs is not None and (args.output != 'figures' or args.gene_data):
        meta_con = sqlite_connect(database=args.sqlite, sqlite_cache=args.sqlite_cache)
        ingest_gene_annotations(meta_con, args.component_graphs)
        meta_con.close()

    # Share a single read-only connection across output functions
    needs_meta_con = (args.output in ('all', 'presenceabsence', 'genedata', 'figures')
                     or args.gene_data)
    if needs_meta_con:
        meta_con = sqlite_connect(database=args.sqlite, sqlite_cache=args.sqlite_cache)
        meta_con.execute("PRAGMA query_only=ON;")
    else:
        meta_con = None

    if args.output in ('all', 'presenceabsence'):
        generate_gene_presence_absence(
            gml_path=args.gml,
            output_dir=args.outdir,
            con=meta_con,
        )

    if args.gene_data or args.output == 'genedata':
        generate_gene_data(
            output_dir=args.outdir,
            con=meta_con,
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
            output_dir=args.outdir,
            con=meta_con,
        )

    if meta_con is not None:
        meta_con.close()


if __name__ == "__main__":
    cli_main()
