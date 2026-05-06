import sqlite3
import re
import logging
import zlib
import hashlib
from pathlib import Path
from collections import Counter

_GN_SUFFIX_RE = re.compile(r'^(.+)_g(\d+)$')
_SPLIT_CLUSTER_RE = re.compile(r'^(\d+)[a-z]$')

def canon_uv(u, v):
    u, v = str(u), str(v)
    return (u, v) if u <= v else (v, u)

def _sqlite_open(database: str, sqlite_cache: int) -> sqlite3.Connection:
    """Open an SQLite database with tuned PRAGMAs for bulk-insert workloads."""
    Path(database).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(database)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=OFF;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA busy_timeout=5000;")
    con.execute("PRAGMA wal_autocheckpoint=5000;")
    con.execute(f"PRAGMA cache_size=-{sqlite_cache};")
    return con

def sqlite_connect(database: str, sqlite_cache: int) -> sqlite3.Connection:
    """Connect to (or create) the metadata SQLite database."""
    return _sqlite_open(database, sqlite_cache)

def sqlite_connect_sequences(database: str, sqlite_cache: int) -> sqlite3.Connection:
    """Connect to (or create) the dedicated sequence SQLite database."""
    con = _sqlite_open(database, sqlite_cache)
    con.execute("PRAGMA page_size=16384;")  # larger pages for BLOB-heavy tables
    con.executescript("""
    CREATE TABLE IF NOT EXISTS unique_sequences (
        seq_hash TEXT NOT NULL,
        seq_type TEXT NOT NULL,
        seq_data BLOB NOT NULL,
        seq_len  INTEGER NOT NULL,
        PRIMARY KEY (seq_hash, seq_type)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS gene_sequences (
        geneid   TEXT PRIMARY KEY,
        nt_hash  TEXT,
        aa_hash  TEXT
    ) WITHOUT ROWID;
    """)
    con.commit()
    return con


def sqlite_create_sequence_indexes(con: sqlite3.Connection):
    """Create indexes on the sequence database."""
    con.executescript("""
    CREATE INDEX IF NOT EXISTS idx_gene_sequences_nt ON gene_sequences(nt_hash);
    CREATE INDEX IF NOT EXISTS idx_gene_sequences_aa ON gene_sequences(aa_hash);
    """)
    con.commit()


def sqlite_init_schema(con: sqlite3.Connection):
    # cumulative tables keyed by node_id / (u,v)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS nodes (
        node_id TEXT PRIMARY KEY,
        name TEXT,
        size INTEGER,
        degrees INTEGER,
        genomeIDs TEXT,
        maxLenId TEXT,
        hasEnd INTEGER,
        annotation TEXT,
        description TEXT,
        paralog INTEGER,
        mergedDNA TEXT,
        last_iteration INTEGER
    );

    CREATE TABLE IF NOT EXISTS node_members (
        node_id TEXT,
        member TEXT,
        PRIMARY KEY (node_id, member)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_seqids (
        node_id TEXT,
        seqid TEXT,
        member TEXT,
        graph_id INTEGER,
        PRIMARY KEY (node_id, seqid)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_geneids (
        node_id TEXT,
        geneid TEXT,
        member TEXT,
        graph_id INTEGER,
        PRIMARY KEY (node_id, geneid)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_centroids (
        node_id TEXT,
        centroid TEXT,
        PRIMARY KEY (node_id, centroid)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_lengths (
        node_id TEXT,
        length INTEGER,
        count INTEGER,
        PRIMARY KEY (node_id, length)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_longCentroidID (
        node_id TEXT,
        tag TEXT,
        PRIMARY KEY (node_id, tag)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_sequences (
        node_id TEXT PRIMARY KEY,
        dna TEXT,
        protein TEXT
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS edges (
        u TEXT,
        v TEXT,
        size INTEGER,
        genomeIDs TEXT,
        last_iteration INTEGER,
        PRIMARY KEY (u, v)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS edge_members (
        u TEXT,
        v TEXT,
        member TEXT,
        PRIMARY KEY (u, v, member)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS isolate_names (
        graph_id INTEGER NOT NULL,
        member_index INTEGER NOT NULL,
        sample_name TEXT NOT NULL,
        poppunk_cluster TEXT,
        PRIMARY KEY (graph_id, member_index)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS gene_annotations (
        geneid TEXT PRIMARY KEY,
        annotation_id TEXT,
        scaffold_name TEXT,
        gene_name TEXT,
        description TEXT
    ) WITHOUT ROWID;

    """)
    con.commit()

def sqlite_create_indexes(con: sqlite3.Connection):
    # cumulative tables keyed by node_id / (u,v)
    con.executescript("""
    CREATE INDEX IF NOT EXISTS idx_node_members_member ON node_members(member);
    CREATE INDEX IF NOT EXISTS idx_node_seqids_seqid ON node_seqids(seqid);
    CREATE INDEX IF NOT EXISTS idx_node_geneids_geneid ON node_geneids(geneid);
    CREATE INDEX IF NOT EXISTS idx_isolate_names_sample ON isolate_names(sample_name);
    CREATE INDEX IF NOT EXISTS idx_isolate_names_cluster ON isolate_names(poppunk_cluster);
    CREATE INDEX IF NOT EXISTS idx_gene_annotations_annot ON gene_annotations(annotation_id);
    """)
    con.commit()

def add_isolate_names_to_sqlite(con: sqlite3.Connection, graph_id: int, isolate_names: list):
    cur = con.cursor()
    rows = [(int(graph_id), int(i), str(name)) for i, name in enumerate(isolate_names)]
    cur.executemany(
        "INSERT OR IGNORE INTO isolate_names(graph_id, member_index, sample_name) VALUES (?,?,?)",
        rows
    )
    con.commit()

def add_clusters_to_sqlite(con: sqlite3.Connection, clusters_path: str):
    """Read a PopPUNK cluster CSV (header: Taxon,Cluster) and populate
    isolate_names.poppunk_cluster by sample_name. Warns about isolates
    in isolate_names with no matching Taxon."""
    path = Path(clusters_path)
    if not path.exists():
        logging.warning(f"clusters file not found: {clusters_path}; skipping")
        return

    rows = []  # (cluster, sample)
    with open(path, 'r') as f:
        header = next(f, None)
        if header is None or 'Taxon' not in header:
            logging.warning(f"clusters file {clusters_path} missing 'Taxon,Cluster' header; skipping")
            return
        for line in f:
            parts = line.rstrip('\n').split(',')
            if len(parts) < 2:
                continue
            sample, cluster = parts[0].strip(), parts[1].strip()
            if sample and cluster:
                # Collapse size-rebalanced split labels (e.g. "10a","10b" -> "10")
                # back to the original PopPUNK cluster ID. The R splitter in
                # adjust_cluster_sizes.R uses paste0(cl, letters[pieces]),
                # i.e. always <digits><single lowercase letter>.
                m = _SPLIT_CLUSTER_RE.match(cluster)
                if m:
                    cluster = m.group(1)
                rows.append((cluster, sample))

    cur = con.cursor()
    cur.executemany(
        "UPDATE isolate_names SET poppunk_cluster = ? WHERE sample_name = ?",
        rows,
    )
    con.commit()

    missing = [s for (s,) in cur.execute(
        "SELECT sample_name FROM isolate_names WHERE poppunk_cluster IS NULL"
    )]
    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing)-10} more)" if len(missing) > 10 else ""
        logging.warning(
            f"{len(missing)} isolate(s) have no PopPUNK cluster in {clusters_path}: {preview}{more}"
        )
    matched = cur.execute(
        "SELECT COUNT(*) FROM isolate_names WHERE poppunk_cluster IS NOT NULL"
    ).fetchone()[0]
    logging.info(f"Stored PopPUNK clusters for {matched} isolate row(s) from {len(rows)} cluster-file entries")

def add_gene_annotations_to_sqlite(con: sqlite3.Connection, graph_id: int, graph_dir: str):
    """Read gene_data.csv from a component graph directory and store
    annotation_id, scaffold_name, gene_name, and description per gene in SQLite."""
    gene_data_path = Path(graph_dir) / "gene_data.csv"
    if not gene_data_path.exists():
        logging.warning(f"gene_data.csv not found in {graph_dir}, skipping gene annotations")
        return
    BATCH = 50000
    rows = []
    total = 0
    with open(gene_data_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            # columns: gff_file, scaffold_name, clustering_id, annotation_id,
            #          prot_sequence, dna_sequence, gene_name, description
            parts = line.rstrip('\n').split(",")
            if len(parts) < 4:
                continue
            scaffold_name = parts[1]
            clustering_id = parts[2]
            annotation_id = parts[3]
            gene_name = parts[6] if len(parts) > 6 else ""
            description = ",".join(parts[7:]) if len(parts) > 7 else ""
            suffixed_id = f"{clustering_id}_g{graph_id}"
            rows.append((suffixed_id, annotation_id, scaffold_name, gene_name, description))
            if len(rows) >= BATCH:
                con.executemany(
                    "INSERT OR IGNORE INTO gene_annotations(geneid, annotation_id, scaffold_name, gene_name, description) VALUES (?,?,?,?,?)",
                    rows
                )
                total += len(rows)
                rows = []
    if rows:
        con.executemany(
            "INSERT OR IGNORE INTO gene_annotations(geneid, annotation_id, scaffold_name, gene_name, description) VALUES (?,?,?,?,?)",
            rows
        )
        total += len(rows)
    con.commit()
    logging.info(f"Stored {total} gene annotations from graph {graph_id}")


def _hash_seq(seq: str) -> str:
    """Return a 128-bit MD5 hex digest for a sequence string."""
    return hashlib.md5(seq.encode()).hexdigest()


def ingest_gene_sequences(seq_con: sqlite3.Connection, component_graphs_tsv: str,
                          batch_size: int = 50000):
    """Read per-gene DNA and protein sequences from all component graph
    gene_data.csv files and store them in the sequence SQLite database
    with deduplication.

    Parameters
    ----------
    seq_con : sqlite3.Connection
        Connection to the dedicated sequence database (pangenome_sequences.sqlite).
    component_graphs_tsv : str
        Path to component graphs TSV listing graph directories.
    batch_size : int
        Number of gene rows to buffer before flushing to SQLite.
    """
    import pandas as pd

    graph_files = pd.read_csv(component_graphs_tsv, sep='\t', header=None)
    n_graphs = len(graph_files)

    cur = seq_con.cursor()
    cur.execute("BEGIN;")

    total_genes = 0
    unique_buf = []   # (seq_hash, seq_type, compressed_blob, seq_len)
    gene_buf = []     # (geneid, nt_hash, aa_hash)
    seen_hashes = set()  # skip redundant zlib.compress for duplicate sequences

    def _flush():
        nonlocal unique_buf, gene_buf
        if unique_buf:
            cur.executemany(
                "INSERT OR IGNORE INTO unique_sequences(seq_hash, seq_type, seq_data, seq_len) "
                "VALUES (?,?,?,?)",
                unique_buf,
            )
        if gene_buf:
            cur.executemany(
                "INSERT OR IGNORE INTO gene_sequences(geneid, nt_hash, aa_hash) "
                "VALUES (?,?,?)",
                gene_buf,
            )
        unique_buf = []
        gene_buf = []

    for idx in range(n_graphs):
        graph_id = idx + 1
        graph_dir = str(graph_files.iloc[idx][0])
        gene_data_path = Path(graph_dir) / "gene_data.csv"

        if not gene_data_path.exists():
            logging.warning(f"gene_data.csv not found in {graph_dir}, skipping sequences")
            continue

        graph_genes = 0
        with open(gene_data_path, 'r') as f:
            next(f)  # skip header
            for line in f:
                # cols: gff_file(0), scaffold_name(1), clustering_id(2),
                #       annotation_id(3), prot_sequence(4), dna_sequence(5), ...
                parts = line.rstrip('\n').split(",")
                if len(parts) < 6:
                    continue

                clustering_id = parts[2]
                prot_seq = parts[4]
                dna_seq = parts[5]
                geneid = f"{clustering_id}_g{graph_id}"

                nt_hash = None
                aa_hash = None

                if dna_seq:
                    nt_hash = _hash_seq(dna_seq)
                    if nt_hash not in seen_hashes:
                        seen_hashes.add(nt_hash)
                        unique_buf.append((
                            nt_hash, 'nt',
                            zlib.compress(dna_seq.encode()),
                            len(dna_seq),
                        ))

                if prot_seq:
                    aa_hash = _hash_seq(prot_seq)
                    if aa_hash not in seen_hashes:
                        seen_hashes.add(aa_hash)
                        unique_buf.append((
                            aa_hash, 'aa',
                            zlib.compress(prot_seq.encode()),
                            len(prot_seq),
                        ))

                gene_buf.append((geneid, nt_hash, aa_hash))
                graph_genes += 1

                if len(gene_buf) >= batch_size:
                    _flush()

        total_genes += graph_genes
        logging.info(f"Read {graph_genes} gene sequences from graph {graph_id}")

    _flush()
    seq_con.execute("COMMIT;")
    logging.info(f"Ingested {total_genes} total gene sequences into SQLite")

    # report deduplication stats
    n_unique = cur.execute("SELECT COUNT(*) FROM unique_sequences").fetchone()[0]
    n_genes = cur.execute("SELECT COUNT(*) FROM gene_sequences").fetchone()[0]
    logging.info(f"  {n_genes} gene entries, {n_unique} unique sequences stored")


def _norm_text_or_none(x):
    # treat empty string/None as None
    if x is None:
        return None
    s = str(x)
    return s if s.strip() != "" else None

def _is_placeholder_seq(dna, protein):
    # identify whether real or placeholder sequence to overwriting real sequences
    if dna is None and protein is None:
        return True
    dna_txt = ";".join(dna) if isinstance(dna, list) else (dna or "")
    prot_txt = ";".join(protein) if isinstance(protein, list) else (protein or "")
    return (dna_txt.strip() == "" and prot_txt.strip() == "")

def add_metadata_to_sqlite(G, iteration: int, con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")

    # ---- UPSERT nodes ----
    node_rows = []
    seq_rows = []
    members_rows = []
    seqid_rows = []
    geneid_rows = []
    centroid_rows = []
    length_rows = []
    longcid_rows = []

    for node_id, data in G.nodes(data=True):
        node_id = str(node_id)

        # check for payload (any form of non-placeholder metadata)
        members = data.get("members") or []
        seqids  = data.get("seqIDs") or []
        geneIDs = (data.get("geneIDs") or "").strip()
        centroids = data.get("centroid") or []
        lengths = data.get("lengths") or []
        longcid = data.get("longCentroidID") or []

        dna = data.get("dna")
        protein = data.get("protein")
        has_seq = not _is_placeholder_seq(dna, protein)

        has_payload = (
            bool(members) or bool(seqids) or bool(geneIDs) or bool(centroids) or
            bool(lengths) or bool(longcid) or has_seq or
            bool(_norm_text_or_none(data.get("annotation"))) or
            bool(_norm_text_or_none(data.get("description"))) or
            bool(_norm_text_or_none(data.get("genomeIDs"))) or
            bool(_norm_text_or_none(data.get("maxLenId"))) or
            bool(_norm_text_or_none(data.get("mergedDNA"))) or
            data.get("hasEnd") not in (None, 0) or
            data.get("paralog") not in (None, 0)
        )

        # if no non-placeholder metadata, skip node
        if not has_payload:
            continue 

        # change any placeholder metadata to NULL
        name = _norm_text_or_none(data.get("name"))
        size = data.get("size")
        degrees = data.get("degrees")
        genomeIDs = _norm_text_or_none(data.get("genomeIDs"))
        maxLenId = _norm_text_or_none(data.get("maxLenId"))
        hasEnd = data.get("hasEnd")
        annotation = _norm_text_or_none(data.get("annotation"))
        description = _norm_text_or_none(data.get("description"))
        paralog = data.get("paralog")
        mergedDNA = _norm_text_or_none(data.get("mergedDNA"))
        
        # only write size if members exist (to prevent adding placeholder size=1)
        members = data.get("members") or []
        if not members:
            size_val = None
        else:
            size_val = int(size) if size is not None else None

        degrees_val = int(degrees) if degrees is not None else None

        node_rows.append((
            node_id, name, size_val, degrees_val, genomeIDs,
            maxLenId, int(hasEnd) if hasEnd is not None else None,
            annotation, description,
            int(paralog) if paralog is not None else None,
            mergedDNA,
            int(iteration)
        ))

        for m in members:
            m = str(m).strip()
            if m:
                members_rows.append((node_id, m))

        for s in (data.get("seqIDs") or []):
            s = str(s).strip()
            if s:
                m = _GN_SUFFIX_RE.match(s)
                if m:
                    prefix, gnum = m.group(1), int(m.group(2))
                    # member index is the first underscore-separated component
                    member_idx = prefix.split("_", 1)[0]
                    member_key = f"{member_idx}_g{gnum}"
                else:
                    member_key = None
                    gnum = None
                seqid_rows.append((node_id, s, member_key, gnum))

        geneIDs = data.get("geneIDs") or ""
        if str(geneIDs).strip():
            for gid in str(geneIDs).split(";"):
                gid = gid.strip()
                if gid:
                    m = _GN_SUFFIX_RE.match(gid)
                    if m:
                        prefix, gnum = m.group(1), int(m.group(2))
                        member_idx = prefix.split("_", 1)[0]
                        member_key = f"{member_idx}_g{gnum}"
                    else:
                        member_key = None
                        gnum = None
                    geneid_rows.append((node_id, gid, member_key, gnum))

        centroids = data.get("centroid") or []
        if isinstance(centroids, str):
            centroids = [centroids]
        for c in centroids:
            c = str(c).strip()
            if c:
                centroid_rows.append((node_id, c))

        lengths = data.get("lengths") or []
        ctr = Counter(int(L) for L in lengths if L is not None)
        for L, c in ctr.items():
            length_rows.append((node_id, int(L), int(c)))

        for t in (data.get("longCentroidID") or []):
            t = str(t).strip()
            if t:
                longcid_rows.append((node_id, t))

        dna = data.get("dna")
        protein = data.get("protein")
        if not _is_placeholder_seq(dna, protein):
            dna_txt = ";".join(dna) if isinstance(dna, list) else dna
            prot_txt = ";".join(protein) if isinstance(protein, list) else protein
            dna_txt = _norm_text_or_none(dna_txt)
            prot_txt = _norm_text_or_none(prot_txt)
            if dna_txt is not None or prot_txt is not None:
                seq_rows.append((node_id, dna_txt, prot_txt))

    # keep old value when excluded is NULL (placeholder)
    cur.executemany("""
        INSERT INTO nodes(node_id,name,size,degrees,genomeIDs,maxLenId,hasEnd,
                        annotation,description,paralog,mergedDNA,last_iteration)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(node_id) DO UPDATE SET
            name         = COALESCE(excluded.name, nodes.name),
            size         = COALESCE(excluded.size, nodes.size),
            degrees      = COALESCE(excluded.degrees, nodes.degrees),
            genomeIDs    = COALESCE(excluded.genomeIDs, nodes.genomeIDs),
            maxLenId     = COALESCE(excluded.maxLenId, nodes.maxLenId),
            hasEnd       = COALESCE(excluded.hasEnd, nodes.hasEnd),
            annotation   = COALESCE(excluded.annotation, nodes.annotation),
            description  = COALESCE(excluded.description, nodes.description),
            paralog      = COALESCE(excluded.paralog, nodes.paralog),
            mergedDNA    = COALESCE(excluded.mergedDNA, nodes.mergedDNA),
            last_iteration = MAX(nodes.last_iteration, excluded.last_iteration)
        WHERE
            excluded.size IS NOT NULL OR
            excluded.degrees IS NOT NULL OR
            excluded.genomeIDs IS NOT NULL OR
            excluded.maxLenId IS NOT NULL OR
            excluded.hasEnd IS NOT NULL OR
            excluded.annotation IS NOT NULL OR
            excluded.description IS NOT NULL OR
            excluded.paralog IS NOT NULL OR
            excluded.mergedDNA IS NOT NULL;
    """, node_rows)

    cur.executemany("INSERT INTO node_members(node_id,member) VALUES (?,?)", members_rows)
    cur.executemany("INSERT INTO node_seqids(node_id,seqid,member,graph_id) VALUES (?,?,?,?)", seqid_rows)
    cur.executemany("INSERT INTO node_geneids(node_id,geneid,member,graph_id) VALUES (?,?,?,?)", geneid_rows)
    cur.executemany("INSERT INTO node_centroids(node_id,centroid) VALUES (?,?)", centroid_rows)
    cur.executemany("INSERT INTO node_longCentroidID(node_id,tag) VALUES (?,?)", longcid_rows)

    # update lengths (increment counts if the length is already present)
    cur.executemany("""
        INSERT INTO node_lengths(node_id,length,count) VALUES (?,?,?)
        ON CONFLICT(node_id,length) DO UPDATE SET
            count = node_lengths.count + excluded.count
    """, length_rows)

    # update sequences
    cur.executemany("""
        INSERT INTO node_sequences(node_id,dna,protein) VALUES (?,?,?)
        ON CONFLICT(node_id) DO UPDATE SET
            dna     = COALESCE(excluded.dna, node_sequences.dna),
            protein = COALESCE(excluded.protein, node_sequences.protein)
    """, seq_rows)

    ### ---- UPSERT edges ----

    edge_rows = []
    edge_member_rows = []

    for u, v, edata in G.edges(data=True):
        u, v = canon_uv(u, v)

        size = edata.get("size")
        genomeIDs = _norm_text_or_none(edata.get("genomeIDs"))

        # treat size as placeholder if members is empty
        emembers = edata.get("members") or []
        if not emembers:
            size_val = None
        else:
            size_val = int(size) if size is not None else None
        
        emembers = edata.get("members") or []
        genomeIDs = _norm_text_or_none(edata.get("genomeIDs"))

        # skip placeholder edges
        if not emembers and genomeIDs is None:
            continue

        edge_rows.append((u, v, size_val, genomeIDs, int(iteration)))

        for m in emembers:
            m = str(m).strip()
            if m:
                edge_member_rows.append((u, v, m))

    cur.executemany("""
        INSERT INTO edges(u,v,size,genomeIDs,last_iteration) VALUES (?,?,?,?,?)
        ON CONFLICT(u,v) DO UPDATE SET
            size = COALESCE(excluded.size, edges.size),
            genomeIDs = COALESCE(excluded.genomeIDs, edges.genomeIDs),
            last_iteration = MAX(edges.last_iteration, excluded.last_iteration)
        WHERE
            excluded.size IS NOT NULL OR
            excluded.genomeIDs IS NOT NULL;
    """, edge_rows)

    cur.executemany("INSERT OR IGNORE INTO edge_members(u,v,member) VALUES (?,?,?)", edge_member_rows)

    cur.execute("COMMIT;")


def load_metadata_from_sqlite(G, con):
    """Populate node/edge metadata on G from the cumulative SQLite tables.

    Used when --metadata-in-graph is set to rebuild the final graph's
    metadata after iteration-level stripping. Only nodes/edges already
    present in G are updated.
    """
    cur = con.cursor()

    node_ids = set(G.nodes())
    edge_ids = set((u, v) for u, v in G.edges())

    for nid in node_ids:
        d = G.nodes[nid]
        d["members"] = []
        d["seqIDs"] = []
        d["geneIDs"] = ''
        d["centroid"] = []
        d["lengths"] = []
        d["longCentroidID"] = []
        d["dna"] = [""]
        d["protein"] = [""]

    for row in cur.execute(
        "SELECT node_id,name,size,degrees,genomeIDs,maxLenId,hasEnd,"
        "annotation,description,paralog,mergedDNA FROM nodes"
    ):
        nid = row[0]
        if nid not in node_ids:
            continue
        d = G.nodes[nid]
        if row[1]  is not None: d["name"]        = row[1]
        if row[2]  is not None: d["size"]        = row[2]
        if row[3]  is not None: d["degrees"]     = row[3]
        if row[4]  is not None: d["genomeIDs"]   = row[4]
        if row[5]  is not None: d["maxLenId"]    = row[5]
        if row[6]  is not None: d["hasEnd"]      = row[6]
        if row[7]  is not None: d["annotation"]  = row[7]
        if row[8]  is not None: d["description"] = row[8]
        if row[9]  is not None: d["paralog"]     = row[9]
        if row[10] is not None: d["mergedDNA"]   = row[10]

    tmp = {}
    for nid, m in cur.execute("SELECT node_id,member FROM node_members"):
        if nid in node_ids:
            tmp.setdefault(nid, []).append(m)
    for nid, vals in tmp.items():
        G.nodes[nid]["members"] = vals

    tmp = {}
    for nid, s in cur.execute("SELECT node_id,seqid FROM node_seqids"):
        if nid in node_ids:
            tmp.setdefault(nid, []).append(s)
    for nid, vals in tmp.items():
        G.nodes[nid]["seqIDs"] = vals

    tmp = {}
    for nid, g in cur.execute("SELECT node_id,geneid FROM node_geneids"):
        if nid in node_ids:
            tmp.setdefault(nid, []).append(g)
    for nid, vals in tmp.items():
        G.nodes[nid]["geneIDs"] = ";".join(vals)

    tmp = {}
    for nid, c in cur.execute("SELECT node_id,centroid FROM node_centroids"):
        if nid in node_ids:
            tmp.setdefault(nid, []).append(c)
    for nid, vals in tmp.items():
        G.nodes[nid]["centroid"] = vals

    tmp = {}
    for nid, L, c in cur.execute("SELECT node_id,length,count FROM node_lengths"):
        if nid in node_ids:
            tmp.setdefault(nid, []).extend([int(L)] * int(c))
    for nid, vals in tmp.items():
        G.nodes[nid]["lengths"] = vals

    tmp = {}
    for nid, t in cur.execute("SELECT node_id,tag FROM node_longCentroidID"):
        if nid in node_ids:
            tmp.setdefault(nid, []).append(t)
    for nid, vals in tmp.items():
        G.nodes[nid]["longCentroidID"] = vals

    for nid, dna, prot in cur.execute(
        "SELECT node_id,dna,protein FROM node_sequences"
    ):
        if nid not in node_ids:
            continue
        d = G.nodes[nid]
        if dna  is not None: d["dna"]     = dna.split(";")
        if prot is not None: d["protein"] = prot.split(";")

    for u, v, size, gIDs in cur.execute(
        "SELECT u,v,size,genomeIDs FROM edges"
    ):
        if (u, v) not in edge_ids:
            continue
        e = G[u][v]
        if size is not None: e["size"]      = size
        if gIDs is not None: e["genomeIDs"] = gIDs

    tmp = {}
    for u, v, m in cur.execute("SELECT u,v,member FROM edge_members"):
        if (u, v) in edge_ids:
            tmp.setdefault((u, v), []).append(m)
    for (u, v), vals in tmp.items():
        G[u][v]["members"] = vals

    return G
