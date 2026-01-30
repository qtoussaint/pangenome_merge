import sqlite3
from pathlib import Path
from collections import Counter

def canon_uv(u, v):
    u, v = str(u), str(v)
    return (u, v) if u <= v else (v, u)

def sqlite_connect(database: str, sqlite_cache: int) -> sqlite3.Connection:
    Path(database).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(database)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=OFF;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA busy_timeout=5000;") 
    con.execute("PRAGMA wal_autocheckpoint=100000;")
    con.execute(f"PRAGMA cache_size=-{sqlite_cache};")
    return con

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
        PRIMARY KEY (node_id, seqid)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS node_geneids (
        node_id TEXT,
        geneid TEXT,
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
    """)
    con.commit()

def sqlite_create_indexes(con: sqlite3.Connection):
    # cumulative tables keyed by node_id / (u,v)
    con.executescript("""
    CREATE INDEX IF NOT EXISTS idx_node_members_member ON node_members(member);
    CREATE INDEX IF NOT EXISTS idx_node_seqids_seqid ON node_seqids(seqid);
    CREATE INDEX IF NOT EXISTS idx_node_geneids_geneid ON node_geneids(geneid);
    """)
    con.commit()

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
                seqid_rows.append((node_id, s))

        geneIDs = data.get("geneIDs") or ""
        if str(geneIDs).strip():
            for gid in str(geneIDs).split(";"):
                gid = gid.strip()
                if gid:
                    geneid_rows.append((node_id, gid))

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
    cur.executemany("INSERT INTO node_seqids(node_id,seqid) VALUES (?,?)", seqid_rows)
    cur.executemany("INSERT INTO node_geneids(node_id,geneid) VALUES (?,?)", geneid_rows)
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
