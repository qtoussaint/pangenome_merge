import sqlite3, json
from pathlib import Path

def canon_uv(u, v):
u, v = str(u), str(v)
return (u, v) if u <= v else (v, u)

def sqlite_connect(database: str) -> sqlite3.Connection:
Path(database).parent.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(database)
con.execute("PRAGMA journal_mode=WAL;")
con.execute("PRAGMA synchronous=NORMAL;")
con.execute("PRAGMA foreign_keys=ON;")
return con

def sqlite_init(con: sqlite3.Connection):
con.executescript("""

CREATE TABLE IF NOT EXISTS nodes (
    iteration INTEGER,
    node_id TEXT,
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
    PRIMARY KEY (iteration, node_id)
);

CREATE TABLE IF NOT EXISTS node_members (
    iteration INTEGER,
    node_id TEXT,
    member TEXT,
    PRIMARY KEY (iteration, node_id, member)
);

CREATE TABLE IF NOT EXISTS node_seqids (
    iteration INTEGER,
    node_id TEXT,
    seqid TEXT,
    PRIMARY KEY (iteration, node_id, seqid)
);

CREATE TABLE IF NOT EXISTS node_geneids (
    iteration INTEGER,
    node_id TEXT,
    geneid TEXT,
    PRIMARY KEY (iteration, node_id, geneid)
);

CREATE TABLE IF NOT EXISTS node_centroids (
    iteration INTEGER,
    node_id TEXT,
    centroid TEXT,
    PRIMARY KEY (iteration, node_id, centroid)
);

CREATE TABLE IF NOT EXISTS node_lengths (
    iteration INTEGER,
    node_id TEXT,
    length INTEGER,
    PRIMARY KEY (iteration, node_id, length)
);

CREATE TABLE IF NOT EXISTS node_longCentroidID (
    iteration INTEGER,
    node_id TEXT,
    tag TEXT,
    PRIMARY KEY (iteration, node_id, tag)
);

CREATE TABLE IF NOT EXISTS node_sequences (
    iteration INTEGER,
    node_id TEXT,
    dna TEXT,
    protein TEXT,
    PRIMARY KEY (iteration, node_id)
);

CREATE TABLE IF NOT EXISTS edges (
    iteration INTEGER,
    u TEXT,
    v TEXT,
    size INTEGER,
    genomeIDs TEXT,
    PRIMARY KEY (iteration, u, v)
);

CREATE TABLE IF NOT EXISTS edge_members (
    iteration INTEGER,
    u TEXT,
    v TEXT,
    member TEXT,
    PRIMARY KEY (iteration, u, v, member)
);

CREATE INDEX IF NOT EXISTS idx_node_members_member ON node_members(member);
CREATE INDEX IF NOT EXISTS idx_node_seqids_seqid ON node_seqids(seqid);
CREATE INDEX IF NOT EXISTS idx_node_geneids_geneid ON node_geneids(geneid);
""")
con.commit()

def add_metadata_to_sqlite(G, database: str, iteration: int):
con = sqlite_connect(database)
sqlite_init(con)
cur = con.cursor()
cur.execute("BEGIN;")

# overwrite snapshot for this iteration
cur.execute("INSERT OR REPLACE INTO iterations(iteration, graph_file_1, graph_file_2) VALUES (?, ?, ?)",
            (iteration, graph_file_1, graph_file_2))

for tbl in ["nodes","node_members","node_seqids","node_geneids","node_centroids","node_lengths",
            "node_longCentroidID","node_sequences","edges","edge_members"]:
    cur.execute(f"DELETE FROM {tbl} WHERE iteration=?", (iteration,))

node_rows = []
members_rows = []
seqid_rows = []
geneid_rows = []
centroid_rows = []
length_rows = []
longcid_rows = []
seq_rows = []

for node_id, data in G.nodes(data=True):
    node_id = str(node_id)
    node_rows.append((
        iteration, node_id,
        data.get("name"),
        int(data.get("size", 0)) if data.get("size") is not None else None,
        int(data.get("degrees", 0)) if data.get("degrees") is not None else None,
        data.get("genomeIDs"),
        str(data.get("maxLenId")) if data.get("maxLenId") is not None else None,
        int(data.get("hasEnd", 0)) if data.get("hasEnd") is not None else None,
        data.get("annotation"),
        data.get("description"),
        int(data.get("paralog", 0)) if data.get("paralog") is not None else None,
        data.get("mergedDNA"),
    ))

    for m in (data.get("members") or []):
        members_rows.append((iteration, node_id, str(m)))

    for s in (data.get("seqIDs") or []):
        seqid_rows.append((iteration, node_id, str(s)))

    geneIDs = data.get("geneIDs", "")
    if geneIDs:
        for gid in str(geneIDs).split(";"):
            gid = gid.strip()
            if gid:
                geneid_rows.append((iteration, node_id, gid))

    for c in (data.get("centroid") or []):
        centroid_rows.append((iteration, node_id, str(c)))

    for L in (data.get("lengths") or []):
        if L is not None:
            length_rows.append((iteration, node_id, int(L)))

    for t in (data.get("longCentroidID") or []):
        longcid_rows.append((iteration, node_id, str(t)))

    dna = data.get("dna")
    protein = data.get("protein")
    # store in panaroo format
    dna_txt = ";".join(dna) if isinstance(dna, list) else dna
    prot_txt = ";".join(protein) if isinstance(protein, list) else protein
    seq_rows.append((iteration, node_id, dna_txt, prot_txt))

cur.executemany("""
    INSERT INTO nodes(iteration,node_id,name,size,degrees,genomeIDs,maxLenId,hasEnd,annotation,description,paralog,mergedDNA)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
""", node_rows)

cur.executemany("INSERT INTO node_members(iteration,node_id,member) VALUES (?,?,?)", members_rows)
cur.executemany("INSERT INTO node_seqids(iteration,node_id,seqid) VALUES (?,?,?)", seqid_rows)
cur.executemany("INSERT INTO node_geneids(iteration,node_id,geneid) VALUES (?,?,?)", geneid_rows)
cur.executemany("INSERT INTO node_centroids(iteration,node_id,centroid) VALUES (?,?,?)", centroid_rows)
cur.executemany("INSERT INTO node_lengths(iteration,node_id,length) VALUES (?,?,?)", length_rows)
cur.executemany("INSERT INTO node_longCentroidID(iteration,node_id,tag) VALUES (?,?,?)", longcid_rows)
cur.executemany("INSERT INTO node_sequences(iteration,node_id,dna,protein) VALUES (?,?,?,?)", seq_rows)

edge_rows = []
edge_member_rows = []
for u, v, edata in G.edges(data=True):
    u, v = canon_uv(u, v)
    edge_rows.append((
        iteration, u, v,
        int(edata.get("size", 0)) if edata.get("size") is not None else None,
        edata.get("genomeIDs")
    ))
    for m in (edata.get("members") or []):
        edge_member_rows.append((iteration, u, v, str(m)))

cur.executemany("INSERT INTO edges(iteration,u,v,size,genomeIDs) VALUES (?,?,?,?,?)", edge_rows)
cur.executemany("INSERT INTO edge_members(iteration,u,v,member) VALUES (?,?,?,?)", edge_member_rows)

cur.execute("COMMIT;")
con.close()
