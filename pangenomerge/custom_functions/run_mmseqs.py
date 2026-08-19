import os
import subprocess

# create mmseqs database
def mmseqs_createdb(fasta, outdb, threads, nt2aa: bool, compressed: int = 1):
    # compressed: pass 0 for databases that will be searched with --search-type 3 (nucleotide).
    # mmseqs' nucleotide search runs `splitsequence` on the target whenever it holds a sequence
    # longer than --max-seq-len (10000); on a --compressed 1 database that step emits a
    # target_seqs_split whose .index offsets overrun its .data, and the prefilter dies with
    # "Invalid database read". Uncompressed databases are unaffected.

    # create compressed amino acid database from fasta
    if nt2aa is True:

        # create nt database:
        tempfile = f'{str(outdb)}_nt'
        cmd = f'mmseqs createdb {str(fasta)} {str(tempfile)} --compressed 1 -v 3 --threads {str(threads)}'
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    

        # convert from nt to amino acid db:
        cmd = f'mmseqs translatenucs {str(tempfile)} {str(outdb)} --compressed 1 -v 3 --threads {str(threads)}'
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    

    if nt2aa is False:
        # create amino acid db:
        cmd = f'mmseqs createdb {str(fasta)} {str(outdb)} --compressed {str(compressed)} -v 3 --threads {str(threads)}'
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    

    
    return

# concatenate two mmseqs databases and index (used to create new pangenome database after graph is updated with new nodes)
def mmseqs_concatdbs(db1, db2, outdb, tmpdir, threads):

    cmd = f'mmseqs concatdbs {str(db1)} {str(db2)} {str(outdb)} --compressed 1 -v 3 --threads 1'
    
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


    # now create the header database for outdb (doesn't happen automatically)
    
    cmd = f"mmseqs concatdbs {str(db1)}_h {str(db2)}_h {str(outdb)}_h --threads 1"

    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


    return

# run mmseqs search
def run_mmseqs_search(
        querydb,
        targetdb,
        resultdb,
        resultm8,
        tmpdir,
        fident,
        coverage,
        threads,
        search_type=None,
        kmer=None):

    # remove any existing results db
    result = subprocess.run(f'rm -f -- {str(resultdb)}*', shell=True, check=True, capture_output=True, text=True)


    # basic inputs/outputs
    cmd = f'mmseqs search {str(querydb)} {str(targetdb)} {str(resultdb)} {str(tmpdir)} '

    # AA search with minimum aligned coverage specified
    # calculate coverage fraction globally (--cov-mode 0)
    # alignment mode 1 might not be possible but will try (otherwise need align mode 3 or -a)
    cmd += f' -a --cov-mode 0 -c {str(coverage)} '

    # minimum identity and sequential sensitivity steps for speedup
    # default mmseqs sensitivity is 5.7 so can lower last step to speed up if needed
    cmd += f' --min-seq-id {str(fident)} --start-sens 1 --sens-steps 3 -s 5.7 -v 3 --threads {str(threads)}'

    # explicit search type (e.g. 3 = nucleotide/nucleotide); default lets mmseqs auto-detect (protein)
    if search_type is not None:
        cmd += f' --search-type {str(search_type)}'

    # explicit k-mer length (bounds nucleotide prefilter index memory; default lets mmseqs pick)
    if kmer is not None:
        cmd += f' -k {str(kmer)}'
    
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


    # output format, verbosity, and threads.
    #
    # Ask for qheader/theader rather than query/target: convertalis mangles any identifier it
    # mistakes for a UCSC accession, stripping a leading "uc" (uctC_1 -> tC_1, ucFOO -> FOO).
    # Callers match graph nodes to hits by name, so a mangled id means the hit is found at full
    # identity and then silently discarded -- the gene is never merged and shows up as accessory
    # content. createdb and translatenucs store the header correctly; only this export corrupts
    # it. qheader/theader carry it verbatim. Swapping the columns rather than adding them keeps
    # the .m8 byte size unchanged.
    cmd = f' mmseqs convertalis {str(querydb)} {str(targetdb)} {str(resultdb)} {str(resultm8)} --format-mode 4 --format-output "qheader,theader,fident,alnlen,qlen,tlen,evalue" -v 3 --threads {str(threads)}'

    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    # present the columns as query/target so every caller reads the .m8 unchanged
    _restore_m8_query_target(resultm8)

    return


def _restore_m8_query_target(resultm8):
    """Rename an .m8's qheader/theader columns to query/target, keeping only each id's first token.

    Six call sites write these files and six readers index them by `query`/`target`; requesting
    qheader/theader is what avoids convertalis' identifier mangling, so the columns are renamed
    here once instead of at every reader. Idempotent: a file already carrying query/target is
    left alone.
    """
    path = str(resultm8)
    if not os.path.exists(path):
        return

    with open(path) as src:
        header = src.readline()
        if not header.strip():
            return
        cols = header.rstrip("\n").split("\t")
        if "qheader" not in cols and "theader" not in cols:
            return
        rename = {"qheader": "query", "theader": "target"}
        idx = [i for i, c in enumerate(cols) if c in rename]

        tmp = f"{path}.tmp"
        with open(tmp, "w") as dst:
            dst.write("\t".join(rename.get(c, c) for c in cols) + "\n")
            for line in src:
                # a FASTA header may carry a description after whitespace; the id is the first token
                fields = line.rstrip("\n").split("\t")
                for i in idx:
                    if i < len(fields):
                        fields[i] = fields[i].split()[0] if fields[i].strip() else fields[i]
                dst.write("\t".join(fields) + "\n")

    os.replace(tmp, path)
