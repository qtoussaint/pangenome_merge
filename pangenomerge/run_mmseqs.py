import subprocess

# create mmseqs database
def mmseqs_createdb(fasta, outdb, threads, nt2aa: bool):

    # create compressed amino acid database from fasta
    if nt2aa is True:

        # create nt database:
        tempfile = f'{str(outdb)}_nt'
        cmd = f'mmseqs createdb {str(fasta)} {str(tempfile)} --compressed 1 -v 3 --threads {str(threads)}'
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        check_result(result)

        # convert from nt to amino acid db:
        cmd = f'mmseqs translatenucs {str(tempfile)} {str(outdb)} --compressed 1 -v 3 --threads {str(threads)}'
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        check_result(result)

    if nt2aa is False:
        # create amino acid db:
        cmd = f'mmseqs createdb {str(fasta)} {str(outdb)} --compressed 1 -v 3 --threads {str(threads)}'
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        check_result(result)

    
    return

# concatenate two mmseqs databases and index (used to create new pangenome database after graph is updated with new nodes)
def mmseqs_concatdbs(db1, db2, outdb, tmpdir, threads):

    cmd = f'mmseqs concatdbs {str(db1)} {str(db2)} {str(outdb)} --compressed 1 -v 3 --threads 1'
    
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    check_result(result)

    # now create the header database for outdb (doesn't happen automatically)
    
    cmd = f"mmseqs concatdbs {str(db1)} {str(db2)} {str(outdb)} --threads 1"

    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    check_result(result)

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
        threads):

    # remove any existing results db
    result = subprocess.run(f'rm -f -- {str(resultdb)}*', shell=True, check=True, capture_output=True, text=True)
    check_result(result)

    # basic inputs/outputs
    cmd = f'mmseqs search {str(querydb)} {str(targetdb)} {str(resultdb)} {str(tmpdir)} '
    
    # AA search with minimum aligned coverage specified
    # calculate coverage fraction globally (--cov-mode 0)
    # alignment mode 1 might not be possible but will try (otherwise need align mode 3 or -a)
    cmd += f' -a --cov-mode 0 -c {str(coverage)} '
    
    # minimum identity and sequential sensitivity steps for speedup
    # default mmseqs sensitivity is 5.7 so can lower last step to speed up if needed
    cmd += f' --min-seq-id {str(fident)} --start-sens 1 --sens-steps 3 -s 5.7 -v 3 --threads {str(threads)}'
    
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    check_result(result)

    # output format, verbosity, and threads
    cmd = f' mmseqs convertalis {str(querydb)} {str(targetdb)} {str(resultdb)} {str(resultm8)} --format-mode 4 --format-output "query,target,fident,alnlen,qlen,tlen,evalue" -v 3 --threads {str(threads)}'

    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    check_result(result)

    return

def check_result(result):
    if result.returncode != 0:
        logging.error(f"MMseqs command failed: {cmd}")
        logging.error(f"STDOUT:\n{result.stdout}")
        logging.error(f"STDERR:\n{result.stderr}")
        result.check_returncode()