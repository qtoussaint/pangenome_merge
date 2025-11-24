import subprocess

# create mmseqs database
def mmseqs_createdb(fasta, outdb, threads, nt2aa: bool):

    # create compressed amino acid database from fasta
    if nt2aa is True:

        # create nt database:
        tempfile = f'{str(outdb)}_nt'
        cmd = f'mmseqs createdb {str(fasta)} {str(tempfile)} --compressed 1 -v 3 --threads {str(threads)}'
        subprocess.run(cmd, shell=True, check=True)

        # convert from nt to amino acid db:
        cmd = f'mmseqs translatenucs {str(tempfile)} {str(outdb)} --compressed 1 -v 3 --threads {str(threads)}'
        subprocess.run(cmd, shell=True, check=True)

    if nt2aa is False:
        # create amino acid db:
        cmd = f'mmseqs createdb {str(fasta)} {str(outdb)} --compressed 1 -v 3 --threads {str(threads)}'
        subprocess.run(cmd, shell=True, check=True)
    
    return

# concatenate two mmseqs databases and index (used to create new pangenome database after graph is updated with new nodes)
def mmseqs_concatdbs(db1, db2, outdb, tmpdir, threads):

    cmd = f'mmseqs concatdbs {str(db1)} {str(db2)} {str(outdb)} --preserve-keys 1 --compressed 1 -v 3 --threads {str(threads)}'
    
    subprocess.run(cmd, shell=True, check=True)

    #cmd = f'mmseqs createindex {str(outdb)} {str(tmpdir)} -v 2 --threads {str(threads)}'

    #subprocess.run(cmd, shell=True, check=True)

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
    subprocess.run(f'rm -f -- {str(resultdb)}*', shell=True, check=True)

    # basic inputs/outputs
    cmd = f'mmseqs search {str(querydb)} {str(targetdb)} {str(resultdb)} {str(tmpdir)} '
    
    # AA search with minimum aligned coverage specified
    # calculate coverage fraction globally (--cov-mode 0)
    # alignment mode 1 might not be possible but will try (otherwise need align mode 3 or -a)
    cmd += f' -a --cov-mode 0 -c {str(coverage)} '
    
    # minimum identity and sequential sensitivity steps for speedup
    # default mmseqs sensitivity is 5.7 so can lower last step to speed up if needed
    cmd += f' --min-seq-id {str(fident)} --start-sens 1 --sens-steps 3 -s 5.7 -v 3 --threads {str(threads)}'
    
    subprocess.run(cmd, shell=True, check=True)

    # output format, verbosity, and threads
    cmd = f' mmseqs convertalis {str(querydb)} {str(targetdb)} {str(resultdb)} {str(resultm8)} --format-mode 4 --format-output "query,target,fident,alnlen,qlen,tlen,evalue" -v 3 --threads {str(threads)}'

    subprocess.run(cmd, shell=True, check=True)

    return
