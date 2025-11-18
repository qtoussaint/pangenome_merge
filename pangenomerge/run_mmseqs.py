import subprocess

# create mmseqs database
def mmseqs_createdb(fasta, outDB, threads):

    # create compressed amino acid database from fasta
    cmd = f'mmseqs createdb {str(fasta)} {str(outDB)} --db-type 2 --compressed -v 1 --threads {str(threads)}'

    subprocess.run(cmd, shell=True, check=True)
    return

# concatenate two mmseqs databases
def mmseqs_concatdbs(db1, db2, outDB, threads):

    cmd = f'mmseqs concatdbs {str(db1)} {str(db2)} {str(outDB)} --preserve-keys --compressed -v 1 --threads {str(threads)}'
    
    subprocess.run(cmd, shell=True, check=True)
    return

# run mmseqs search
def run_mmseqs_search(
        querydb,
        targetdb,
        outdir,
        tmpdir,
        fident,
        coverage,
        threads):

    # basic inputs/outputs
    cmd = f'mmseqs search {str(querydb)} {str(targetdb)} {str(outdir)} {str(tmpdir)} '
    
    # translated AA search, align
    cmd += f' --search-type 2 --alignment-mode 1 --cov-mod 0 -c {str(coverage)} '
    
    cmd += f'--min-seq-id {str(fident)} --start-sens 1 --sens-steps 3 -s 7 '
    
    cmd += f' --format-mode 4 --format-output "query,target,fident,alnlen,qlen,tlen,evalue" -v 1 --threads {str(threads)}'

    subprocess.run(cmd, shell=True, check=True)

    return
