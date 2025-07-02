import subprocess

# run mmseqs easy-search
def run_mmseqs_easysearch(
        query,
        target,
        outdir,
        tmpdir,
        threads):

    cmd = "mmseqs easy-search "
    
    cmd += str(query) + " "
    
    cmd += str(target) + " "
    
    cmd += str(outdir) + " " 

    cmd += str(tmpdir)
    
    cmd += ' --search-type 2 --format-mode 4 -a --format-output "query,target,fident,nident,alnlen,qlen,tlen,evalue" -v 1 --threads '
    
    cmd += str(threads)

    subprocess.run(cmd, shell=True, check=True)

    return
