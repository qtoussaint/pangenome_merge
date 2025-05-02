import subprocess

# run mmseqs easy-search
def run_mmseqs_easysearch(
        query,
        target,
        outdir,
        tmpdir):

    cmd = "mmseqs easy-search "
    
    cmd += str(query) + " "
    
    cmd += str(target) + " "
    
    cmd += str(outdir) + " " 

    cmd += str(tmpdir)
    
    cmd += ' --search-type 2 --format-mode 4 -a --format-output "query,target,fident,nident,alnlen,qlen,tlen,evalue"'

    subprocess.run(cmd, shell=True, check=True)

    return
