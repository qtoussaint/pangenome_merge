#!/bin/bash
#SBATCH --job-name=snakemake
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/logs/snakemake/snakemake.%j.out
#SBATCH --error=/path/to/logs/snakemake/snakemake.%j.err

# path to snakemake pipeline (developed in pangenomerge)
snake=/path/to/pangenome_merge/snakemake/Snakefile

# path to config file (unique for each project)
config=/path/to/project_directory/config.yaml

# maximum number of concurrent jobs (across all rules)
max_concurrent=200

# maximum number of concurrent jobs in each task in group
# gets weird if you do >1
max_concurrent_array=1

# the slurm account you'd like to submit jobs from
slurm_acct=myaccount

# with snakemake env activated:
snakemake --executor slurm -j $max_concurrent --group-components job_array=$max_concurrent_array --default-resources slurm_account=$slurm_acct --use-conda --latency-wait 60 --verbose --snakefile $snake --configfile $config

# once you've created a config.yaml for your project added your desired options to this script, run snakemake using:
sbatch example_slurm_run.sh

### TROUBLESHOOTING

# fresh install of micromamba 
#source ~/.bashrc
#micromamba activate snakemake

# possibly necessary to avoid conda issues
#unset -f conda
#hash -r
#conda --version   # should now be 26.1.0