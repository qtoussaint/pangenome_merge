#!/bin/bash
#SBATCH --job-name=snakemake
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/logs/snakemake/snakemake.%j.out
#SBATCH --error=/path/to/logs/snakemake/snakemake.%j.err

# maximum number of concurrent jobs (across all rules)
max_concurrent=200

# maximum number of concurrent jobs in job array
max_concurrent_array=20

# fresh install of micromamba 
#source ~/.bashrc
#micromamba activate snakemake

# with snakemake env activated:
snakemake --executor slurm -j $max_concurrent --group-components job_array=$max_concurrent_array --default-resources slurm_account=jlees

# then run:
#sbatch run_snakemake.sbatch
