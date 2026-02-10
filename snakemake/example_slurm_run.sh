#!/bin/bash
#SBATCH --job-name=smk-driver
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/smk-driver.%j.out
#SBATCH --error=logs/smk-driver.%j.err

# with snakemake conda env activated:

snakemake --profile slurm -s Snakefile  -j 200

# then run:
#sbatch run_snakemake.sbatch
