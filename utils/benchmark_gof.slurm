#!/bin/bash
#SBATCH --job-name=benchmark_gof     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ftownes@princeton.edu

#example usage, --mem-per-cpu default is 4G per core
#DAT=./scrna/sshippo/data/sshippo_J2000.h5ad
#sbatch --mem=180G ./utils/benchmark_gof.slurm $DAT

module purge
module load anaconda3/2021.5
conda activate fwt

#first command line arg $1 is file path to dataset
#second command line arg $2 is an integer for the pct of data to be validation set (typically 5, implying 95% of observations are for training data)
python -um utils.benchmark_gof $1 $2
