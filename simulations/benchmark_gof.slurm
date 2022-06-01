#!/bin/bash
#SBATCH --job-name=benchmark_gof     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ftownes@princeton.edu

#example usage, --mem-per-cpu default is 4G per core
#PTH=./simulations/bm_sp
#sbatch --mem=16G ./simulations/benchmark_gof.slurm $PTH

module purge
module load anaconda3/2021.5
conda activate fwt

#first command line arg $1 is file path to parent directory
python -um simulations.benchmark_gof $1
