#!/bin/bash
#SBATCH --job-name=benchmark     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=01:02:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ftownes@princeton.edu

#example usage, default is 4G memory per core (--mem-per-cpu), --mem is total
#CSV=./simulations/bm_sp/results/benchmark.csv
#PTH=./simulations/bm_sp
#sbatch --mem=16G --array=1-$(wc -l < $CSV) ./simulations/benchmark.slurm $PTH

module purge
module load anaconda3/2021.5
conda activate fwt

#first command line arg $1 is file path to parent directory of dataset
python -um simulations.benchmark $SLURM_ARRAY_TASK_ID $1
