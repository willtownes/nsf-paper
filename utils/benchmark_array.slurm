#!/bin/bash
#SBATCH --job-name=benchmark     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ftownes@princeton.edu

#example usage, default is 4G memory per core (--mem-per-cpu), --mem is total
#CSV=./scrna/visium_brain_sagittal/results/benchmark.csv
#DAT=./scrna/visium_brain_sagittal/data/visium_brain_sagittal_J2000.h5ad
#sbatch --mem=72G --array=1-$(wc -l < $CSV) ./util/benchmark_array.slurm $DAT

module purge
module load anaconda3/2021.5
conda activate fwt

#first command line arg $1 is file path to dataset
python -um utils.benchmark $SLURM_ARRAY_TASK_ID $1
