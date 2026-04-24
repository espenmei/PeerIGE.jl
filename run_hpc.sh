#!/bin/bash

#SBATCH --job-name=peerige
#SBATCH --account=YourAccount          # replace with your project account
#SBATCH --time=5-00:00:00              # 120 hours = 5 days
#SBATCH --ntasks=1                     # single process, multi-threaded
#SBATCH --cpus-per-task=32             # threads for z-solves and AI CG solves
#SBATCH --mem-per-cpu=8G               # 32 * 8G = 256 GB total

set -o errexit
set -o nounset

module --quiet purge
module load Julia/1.12.2  # adjust to the version available on your cluster
module list

# Prevent BLAS from spawning its own threads inside Julia's Threads.@threads blocks
export OPENBLAS_NUM_THREADS=1

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Julia threads: $SLURM_CPUS_PER_TASK"
echo "Working directory: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

if [ ! -f Project.toml ]; then
	echo "Project.toml not found in $SLURM_SUBMIT_DIR"
	echo "Submit this job from the PeerIGE.jl project root."
	exit 1
fi

julia --threads "$SLURM_CPUS_PER_TASK" --project=. run_emai.jl

echo "Job finished: $(date)"
