#!/bin/bash

#SBATCH --job-name=peerige
#SBATCH --account=p805
#SBATCH --time=2-00:00:00              # days-hrs:min:sec
#SBATCH --ntasks=1                     # single process, multi-threaded
#SBATCH --cpus-per-task=32             # threads for z-solves and AI CG solves
#SBATCH --mem-per-cpu=3G

# Submit from the PeerIGE.jl project root: sbatch run_hpc.sh
# Data folder, run name and model are all set inside RUN_SCRIPT.
RUN_SCRIPT="run_emai_hpc.jl"

set -o errexit
set -o nounset

module --quiet purge
module load Julia/1.12.2
module list

# Prevent BLAS from spawning its own threads inside Julia's Threads.@threads blocks
export OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
echo "Job started: $(date) on $(hostname), threads: $SLURM_CPUS_PER_TASK"

julia --threads "$SLURM_CPUS_PER_TASK" --project=. "$RUN_SCRIPT"

echo "Job finished: $(date)"