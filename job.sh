#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --switches=1@600   # try to find a set of nodes on the same switch when doing multi-node.

set -e  # exit on the first error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Job has been preempted ${SLURM_RESTART_COUNT:-0} times so far."

## Code checkpointing with git to avoid unexpected bugs ##
UV_DIR=$(./code_checkpointing.sh)
echo "Git commit used for this job: ${GIT_COMMIT:-not set - code checkpointing is not enabled}"
echo "Running uv commands in directory: $UV_DIR"


# These environment variables are used by the torch and jax distributed modules, and should
# ideally be set before running the python script, or at the very beginning of the python script.
# Master address is the hostname of the first node in the job.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

echo "Preparing dataset"
# TODO:
# We should copy the dataset files from $SCRATCH to $SLURM_TMPDIR at the start of the job if they exist.
# Otherwise, we should prepare the dataset in $SLURM_TMPDIR and copy it back to $SCRATCH at the end of the job.
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES:-1} bash -c "uv run --directory=$UV_DIR python prepare.py"

#TODO: We might need to create links to the results.tsv file in the submit directory,
#since if `safe_sbatch` was used to submit the job, UV_DIR might be inside SLURM_TMPDIR.
# If UV_DIR is not empty, then UV_DIR is in $SLURM_TMPDIR.
if [[ -n "$UV_DIR" ]]; then
    bash -c "ln -s $SLURM_SUBMIT_DIR/results.tsv"
    bash -c "ln -s $SLURM_SUBMIT_DIR/logs"
fi

# Here we use srun to launch the task across potentially many GPUs and nodes.
# This setup is fully flexible as to how the GPUs are distributed accross nodes.

# Important note: Some variables (for example RANK, LOCAL_RANK, or machine_rank
# in accelerate) vary between tasks, so we need to escape env variables such as $SLURM_PROCID,
# $SLURM_TMPDIR and $SLURM_NODEID so they are evaluated within each task, not just once here
# on the first node.

# --gres-flags=allow-task-sharing is required to allow tasks on the same node to
# access GPUs allocated to other tasks on that node. Without this flag,
# --gpus-per-task=1 would isolate each task to only see its own GPU, which
# causes mysterious NCCL errors when NCCL tries to communicate to local GPUs
# via shared memory but fails due to cgroups isolation.
# See https://slurm.schedmd.com/srun.html#OPT_gres-flags
# and https://support.schedmd.com/show_bug.cgi?id=17875 for details.
# set +e
srun --gres-flags=allow-task-sharing bash -c \
    "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    uv run --directory=$UV_DIR $@"
# rsync -a $UV_DIR/logs logs
