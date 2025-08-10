#!/usr/bin/env bash
# Usage:
# Important: run as a module with -m
#   ./submit_train.sh python3 -m training.vtxFramework_v2
#   ./submit_train.sh python3 -m training.vtxFramework_v2 --cfg configs/expA.yaml --seed 123

# exit immediately if any command returns a non-zero status.
# treat use of unset variables as an error (helps catch typos like $PYTNONPATH).
# a pipeline fails if any command in it fails (not just the last one).
set -euo pipefail


# don’t accidentally run the script without the command to execute.
if [ $# -lt 1 ]; then
  echo "Usage: $0 <command...>"
  exit 1
fi


# --- where to stage snapshots
SCRATCH_BASE="/scratch-cbe/users/alikaan.gueven/ML_KAAN/runs"
mkdir -p "$SCRATCH_BASE"


LOG_JSON="${SCRATCH_BASE}/submit_log.json"
JOB_OUT_DIR="/scratch-cbe/users/alikaan.gueven/job_outs"
mkdir -p "$JOB_OUT_DIR"


# --- figure out repo root and a commit hash (if available)
if git rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git rev-parse --show-toplevel)"
  GITHASH="$((cd "$REPO_ROOT" && git rev-parse --short HEAD) 2>/dev/null || echo 'no-git')"
else
  REPO_ROOT="$(pwd)"
  GITHASH="no-git"
fi


# find next runN (atomically)
n=1
while :; do
  RUN_DIR="${SCRATCH_BASE}/run${n}"
  if mkdir "$RUN_DIR" 2>/dev/null; then break; fi   # only one submit wins per N
  n=$((n+1))
done


# snapshot code
rsync -a --delete \
  --exclude='.git' \
  --exclude='*.root' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='notebooks/' \
  --exclude='training/tb/' \
  --exclude='training/.neptune/' \
  --exclude='z_tmp/' \
  "$REPO_ROOT/." "$RUN_DIR/"



SBATCH_ARGS=( --chdir="$RUN_DIR"                    \
              --output="$JOB_OUT_DIR/job_%j.out"    \
              slurm_scripts/to_gm.sh "$@"           \
            )


# write the command into a string
SBATCH_CMD="sbatch"
for a in "${SBATCH_ARGS[@]}"; do SBATCH_CMD+=" $(printf %q "$a")"; done


# submit: run inside snapshot; write stdout into the job_out_dir
OUT="$(sbatch --parsable "${SBATCH_ARGS[@]}")"
JOB_ID="${OUT%%_*}"   # get the job ID


# append to JSON log (JobID as key)
python3 utils/submit_logger.py                              \
    --log "$LOG_JSON" --jobid "$JOB_ID" --git "$GITHASH"    \
    --cmd "$SBATCH_CMD"


echo "Submitted batch job $JOB_ID"
# echo "Run dir:  $RUN_DIR"
# echo "Log:      $RUN_DIR/job_${JOB_ID}.out"