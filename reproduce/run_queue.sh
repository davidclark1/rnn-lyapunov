#!/bin/bash
# Run a job list on GPUs: one worker per GPU pops lines from a shared queue (under flock) until it is empty.
#
#   reproduce/run_queue.sh <jobs.txt> <gpu ids...>          e.g.  reproduce/run_queue.sh reproduce/jobs/fig2_theory.txt 0 1 2 3
#
# Each line of <jobs.txt> is a python command relative to the repo root (blank lines and '#' comments are skipped);
# "--device cuda:0" is appended and CUDA_VISIBLE_DEVICES pins one GPU per worker (one process per GPU).  The queue
# is a copy, logs/queue_<name>.txt; logs per job in logs/<name>_<host>_gpu<id>_<n>.log, a status line per job in
# logs/<name>_<host>_gpu<id>.status.  To use several nodes, start the script on each node with the same <jobs.txt>
# AFTER the first start (the queue copy is shared through the file system; a second start reuses it).
set -u
cd "$(dirname "$0")/.." && source env.sh >/dev/null 2>&1
jobs=$1; shift; name=$(basename "$jobs" .txt)
mkdir -p logs; Q=logs/queue_$name.txt; L=logs/queue_$name.lock
[ -e "$Q" ] || command grep -v '^\s*\(#\|$\)' "$jobs" > "$Q"
for gpu in "$@"; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu; S=logs/${name}_$(hostname -s)_gpu$gpu.status; n=0
    while true; do
      cmd=$(flock "$L" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
      [ -z "$cmd" ] && break; n=$((n+1))
      echo "[$(date +%F' '%T)] START $cmd" >> "$S"
      python $cmd --device cuda:0 > "logs/${name}_$(hostname -s)_gpu${gpu}_$n.log" 2>&1
      rc=$?
      echo "[$(date +%F' '%T)] END rc=$rc $cmd" >> "$S"
    done
    echo "[$(date +%F' '%T)] QUEUE EMPTY" >> "$S"
  ) &
done
wait
