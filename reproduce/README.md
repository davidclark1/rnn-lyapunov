# Recomputing the paper's data from scratch

`data/` holds every number that the paper plots. The files in `jobs/` regenerate all of `data/` with the scripts
in `scripts/`, one command per data file, with the paper's exact settings. Each job list starts with a GPU-hour
estimate taken from the original runs on NVIDIA A100/H100 GPUs.

## Run a job list

```bash
source env.sh                                              # or activate your venv
reproduce/run_queue.sh reproduce/jobs/<list>.txt 0 1 2 3   # one worker per listed GPU id
```

- **Workers and logs.** Each worker pops the next command from a shared queue (`logs/queue_<list>.txt`, protected
  by `flock`), appends `--device cuda:0`, and runs it with `CUDA_VISIBLE_DEVICES` pinned to its GPU. Per-job logs
  and a status file per GPU go to `logs/`.
- **Several nodes.** Start the same command on each node, with that node's GPU ids. The queue file is shared through
  the file system.
- **Restart.** A failed job is marked `rc != 0` in the status file. To rerun, delete `logs/queue_<list>.txt` and
  start again; finished results are simply overwritten.
- **Output.** Results are written to `results/<kind>/` (git-ignored), in the same formats as `data/`.

## Rebuild the tables and figures from your results

Every loader and table builder in `paper/` takes `root=`. Point it at `results/` instead of `data/`:

```python
from paper import tables
tables.dimension_entropy_table(root="results")
```

A full comparison of recomputed and bundled numbers is `python tools/verify_dataset.py`. That script compares
against the original research tree; adapt its `root` to compare `results/` with `data/`.

## Expectations

- **Theory** (F(s), D_KY/N, h_KS/N, PR theory, maximum exponent) is deterministic. On the same GPU type it
  reproduces `data/` to ~1e-11 or better. On other hardware, expect differences at the level of the fixed-point
  tolerance (1e-7 in F).
- **Simulations** of chaotic networks are reproducible bitwise only on identical hardware and library versions.
  Elsewhere, trajectories decorrelate after a few Lyapunov times. Spectra and participation ratios then agree
  within their statistical errors (`docs/NUMERICS.md`).
- **DMFT solutions** are cached in `$RNN_LYAPUNOV_CACHE/dmft` (default `~/.cache/rnn_lyapunov/dmft`). Delete the
  cache to recompute them.
