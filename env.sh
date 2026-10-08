# Flatiron rusty: `source env.sh` loads the module Python (torch + CUDA, numpy, scipy, matplotlib, pytest) and the venv.
# Elsewhere: python -m venv .venv && source .venv/bin/activate && pip install -e ".[test]"
source /etc/profile.d/modules.sh 2>/dev/null
module load python cuda cudnn nccl 2>/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
