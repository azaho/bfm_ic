#!/bin/bash
#SBATCH --job-name=bfm_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:RTXA6000:1 # A6000
#SBATCH --mem-per-gpu=128G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-1      # 14 jobs (108/8 rounded up)
#SBATCH --output r/jupyter.out # STDOUT

source .venv/bin/activate

unset XDG_RUNTIME_DIR
port=5969

jupyter lab --ip=0.0.0.0 --port=${port} --no-browser --NotebookApp.allow_origin='*' --NotebookApp.port_retries=0