#!/bin/bash
#SBATCH --job-name=bfm_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t 24:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-7      # 14 jobs (108/8 rounded up)
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH -p yanglab

source .venv/bin/activate

python ttt_clip_old_newer.py --two_views_split_eval $(($SLURM_ARRAY_TASK_ID%2)) --two_views_split $(($SLURM_ARRAY_TASK_ID/2%2)) --wandb_project ne_bofa1 --rs XXXXX \
    --dtype bfloat16 \
    --optimizer Muon \
    --spectrogram 1 \
    --binarize_eval 1 \
    --temp_clip_param 1 \
    --test_chunks_interleaved 0 \
    --multisubj_eval 1 \
    --electrode_embedding_init $(if [ $(($SLURM_ARRAY_TASK_ID/4)) -eq 0 ]; then echo "coordinates_nograd"; else echo "zeros"; fi) \
    --dr 0.2 \
    --dm 192 \
    --n_freq_features 64 \
    --bs 100 \
    --lrmax 0.0015 \
    --lrmin 0.0015 \
    --weight_decay 0 \
    --max_gradient_norm -1 \
    --subjects 1234567890 \
    --wait_n_intervals 0 \
    --nl 10