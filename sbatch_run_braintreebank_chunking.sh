#!/bin/bash
#SBATCH --job-name=bfm_data_prep          # Name of the job
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=64G    # memory per cpu-core
#SBATCH -t 8:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-19       # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /shared/anzah/bfm_ic/reports/%A_%a.out # STDOUT

source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

spectrogram_string=('--spectrogram 0 --save_to_dir braintreebank_data_chunks_raw' '--spectrogram 1 --save_to_dir braintreebank_data_chunks')
sub_id = $((SLURM_ARRAY_TASK_ID % 10)+1)
spectrogram_id = $((SLURM_ARRAY_TASK_ID / 10))

echo "sub_id: $sub_id"
echo "spectrogram_id: $spectrogram_id"

python braintreebank_process_chunks.py --sub_id $sub_id $spectrogram_string[$spectrogram_id]