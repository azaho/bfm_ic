#!/bin/bash
#SBATCH --job-name=bfm_data_prep          # Name of the job
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=64G    # memory per cpu-core
#SBATCH -t 8:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-19       # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /shared/anzah/bfm_ic/r/%A_%a.out # STDOUT

source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

filename_string=('braintreebank_process_benchmark_chunks.py' 'braintreebank_process_chunks.py')
spectrogram_string=('--spectrogram 0 --save_to_dir braintreebank_benchmark_data_chunks_raw' '--spectrogram 1 --save_to_dir braintreebank_benchmark_data_chunks')
sub_id=$(((SLURM_ARRAY_TASK_ID % 10)+1))
spectrogram_id=$((SLURM_ARRAY_TASK_ID / 10 % 2))
filename_id=$((SLURM_ARRAY_TASK_ID / 20 % 2))

echo "sub_id: $sub_id"
echo "spectrogram_id: $spectrogram_id"

python ${filename_string[filename_id]} --sub_id $sub_id ${spectrogram_string[spectrogram_id]}
