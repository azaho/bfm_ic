#!/bin/bash
#SBATCH --job-name=bfm_data_prep          # Name of the job
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=32G    # memory per cpu-core
#SBATCH -t 8:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-39       # 40 total combinations (4*5*2)
#SBATCH --output ./r/%A_%a.out # STDOUT
#SBATCH --error ./r/%A_%a.err   # STDERR

source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

filename_string=(
    'braintreebank_process_benchmark_chunks.py --spectrogram 0 --save_to_dir braintreebank_datachunks_benchmark_raw'
    'braintreebank_process_benchmark_chunks.py --spectrogram 1 --save_to_dir braintreebank_datachunks_benchmark'
    'braintreebank_process_chunks.py --spectrogram 0 --save_to_dir braintreebank_datachunks_raw'
    'braintreebank_process_chunks.py --spectrogram 1 --save_to_dir braintreebank_datachunks'
)
sub_id=$(((SLURM_ARRAY_TASK_ID % 10)+1))
filename_id=$((SLURM_ARRAY_TASK_ID / 10 % 4))

echo "sub_id: $sub_id"
echo "filename_id: $filename_id"

python ${filename_string[filename_id]} --sub_id $sub_id