# NOTE: Settings in this file have global effect on the code. All parts of the pipeline have to run with the same settings.
# If you want to change a setting, you have to rerun all parts of the pipeline with the new setting. Otherwise, things will break.

# Root directory for the data
ROOT_DIR = "" # "" usually
# Sampling rate
SAMPLING_RATE = 2048
N_PER_SEG = 256
SPECTROGRAM_DIMENSIONALITY = 128 # number of frequency bins in the spectrogram

# Laplacian rereferencing of electrodes before processing
LAPLACIAN_REREFERENCED = False
# Allow corrupted electrodes to be part of the pipeline
ALLOW_CORRUPTED_ELECTRODES = False

WINDOW_LENGTH = N_PER_SEG * 8 * 10 * 18 # 180 seconds per chunk, to have fewer files
BENCHMARK_CHUNK_SIZE = 100

# Disable file locking for HDF5 files. This is helpful for parallel processing.
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

BENCHMARK_START_DATA_BEFORE_ONSET = 1.5 # in seconds
BENCHMARK_END_DATA_AFTER_ONSET = 3.5 # in seconds
BENCHMARK_PADDING_TIME = 0 # in seconds
BENCHMARK_NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP = 0.0 # proportion of overlap between consecutive nonverbal chunks (0 means no overlap)

assert BENCHMARK_NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP >= 0 and BENCHMARK_NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP < 1, "BENCHMARK_NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP must be between 0 and 1, strictly below 1"
