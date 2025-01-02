# NOTE: Settings in this file have global effect on the code. All parts of the pipeline have to run with the same settings.
# If you want to change a setting, you have to rerun all parts of the pipeline with the new setting. Otherwise, things will break.

# Root directory for the data
ROOT_DIR = "" # "" usually
# Sampling rate
SAMPLING_RATE = 2048
SPECTROGRAM_DIMENSIONALITY = 129 # number of frequency bins in the spectrogram

# Laplacian rereferencing of electrodes before processing
LAPLACIAN_REREFERENCED = False
# Allow corrupted electrodes to be part of the pipeline
ALLOW_CORRUPTED_ELECTRODES = False

# Disable file locking for HDF5 files. This is helpful for parallel processing.
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

BENCHMARK_START_DATA_BEFORE_ONSET = 1 # in seconds
BENCHMARK_END_DATA_AFTER_ONSET = 2 # in seconds
BENCHMARK_PADDING_TIME = 1 # in seconds
