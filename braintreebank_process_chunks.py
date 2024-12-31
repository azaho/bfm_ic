# Part of the code is adapted from https://braintreebank.dev/, file "quickstart.ipynb"

import h5py
import os
import json
import pandas as pd
from scipy import signal, stats
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from braintreebank_config import *
from braintreebank_utils import Subject

def process_subject_trial(sub_id, trial_id, laplacian_rereferenced=False, max_chunks=None, nperseg=256,
                          window_length=None, verbose=True, global_per_electrode_normalizing_params=True, allow_corrupted=ALLOW_CORRUPTED_ELECTRODES, 
                          only_laplacian=False, save_to_dir="braintreebank_data_chunks", spectrogram=True, save_plot=False):
    if window_length is None: window_length = nperseg * 8 * 10 # 10 seconds
    assert window_length % nperseg == 0, "Window length must be divisible by nperseg"
    assert (not laplacian_rereferenced) or (only_laplacian), "Laplacian rereferenced is only supported when only_laplacian is True"

    subject = Subject(sub_id, allow_corrupted=allow_corrupted)
    subject.load_neural_data(trial_id)
    electrode_labels = subject.laplacian_electrodes if only_laplacian else subject.electrode_labels
    n_electrodes = len(electrode_labels)
    total_samples = subject.neural_data[trial_id]['electrode_0'].shape[0]

    if global_per_electrode_normalizing_params:
        if verbose: print("Computing normalizing parameters for electrodes")
        normalizing_params = {}
        for i, electrode_label in enumerate(electrode_labels):
            if spectrogram:
                mean, std = subject.get_spectrogram_normalizing_params(electrode_label, trial_id, laplacian_rereferenced=laplacian_rereferenced, cache=True)
            else:
                mean, std = subject.get_electrode_data_normalizing_params(electrode_label, trial_id, laplacian_rereferenced=laplacian_rereferenced, cache=True)
            normalizing_params[electrode_label] = (mean, std)
            #print(f"Normalizing params for {electrode_label}: mean={mean}, std={std}")
        if verbose: print("Normalizing parameters computed")

    if verbose: print(f"Processing subject {sub_id} trial {trial_id}")
    if verbose: print(f"Total time: {total_samples/subject.sampling_rate} seconds. One chunk is {window_length/subject.sampling_rate} seconds. There are {total_samples//window_length} chunks.")
    windows_range = range(0, total_samples, window_length)
    if windows_range[-1] + window_length > total_samples: windows_range = windows_range[:-1] # remove last chunk if it's not full
    if max_chunks is not None: windows_range = windows_range[:max_chunks]
    for window_from in windows_range:
        window_to = window_from + window_length
        if spectrogram:
            data_chunk = np.zeros((n_electrodes, (window_to - window_from) // nperseg, 37), dtype=np.float32)
            for i, electrode_label in enumerate(electrode_labels):
                f, t, Sxx = subject.get_spectrogram(electrode_label, trial_id, window_from=window_from, window_to=window_to, 
                                                    normalize_per_freq=True, laplacian_rereferenced=laplacian_rereferenced, cache=True,
                                                    normalizing_params=normalizing_params[electrode_label] if global_per_electrode_normalizing_params else None)
                data_chunk[i, :, :] = Sxx.T # data_chunk shape: (n_electrodes, n_time_bins, n_freqs)
        else:
            data_chunk = np.zeros((n_electrodes, (window_to - window_from) // nperseg, nperseg), dtype=np.float32)
            for i, electrode_label in enumerate(electrode_labels):
                if laplacian_rereferenced:
                    data_chunk[i, :, :] = subject.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to, cache=True).reshape(-1, nperseg)
                else:
                    data_chunk[i, :, :] = subject.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to, cache=True).reshape(-1, nperseg)
                if global_per_electrode_normalizing_params:
                    data_chunk[i, :, :] = (data_chunk[i, :, :] - normalizing_params[electrode_label][0].item()) / normalizing_params[electrode_label][1].item()
                else:
                    data_chunk[i, :, :] = (data_chunk[i, :, :] - np.mean(data_chunk[i, :, :]).item()) / np.std(data_chunk[i, :, :]).item()
        #print(data_chunk)
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        np.save(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_chunk{window_from//window_length}.npy', data_chunk)
        if verbose: print(f"Saved chunk {window_from//window_length}")

    # Save a plot of an example data chunk
    n_electrodes = data_chunk.shape[0]

    if save_plot:
        n_cols = 10
        n_rows = int(np.ceil(n_electrodes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
        axes = axes.flatten()
        for i in range(n_electrodes):
            im = axes[i].imshow(data_chunk[i].T, aspect='auto', origin='lower', cmap='viridis', vmin=-2, vmax=3)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        # Hide any unused subplots
        for i in range(n_electrodes, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_chunk{window_from//window_length}.pdf')
        plt.close()
        if verbose: print(f"Saved plot of chunk {window_from//window_length}")

    # Save metadata about the subject and trial
    n_time_bins = data_chunk.shape[1]
    n_freq_features = data_chunk.shape[2]
    with open(f'{save_to_dir}/subject{sub_id}_trial{trial_id}.json', 'w') as f:
        json.dump(
            {'subject_id': int(sub_id), 
             'trial_id': int(trial_id), 
             'laplacian_rereferenced': laplacian_rereferenced,
             'only_laplacian': only_laplacian,
             'allow_corrupted': allow_corrupted,
             'n_electrodes': int(n_electrodes),
             'n_time_bins': int(n_time_bins),
             'n_freq_features': int(n_freq_features),
             'total_samples': int(total_samples),
             'n_chunks': int(len(windows_range))}, f)
    if verbose: print(f"Saved metadata for subject {sub_id} trial {trial_id}")
    subject.close_all_files()
    del subject

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_id', type=int, required=False, help='Subject ID', default=-1)
    parser.add_argument('--trial_id', type=int, required=False, help='Trial ID', default=-1)
    parser.add_argument('--laplacian_rereferenced', type=bool, required=False, help='Laplacian rereferenced', default=LAPLACIAN_REREFERENCED)
    parser.add_argument('--max_chunks', type=int, required=False, help='Maximum number of chunks to process', default=None)
    parser.add_argument('--save_to_dir', type=str, required=False, help='Directory to save the data chunks', default="braintreebank_data_chunks")
    parser.add_argument('--spectrogram', type=int, required=False, help='Whether to use spectrogram features', default=0)
    parser.add_argument('--save_plot', type=bool, required=False, help='Whether to save a plot of an example data chunk', default=False)
    args = parser.parse_args()
    sub_id = args.sub_id
    trial_id = args.trial_id
    laplacian_rereferenced = args.laplacian_rereferenced
    max_chunks = args.max_chunks
    save_to_dir = args.save_to_dir
    spectrogram = args.spectrogram==1
    save_plot = args.save_plot
    assert (not sub_id<0) or (trial_id<0) # if no sub id provided, then process all trials for all subjects

    nperseg = 256 # TODO: move to braintreebank_config.py
    window_length = nperseg * 8 * 10 * 18 # 180 seconds per chunk, to have fewer files
    process_subject_ids = [sub_id] if sub_id > 0 else range(1, 11)
    for sub_id in process_subject_ids:
        process_trial_ids = [trial_id] if trial_id >= 0 else np.arange([3, 7, 3, 3, 1, 3, 2, 1, 1, 2][sub_id-1]) # if no trial id provided, then process all trials for the subject
        if (sub_id == 6) and (trial_id < 0): process_trial_ids = [0, 1, 4] # special case for subject 6 that only has trials 0, 1, and 4
        for trial_id in process_trial_ids:
            process_subject_trial(sub_id, trial_id, laplacian_rereferenced=laplacian_rereferenced, max_chunks=max_chunks, verbose=True, 
                                  global_per_electrode_normalizing_params=True, save_to_dir=save_to_dir, window_length=window_length, nperseg=nperseg, spectrogram=spectrogram, save_plot=save_plot)