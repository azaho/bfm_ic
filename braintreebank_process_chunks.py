# Part of the code is adapted from https://braintreebank.dev/, file "quickstart.ipynb"

import h5py
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy import signal, stats
import numpy as np
import seaborn as sns
import argparse

root_dir = ""
laplacian_rereferenced = False
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["HDF5_USE_FILE_LOCKING"]= "FALSE"

class Subject:
    def __init__(self, subject_id, sampling_rate=2048, allow_corrupted=True):
        self.subject_id = subject_id
        self.sampling_rate = sampling_rate
        self.localization_data = self._load_localization_data()
        self.neural_data = {}
        self.neural_data_cache = {}
        self.h5f_files = {}
        self.corrupted_electrodes = self._get_corrupted_electrodes()
        self.electrode_labels = self._get_electrode_names(allow_corrupted=allow_corrupted)
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}
        self.laplacian_electrodes, self.electrode_neighbors = self._get_all_laplacian_electrodes()

    def check_electrodes(self, trial_id):
        # Check for any mismatches between neural data keys and electrode labels
        neural_keys = set(list(self.neural_data[trial_id].keys()))
        electrode_label_set = set(["electrode_"+str(i) for i in self.electrode_ids.values()])
        # Find electrodes in neural data but not in labels
        extra_in_neural = neural_keys - electrode_label_set
        if extra_in_neural:
            print("WARNING: Electrodes in neural data but not in labels:", extra_in_neural)
        # Find electrodes in labels but not in neural data    
        missing_in_neural = electrode_label_set - neural_keys
        if missing_in_neural:
            print("WARNING: Electrodes in labels but not in neural data:", missing_in_neural)

    def _clean_electrode_label(self, electrode_label):
        if '*' in electrode_label:
            electrode_label = electrode_label.replace('*', '')
        if '#' in electrode_label:
            electrode_label = electrode_label.replace('#', '')
        return electrode_label

    def _get_corrupted_electrodes(self):
        corrupted_electrodes_file = os.path.join(root_dir, f'braintreebank_corrupted_elec.json')
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        return [self._clean_electrode_label(e) for e in corrupted_electrodes[f'subject{self.subject_id}']]

    def _get_electrode_names(self, allow_corrupted=True):
        electrode_labels_file = os.path.join(root_dir, f'braintreebank/electrode_labels/sub_{self.subject_id}/electrode_labels.json')
        electrode_labels = json.load(open(electrode_labels_file))
        electrode_labels = [self._clean_electrode_label(e) for e in electrode_labels]
        if allow_corrupted: return electrode_labels
        else: return [e for e in electrode_labels if e not in self.corrupted_electrodes]

    def _get_all_laplacian_electrodes(self, verbose=False):
        """
            Get all laplacian electrodes for a given subject. This function is originally from
            https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
        """
        def stem_electrode_name(name):
            #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
            #names look like 'T1b2
            reverse_name = reversed(name)
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return ''.join(reversed(stem)), int(''.join(reversed(num)))
        def has_neighbors(stem, stems):
            (x,y) = stem
            return ((x,y+1) in stems) and ((x,y-1) in stems)
        def get_neighbors(stem):
            (x,y) = stem
            return [f'{x}{y}' for (x,y) in [(x,y+1), (x,y-1)]]
        stems = [stem_electrode_name(e) for e in self.electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
        neighbors = {e: get_neighbors(stem_electrode_name(e)) for e in electrodes}
        return electrodes, neighbors

    def load_neural_data(self, trial_id):
        if trial_id in self.neural_data: return
        neural_data_file = os.path.join(root_dir, f'braintreebank/sub_{self.subject_id}_trial{trial_id:03}.h5')
        h5f = h5py.File(neural_data_file, 'r', locking=False)
        self.h5f_files[trial_id] = h5f
        self.neural_data[trial_id] = h5f['data']
        self.neural_data_cache[trial_id] = {}

    def _load_localization_data(self):
        """Load localization data for this electrode's subject from depth-wm.csv"""
        loc_file = f'braintreebank/localization/sub_{self.subject_id}/depth-wm.csv'
        df = pd.read_csv(loc_file)
        return df
    
    def clear_neural_data_cache(self, trial_id):
        self.neural_data_cache[trial_id] = {}

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None, cache=True):
        """
        Get the data for a given electrode for a given trial.
        If cache is True, all of the data is cached in self.neural_data_cache[trial_id][electrode_label] (not just the window)
        """
        neural_data_key = "electrode_"+str(self.electrode_ids[electrode_label])
        if trial_id not in self.neural_data_cache: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.neural_data[trial_id][neural_data_key].shape[0]
        if cache:
            if electrode_label not in self.neural_data_cache[trial_id]:
                self.neural_data_cache[trial_id][electrode_label] = self.neural_data[trial_id][neural_data_key][:]
            return self.neural_data_cache[trial_id][electrode_label][window_from:window_to]
        else: return self.neural_data[trial_id][neural_data_key][window_from:window_to]
    def get_laplacian_rereferenced_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None, cache=True):
        if electrode_label not in self.laplacian_electrodes:
            raise ValueError(f"Electrode {electrode_label} does not have neighbors")
        neighbors = self.electrode_neighbors[electrode_label]
        neighbor_data = [self.get_electrode_data(n, trial_id, window_from=window_from, window_to=window_to, cache=cache) for n in neighbors]
        return self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to, cache=cache)-np.mean(neighbor_data, axis=0)
    def get_spectrogram(self, electrode_label, trial_id, window_from=None, window_to=None, 
                        normalizing_params=None, laplacian_rereferenced=False, return_power=True, 
                        normalize_per_freq=False, cache=True, nperseg=256, noverlap=0, power_smoothing_factor=1e-5):
        if laplacian_rereferenced: 
            data = self.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, cache=cache, window_from=window_from, window_to=window_to)
        else: data = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to, cache=cache)

        f, t, Sxx = signal.spectrogram(data, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap, window='boxcar')
        f, Sxx = f[(f<300) & (f>=8)], Sxx[(f<300) & (f>=8)] # only keep frequencies up to 300 Hz and above 8 Hz
        if return_power: Sxx = 10 * np.log10(Sxx + power_smoothing_factor) # puts a lower bound of -50 on the power with the default power_smoothing_factor
        if normalize_per_freq: 
            if normalizing_params is None: 
                normalizing_params = np.mean(Sxx, axis=1), np.std(Sxx, axis=1)
            Sxx = (Sxx - normalizing_params[0][:, None])/normalizing_params[1][:, None]
        return f, t, Sxx
    def get_electrode_data_partial(self, electrode_label, trial_id, window_from, window_to, cache=True):
        data = self.get_electrode_data(electrode_label, trial_id, cache=cache)
        return data[window_from:window_to]
    def get_spectrogram_normalizing_params(self, electrode_label, trial_id, laplacian_rereferenced=False, cache=True):
        f, t, Sxx = self.get_spectrogram(electrode_label, trial_id, laplacian_rereferenced=laplacian_rereferenced, cache=cache)
        return np.mean(Sxx, axis=1), np.std(Sxx, axis=1)
    def close_all_files(self):
        for h5f in self.h5f_files.values():
            h5f.close()

def process_subject_trial(sub_id, trial_id, laplacian_rereferenced=False, max_chunks=None, nperseg=256, noverlap=0, window_length=None, verbose=True, global_per_electrode_normalizing_params=True, allow_corrupted=False, only_laplacian=False, save_to_dir="braintreebank_data_chunks"):
    if window_length is None: window_length = nperseg * 8 * 10 # 10 seconds
    assert window_length % nperseg == 0, "Window length must be divisible by nperseg"
    assert (not laplacian_rereferenced) or (only_laplacian), "Laplacian rereferenced is only supported when only_laplacian is True"

    subject = Subject(sub_id, allow_corrupted=allow_corrupted)
    subject.load_neural_data(trial_id)
    electrode_labels = subject.laplacian_electrodes if only_laplacian else subject.electrode_labels
    n_electrodes = len(electrode_labels) # TODO: remove corrupted electrodes
    total_samples = subject.neural_data[trial_id]['electrode_0'].shape[0]

    if global_per_electrode_normalizing_params:
        if verbose: print("Computing normalizing parameters for electrodes")
        normalizing_params = {}
        for i, electrode_label in enumerate(electrode_labels):
            mean, std = subject.get_spectrogram_normalizing_params(electrode_label, trial_id, laplacian_rereferenced=laplacian_rereferenced, cache=False)
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
        data_chunk = np.zeros((n_electrodes, (window_to - window_from) // nperseg, 37), dtype=np.float32)
        for i, electrode_label in enumerate(electrode_labels):
            f, t, Sxx = subject.get_spectrogram(electrode_label, trial_id, window_from=window_from, window_to=window_to, 
                                                normalize_per_freq=True, laplacian_rereferenced=laplacian_rereferenced, cache=False,
                                                normalizing_params=normalizing_params[electrode_label] if global_per_electrode_normalizing_params else None)
            data_chunk[i, :, :] = Sxx.T # data_chunk shape: (n_electrodes, n_time_bins, n_freqs)
        #print(data_chunk)
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        np.save(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_chunk{window_from//window_length}.npy', data_chunk)
        if verbose: print(f"Saved chunk {window_from//window_length}")

    # Save a plot of an example data chunk
    n_electrodes = data_chunk.shape[0]
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
    parser.add_argument('--laplacian_rereferenced', type=bool, required=False, help='Laplacian rereferenced', default=False)
    parser.add_argument('--max_chunks', type=int, required=False, help='Maximum number of chunks to process', default=None)
    parser.add_argument('--save_to_dir', type=str, required=False, help='Directory to save the data chunks', default="braintreebank_data_chunks")
    args = parser.parse_args()
    sub_id = args.sub_id
    trial_id = args.trial_id
    laplacian_rereferenced = args.laplacian_rereferenced
    max_chunks = args.max_chunks
    save_to_dir = args.save_to_dir
    assert (not sub_id<0) or (trial_id<0) # if no sub id provided, then process all trials for all subjects

    nperseg = 256
    window_length = nperseg * 8 * 10 * 10 # 100 seconds per chunk, to have fewer files
    process_subject_ids = [sub_id] if sub_id > 0 else range(1, 11)
    for sub_id in process_subject_ids:
        process_trial_ids = [trial_id] if trial_id >= 0 else np.arange([3, 7, 3, 3, 1, 3, 2, 1, 1, 2][sub_id-1]) # if no trial id provided, then process all trials for the subject
        if (sub_id == 6) and (trial_id < 0): process_trial_ids = [0, 1, 4] # special case for subject 6 that only has trials 0, 1, and 4
        for trial_id in process_trial_ids:
            process_subject_trial(sub_id, trial_id, laplacian_rereferenced=laplacian_rereferenced, max_chunks=max_chunks, verbose=True, 
                                  global_per_electrode_normalizing_params=True, save_to_dir=save_to_dir, window_length=window_length, nperseg=nperseg)