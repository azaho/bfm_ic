import h5py
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy import signal, stats
import numpy as np
from braintreebank_config import *

class Subject:
    """ 
        This class is used to load the neural data for a given subject and trial.
        It also contains methods to get the data for a given electrode and trial, and to get the spectrogram for a given electrode and trial.
    """
    def __init__(self, subject_id, sampling_rate=SAMPLING_RATE, allow_corrupted=ALLOW_CORRUPTED_ELECTRODES):
        self.subject_id = subject_id
        self.sampling_rate = sampling_rate
        self.localization_data = self._load_localization_data()
        self.neural_data = {}
        self.neural_data_cache = {}
        self.h5f_files = {}
        self.electrode_labels = self._get_all_electrode_names()
        self.corrupted_electrodes = self._get_corrupted_electrodes()
        if not allow_corrupted:
            self.electrode_labels = [e for e in self.electrode_labels if e not in self.corrupted_electrodes]
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
        return electrode_label.replace('*', '').replace('#', '')

    def _get_corrupted_electrodes(self):
        corrupted_electrodes_file = os.path.join(ROOT_DIR, f'braintreebank_corrupted_elec.json')
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[f'subject{self.subject_id}']]
        # add electrodes that start with "DC" to corrupted electrodes, because they don't have brain signal, instead are used for triggers
        corrupted_electrodes += [e for e in self.electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))] 
        return corrupted_electrodes

    def _get_all_electrode_names(self):
        electrode_labels_file = os.path.join(ROOT_DIR, f'braintreebank/electrode_labels/sub_{self.subject_id}/electrode_labels.json')
        electrode_labels = json.load(open(electrode_labels_file))
        electrode_labels = [self._clean_electrode_label(e) for e in electrode_labels]
        return electrode_labels

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
        neural_data_file = os.path.join(ROOT_DIR, f'braintreebank/sub_{self.subject_id}_trial{trial_id:03}.h5')
        h5f = h5py.File(neural_data_file, 'r', locking=False)
        self.h5f_files[trial_id] = h5f
        self.neural_data[trial_id] = h5f['data']
        self.neural_data_cache[trial_id] = {}

    def _load_localization_data(self):
        """Load localization data for this electrode's subject from depth-wm.csv"""
        loc_file = os.path.join(ROOT_DIR, f'braintreebank/localization/sub_{self.subject_id}/depth-wm.csv')
        df = pd.read_csv(loc_file)
        df['Electrode'] = df['Electrode'].apply(self._clean_electrode_label)
        return df
    
    def get_electrode_coordinates(self):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) array of coordinates (L, I, P) without any preprocessing of the coordinates
                All coordinates are in between 50mm and 200mm for this dataset (check braintreebank_utils.ipynb for statistics)
        """
        # Load the brain regions file for this subject
        regions_df = self.localization_data
        # Create array of coordinates in same order as electrode_labels
        coordinates = np.zeros((len(self.electrode_labels), 3))
        for i, label in enumerate(self.electrode_labels):
            assert label in regions_df['Electrode'].values, f"Electrode {label} not found in regions file of subject {self.subject_id}"
            row = regions_df[regions_df['Electrode'] == label].iloc[0]
            coordinates[i] = [row['L'], row['I'], row['P']]
        return coordinates
    
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
                        normalize_per_freq=False, cache=True, nperseg=256, noverlap=0, power_smoothing_factor=1e-5,
                        min_freq=8, max_freq=300):
        if laplacian_rereferenced: 
            data = self.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, cache=cache, window_from=window_from, window_to=window_to)
        else: data = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to, cache=cache)

        f, t, Sxx = signal.spectrogram(data, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap, window='boxcar')
        mask = np.ones(f.shape, dtype=bool)
        if min_freq is not None:
            mask = (f>=min_freq)
        if max_freq is not None:
            mask = (f<=max_freq) & mask
        f, Sxx = f[mask], Sxx[mask] # only keep frequencies up to some frequency
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
    def get_electrode_data_normalizing_params(self, electrode_label, trial_id, laplacian_rereferenced=False, cache=True):
        if laplacian_rereferenced:
            data = self.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, cache=cache)
        else:
            data = self.get_electrode_data(electrode_label, trial_id, cache=cache)
        return np.mean(data, axis=0), np.std(data, axis=0)
    def close_all_files(self):
        for h5f in self.h5f_files.values():
            h5f.close()
        self.clear_neural_data_cache()

if __name__ == "__main__":
    # all subject trials including the special case for subject 6 which only has trials 0, 1, and 4
    all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

    #####

    # Dictionary to store MSE values for each subject
    subject_mse_dict = {}

    for sub_id, trial_id in all_subject_trials:
        persistence_mse_array = []
        for chunk_id in range(10):
            chunk_path = f"braintreebank_data_chunks/subject{sub_id}_trial{trial_id}_chunk{chunk_id}.npy"
            chunk_data = np.load(chunk_path) # (n_electrodes, n_time_bins, n_freq_features)
            persistence_mse = np.mean((chunk_data[:, 1:, :] - chunk_data[:, :-1, :])**2)
            persistence_mse_array.append(persistence_mse)
        trial_mean = np.mean(persistence_mse_array)
        print(f"Subject {sub_id} trial {trial_id} mean persistence MSE: {trial_mean}")
        
        # Add to subject dictionary
        if sub_id not in subject_mse_dict:
            subject_mse_dict[sub_id] = []
        subject_mse_dict[sub_id].append(trial_mean)
    # Print mean for each subject
    print("\nMean persistence MSE per subject:")
    all_means = []
    for sub_id in sorted(subject_mse_dict.keys()):
        subject_mean = np.mean(subject_mse_dict[sub_id])
        print(f"Subject {sub_id}: {subject_mean}")
        all_means.append(subject_mean)
    # Print overall mean
    print(f"\nOverall mean persistence MSE: {np.mean(all_means)}")

    # Dictionary to store mean MSE values for each subject
    subject_mean_mse_dict = {}

    for sub_id, trial_id in all_subject_trials:
        mean_mse_array = []
        for chunk_id in range(10):
            chunk_path = f"braintreebank_data_chunks/subject{sub_id}_trial{trial_id}_chunk{chunk_id}.npy"
            chunk_data = np.load(chunk_path) # (n_electrodes, n_time_bins, n_freq_features)
            
            # Calculate mean per electrode across time
            electrode_means = np.mean(chunk_data, axis=1, keepdims=True)  # (n_electrodes, 1, n_freq_features)
            
            # Calculate MSE between actual values and mean prediction
            mean_mse = np.mean((chunk_data - electrode_means)**2)
            mean_mse_array.append(mean_mse)
            
        trial_mean = np.mean(mean_mse_array)
        print(f"Subject {sub_id} trial {trial_id} mean MSE using electrode means: {trial_mean}")
        
        # Add to subject dictionary
        if sub_id not in subject_mean_mse_dict:
            subject_mean_mse_dict[sub_id] = []
        subject_mean_mse_dict[sub_id].append(trial_mean)

    # Print mean for each subject
    print("\nMean MSE per subject (using electrode means):")
    all_means = []
    for sub_id in sorted(subject_mean_mse_dict.keys()):
        subject_mean = np.mean(subject_mean_mse_dict[sub_id])
        print(f"Subject {sub_id}: {subject_mean}")
        all_means.append(subject_mean)

    # Print overall mean
    print(f"\nOverall mean MSE using electrode means: {np.mean(all_means)}")

    # Now calculate MSE using global mean across all electrodes
    subject_global_mean_mse_dict = {}

    for sub_id, trial_id in all_subject_trials:
        global_mean_mse_array = []
        for chunk_id in range(10):
            chunk_path = f"braintreebank_data_chunks/subject{sub_id}_trial{trial_id}_chunk{chunk_id}.npy"
            chunk_data = np.load(chunk_path)
            
            # Calculate global mean across all electrodes and time
            global_mean = np.mean(chunk_data, axis=(0,1), keepdims=True)  # (1, 1, n_freq_features)
            
            # Calculate MSE between actual values and global mean prediction
            global_mean_mse = np.mean((chunk_data - global_mean)**2)
            global_mean_mse_array.append(global_mean_mse)
            
        trial_mean = np.mean(global_mean_mse_array)
        print(f"Subject {sub_id} trial {trial_id} mean MSE using global mean: {trial_mean}")
        
        if sub_id not in subject_global_mean_mse_dict:
            subject_global_mean_mse_dict[sub_id] = []
        subject_global_mean_mse_dict[sub_id].append(trial_mean)

    # Print mean for each subject
    print("\nMean MSE per subject (using global mean):")
    all_global_means = []
    for sub_id in sorted(subject_global_mean_mse_dict.keys()):
        subject_mean = np.mean(subject_global_mean_mse_dict[sub_id])
        print(f"Subject {sub_id}: {subject_mean}")
        all_global_means.append(subject_mean)

    # Print overall mean
    print(f"\nOverall mean MSE using global mean: {np.mean(all_global_means)}")


    #####

    from braintreebank_process_chunks import *

    for sub_id, trial_id in all_subject_trials:
        print(f"Processing subject {sub_id} trial {trial_id}")
        subject = Subject(sub_id)
        subject.load_neural_data(trial_id)
        subject.check_electrodes(trial_id)
        subject.close_all_files()
        del subject