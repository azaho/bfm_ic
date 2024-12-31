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


# Data frames column IDs
start_col, end_col, lbl_col = 'start', 'end', 'pos'
trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'
word_time_col, word_text_col, is_onset_col, is_offset_col = 'word_time', 'text', 'is_onset', 'is_offset'
def obtain_aligned_words_df(sub_id, trial_id, verbose=True, save_to_dir="braintreebank_benchmark_data_chunks"):
    # Check if aligned words dataframe already exists
    # if save_to_dir is not None:
    #     save_file = os.path.join(save_to_dir, f'subject{sub_id}_trial{trial_id}_words_df.csv')
    #     if os.path.exists(save_file):
    #         if verbose: print(f"Loading existing aligned words dataframe from {save_file}")
    #         return pd.read_csv(save_file)

    # Path to neural data h5 file
    neural_data_file = os.path.join(ROOT_DIR, f'braintreebank/sub_{sub_id}_trial{trial_id:03}.h5')
    # Path to trigger times csv file
    trigger_times_file = os.path.join(ROOT_DIR, f'braintreebank/subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')
    # Path format to trial metadata json file
    metadata_file = os.path.join(ROOT_DIR, f'braintreebank/subject_metadata/sub_{sub_id}_trial{trial_id:03}_metadata.json')
    with open(metadata_file, 'r') as f:
        meta_dict = json.load(f)
        title = meta_dict['title']
        movie_id = meta_dict['filename']
    # # Path to transcript csv file
    transcript_file_format = os.path.join(ROOT_DIR, f'braintreebank/transcripts/{movie_id}/features.csv')
    # Path format to electrode labels file -- mapping each ID to an subject specific label
    electrode_labels_file = os.path.join(ROOT_DIR, f'braintreebank/electrode_labels/sub_{sub_id}/electrode_labels.json')

    if verbose: print(f"Computing words dataframe for subject {sub_id} trial {trial_id}")
    trigs_df = pd.read_csv(trigger_times_file)
    words_df = pd.read_csv(transcript_file_format.format(movie_id)).set_index('Unnamed: 0')
    words_df = words_df.dropna().reset_index(drop=True)  # no need to keep word particles identified by NaN word times
    def estimate_sample_index(t, near_t, near_trig):
        """
        Estimates the word onset data sample by interpolation from nearest trigger.
        
        Inputs:
        t - word movie time
        near_t - nearest trigger movie time
        near_trig - nearest trigger sample index
        
        Output:
        Estimated word onset sample index
        """
        trig_diff = (t - near_t) * SAMPLING_RATE
        return round(near_trig + trig_diff)
    def add_estimated_sample_index(w_df, t_df):
        """
        Computes and adds data sample indices to annotated movie word onsets by interpolation from nearest trigger.
        
        Inputs:
        w_df - movie annotated words data frame
        t_df - computer triggers data frame
        
        Output:
        Movie annotated words data frame augmented with estimated data sample indices
        """
        tmp_w_df = w_df.copy(deep=True)
        last_t = t_df.loc[len(t_df) - 1, trig_time_col]
        for i, t, endt in zip(w_df.index, w_df[start_col], w_df[end_col]):
            if t > last_t:  # if movie continues after triggers
                break
            idx = (abs(t_df[trig_time_col] - t)).idxmin()  # find nearest movie time index
            tmp_w_df.loc[i, :] = w_df.loc[i, :]
            tmp_w_df.loc[i, est_idx_col] = \
                    estimate_sample_index(t, t_df.loc[idx, trig_time_col], t_df.loc[idx, trig_idx_col])

            end_idx = (abs(t_df[trig_time_col] - endt)).idxmin()  # find nearest movie time index
            tmp_w_df.loc[i, est_end_idx_col] = \
                    estimate_sample_index(endt, t_df.loc[end_idx, trig_time_col], t_df.loc[end_idx, trig_idx_col])
        return tmp_w_df
    words_df = add_estimated_sample_index(words_df, trigs_df)  # align all words to data samples
    words_df = words_df.dropna().reset_index(drop=True)  # no need to keep words with no start time
    # Remove words that would create invalid windows (too close to the start or end of the trial)
    total_samples = trigs_df.loc[len(trigs_df) - 1, trig_idx_col]
    valid_words = (words_df[est_idx_col] >= BENCHMARK_START_DATA_BEFORE_ONSET * SAMPLING_RATE) & \
                 (words_df[est_end_idx_col] <= total_samples - BENCHMARK_END_DATA_AFTER_ONSET * SAMPLING_RATE)
    words_df = words_df[valid_words].reset_index(drop=True)

    # add percentile columns
    numerical_cols = ['rms', 'pitch']
    for col in numerical_cols:
        all_values = words_df[col].to_numpy() # shape: (n_words,)
        words_df[f'{col}_percentile'] = words_df.apply(lambda row: np.mean(all_values <= row[col]), axis=1)

    if verbose: print(f"Kept {len(words_df)} words after removing invalid windows")
    if verbose: print(f"Done.")
    # Save the processed words dataframe
    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        words_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_words_df.csv', index=False)
    return words_df

def process_subject_trial(sub_id, trial_id, words_df, laplacian_rereferenced=LAPLACIAN_REREFERENCED, nperseg=256, max_chunks=None, start_data_before_onset=BENCHMARK_START_DATA_BEFORE_ONSET, end_data_after_onset=BENCHMARK_END_DATA_AFTER_ONSET,
                          chunk_batch_size=100, verbose=True, global_per_electrode_normalizing_params=True, allow_corrupted=ALLOW_CORRUPTED_ELECTRODES, 
                          only_laplacian=False, save_to_dir="braintreebank_benchmark_data_chunks", padding_time=BENCHMARK_PADDING_TIME, spectrogram=True):
    window_length = (start_data_before_onset + end_data_after_onset) * SAMPLING_RATE
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

    n_chunks = len(words_df) // chunk_batch_size
    if max_chunks is not None: n_chunks = min(n_chunks, max_chunks)
    if verbose: print(f"Processing subject {sub_id} trial {trial_id} ({len(words_df)} words, {n_chunks} chunks)")
    for chunk_i in range(n_chunks):
        chunk_words_df = words_df[chunk_i*chunk_batch_size:(chunk_i+1)*chunk_batch_size]
        data_chunk = np.zeros((chunk_batch_size, n_electrodes, window_length // nperseg, 37 if spectrogram else nperseg), dtype=np.float32)
        for word_i, row in chunk_words_df.iterrows():
            window_start_sample = int(row[est_idx_col] - start_data_before_onset * SAMPLING_RATE)
            window_end_sample = int(row[est_idx_col] + end_data_after_onset * SAMPLING_RATE)
            assert window_start_sample >= 0 and window_end_sample <= total_samples, f"Window extends beyond data boundaries for word {word_i} of chunk {chunk_i} of subject {sub_id} trial {trial_id}"
            for i, electrode_label in enumerate(electrode_labels):
                if spectrogram:
                    f, t, Sxx = subject.get_spectrogram(electrode_label, trial_id, window_from=window_start_sample, window_to=window_end_sample, 
                                                        normalize_per_freq=True, laplacian_rereferenced=laplacian_rereferenced, cache=True,
                                                        normalizing_params=normalizing_params[electrode_label] if global_per_electrode_normalizing_params else None)
                    data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] = Sxx.T # data_chunk shape: (n_chunks, n_electrodes, n_time_bins, n_freqs)
                else:
                    if laplacian_rereferenced:
                        data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] = subject.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, window_from=window_start_sample, window_to=window_end_sample, cache=True).reshape(-1, nperseg)
                    else:
                        data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] = subject.get_electrode_data(electrode_label, trial_id, window_from=window_start_sample, window_to=window_end_sample, cache=True).reshape(-1, nperseg)
                    if global_per_electrode_normalizing_params:
                        data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] = (data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] - normalizing_params[electrode_label][0].item()) / normalizing_params[electrode_label][1].item()
                    else:
                        data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] = (data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :] - np.mean(data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :]).item()) / np.std(data_chunk[word_i-chunk_i*chunk_batch_size, i, :, :]).item()
        chunk_words_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_chunk{chunk_i}.csv', index=False)
        np.save(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_chunk{chunk_i}.npy', data_chunk)
        if verbose: print(f"Saved chunk {chunk_i} (shape: {data_chunk.shape})")
    subject.close_all_files()

    # Process gaps between words
    if verbose: print("Processing gaps between words...")
    
    gap_data_chunk = np.zeros((chunk_batch_size, n_electrodes, window_length // nperseg, 37 if spectrogram else nperseg), dtype=np.float32)
    gap_chunk_count = 0
    gap_chunk_num = 0

    # TODO: test this
    # Iterate through consecutive word pairs
    for i in range(len(words_df)-1):
        current_word = words_df.iloc[i]
        next_word = words_df.iloc[i+1]
        
        # Calculate gap between current word offset and next word onset
        current_word_offset = int(current_word[est_end_idx_col] + padding_time * SAMPLING_RATE)
        next_word_onset = int(next_word[est_idx_col] - padding_time * SAMPLING_RATE)
        gap_samples = next_word_onset - current_word_offset

        window_start_sample = current_word_offset
        window_end_sample = window_start_sample + window_length

        # If gap is large enough for a window
        while (gap_samples >= window_length) and (window_end_sample <= total_samples):
            # Add to current chunk
            for j, electrode_label in enumerate(electrode_labels):
                if spectrogram:
                    f, t, Sxx = subject.get_spectrogram(electrode_label, trial_id, 
                                                        window_from=window_start_sample,
                                                        window_to=window_end_sample,
                                                        normalize_per_freq=True,
                                                        laplacian_rereferenced=laplacian_rereferenced,
                                                        cache=False,
                                                        normalizing_params=normalizing_params[electrode_label] if global_per_electrode_normalizing_params else None)
                    gap_data_chunk[gap_chunk_count, j, :, :] = Sxx.T
                else:
                    if laplacian_rereferenced:
                        gap_data_chunk[gap_chunk_count, j, :, :] = subject.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, window_from=window_start_sample, window_to=window_end_sample, cache=True).reshape(-1, nperseg)
                    else:
                        gap_data_chunk[gap_chunk_count, j, :, :] = subject.get_electrode_data(electrode_label, trial_id, window_from=window_start_sample, window_to=window_end_sample, cache=True).reshape(-1, nperseg)
                    if global_per_electrode_normalizing_params:
                        gap_data_chunk[gap_chunk_count, j, :, :] = (gap_data_chunk[gap_chunk_count, j, :, :] - normalizing_params[electrode_label][0].item()) / normalizing_params[electrode_label][1].item()
                    else:
                        gap_data_chunk[gap_chunk_count, j, :, :] = (gap_data_chunk[gap_chunk_count, j, :, :] - np.mean(gap_data_chunk[gap_chunk_count, j, :, :]).item()) / np.std(gap_data_chunk[gap_chunk_count, j, :, :]).item()
    
            gap_chunk_count += 1
            window_start_sample += window_length
            window_end_sample += window_length
            gap_samples -= window_length

            # Save chunk if it's full
            if gap_chunk_count == chunk_batch_size:
                if save_to_dir is not None:
                    np.save(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_gap_chunk{gap_chunk_num}.npy', gap_data_chunk)
                    if verbose: print(f"Saved gap chunk {gap_chunk_num} (shape: {gap_data_chunk.shape})")
                # Reset for next chunk
                gap_chunk_num += 1
                gap_chunk_count = 0
                gap_data_chunk = np.zeros((chunk_batch_size, n_electrodes, window_length // nperseg, 37 if spectrogram else nperseg), dtype=np.float32)

    subject.close_all_files()
    del subject

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_id', type=int, required=False, help='Subject ID', default=-1)
    parser.add_argument('--trial_id', type=int, required=False, help='Trial ID', default=-1)
    parser.add_argument('--laplacian_rereferenced', type=bool, required=False, help='Laplacian rereferenced', default=LAPLACIAN_REREFERENCED)
    parser.add_argument('--max_chunks', type=int, required=False, help='Maximum number of chunks to process', default=None)
    parser.add_argument('--save_to_dir', type=str, required=False, help='Directory to save the data chunks', default="braintreebank_benchmark_data_chunks")
    parser.add_argument('--only_update_df', type=bool, required=False, help='Only update the words dataframe without processing neural data', default=False)
    parser.add_argument('--spectrogram', type=int, required=False, help='Use spectrogram', default=0)
    args = parser.parse_args()
    sub_id = args.sub_id
    trial_id = args.trial_id
    laplacian_rereferenced = args.laplacian_rereferenced
    max_chunks = args.max_chunks
    save_to_dir = args.save_to_dir
    only_update_df = args.only_update_df
    spectrogram = args.spectrogram==1
    assert (not sub_id<0) or (trial_id<0) # if no sub id provided, then process all trials for all subjects

    nperseg = 256 # TODO: move to braintreebank_config.py
    process_subject_ids = [sub_id] if sub_id > 0 else range(1, 11)
    for sub_id in process_subject_ids:
        process_trial_ids = [trial_id] if trial_id >= 0 else np.arange([3, 7, 3, 3, 1, 3, 2, 1, 1, 2][sub_id-1]) # if no trial id provided, then process all trials for the subject
        if (sub_id == 6) and (trial_id < 0): process_trial_ids = [0, 1, 4] # special case for subject 6 that only has trials 0, 1, and 4
        for trial_id in process_trial_ids:
            words_df = obtain_aligned_words_df(sub_id, trial_id, save_to_dir=save_to_dir)
            if only_update_df:
                continue
            process_subject_trial(sub_id, trial_id, words_df, laplacian_rereferenced=laplacian_rereferenced, max_chunks=max_chunks, verbose=True, 
                                  global_per_electrode_normalizing_params=True, save_to_dir=save_to_dir, nperseg=nperseg, spectrogram=spectrogram)