import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import glob
import random
from braintreebank_config import *

class BrainTreebankSubjectTrialDataset(Dataset):
    def __init__(self, subject_id, trial_id, dtype, n_time_bins, spectrogram=True, mmap_mode=False):
        """
        Args:
            subject_id (int): Subject ID
            trial_id (int): Trial ID
            dtype (torch.dtype): Data type to load the data in (float32, bfloat16)
            n_time_bins (int): Number of time bins per data item
            spectrogram (bool): Whether to load spectrogram data
        """
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.n_time_bins = n_time_bins
        self.dtype = dtype
        self.spectrogram = spectrogram

        # Load metadata
        metadata_path = (
            f"braintreebank_datachunks{'_raw' if not spectrogram else ''}/"
            f"subject{self.subject_id}_trial{self.trial_id}.json"
        )
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.chunks = []
        for chunk_id in range(self.metadata['n_chunks']):
            chunk_path = (
                f"braintreebank_datachunks{'_raw' if not spectrogram else ''}/"
                f"subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
            )
            if mmap_mode: chunk = np.load(chunk_path, mmap_mode='r')
            else: chunk = np.load(chunk_path) # shape: (n_batch_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
            assert chunk.shape[0] % self.n_time_bins == 0, f"Number of time bins in chunk must be divisible by n_time_bins, but for chunk {chunk_id} of subject {self.subject_id} and trial {self.trial_id} it is {chunk.shape[0]}, not divisible by {self.n_time_bins}"
            self.chunks.append(chunk)
        # Calculate total length
        self.total_samples = chunk.shape[0] * len(self.chunks) // self.n_time_bins
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        chunk_idx = idx * self.n_time_bins // self.chunks[0].shape[0]
        within_chunk_start = idx % (self.chunks[0].shape[0] // self.n_time_bins)
        within_chunk_start *= self.n_time_bins
        within_chunk_end = within_chunk_start + self.n_time_bins
        chunk = self.chunks[chunk_idx][within_chunk_start:within_chunk_end] # shape: (n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
        return torch.from_numpy(chunk.copy()).to(dtype=self.dtype)
    

class BrainTreebankSubjectTrialBenchmarkDatasetBackbone:
    def __init__(self, subject_id, trial_id, dtype, n_time_bins, spectrogram=True, load_nonverbal=True, mmap_mode=False):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.dtype = dtype
        self.n_time_bins = n_time_bins
        self.spectrogram = spectrogram

        file_prefix = (f"braintreebank_datachunks_benchmark{'_raw' if not spectrogram else ''}/"
            f"subject{self.subject_id}_trial{self.trial_id}_"
        )
        words_df_path = file_prefix + "words_df.csv"
        self.all_words_df = pd.read_csv(words_df_path)

        # The benchmark will cut data such that every item starts 1/3 of the way before the word and ends 2/3 of the way after the word
        self.cut_time_bins_from = int(BENCHMARK_START_DATA_BEFORE_ONSET * SAMPLING_RATE // N_PER_SEG - self.n_time_bins//3)
        self.cut_time_bins_to = self.cut_time_bins_from + self.n_time_bins

        self.word_chunk_size = BENCHMARK_CHUNK_SIZE
        self.word_chunks = []
        self.word_chunks_df = []
        self.n_word_chunks = len(self.all_words_df) // self.word_chunk_size
        self.all_words_df = self.all_words_df.iloc[:self.n_word_chunks * self.word_chunk_size] # trim to multiple of chunk size
        for chunk_idx in range(self.n_word_chunks):
            chunk_path = file_prefix + f"chunk{chunk_idx}.npy"
            if mmap_mode: chunk = np.load(chunk_path, mmap_mode='r')
            else: chunk = np.load(chunk_path) # shape: (BENCHMARK_CHUNK_SIZE, n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
            chunk = chunk[:, self.cut_time_bins_from:self.cut_time_bins_to]
            self.word_chunks.append(chunk)
            self.word_chunks_df.append(self.all_words_df.iloc[chunk_idx*BENCHMARK_CHUNK_SIZE:(chunk_idx+1)*BENCHMARK_CHUNK_SIZE])

        if load_nonverbal:
            self.nonverbal_chunks = []
            n_nonverbal_chunk_files = len(glob.glob(file_prefix + "gap_chunk*.npy")) # Count number of gap chunk files
            for chunk_idx in range(n_nonverbal_chunk_files):
                chunk_path = file_prefix + f"gap_chunk{chunk_idx}.npy"
                if mmap_mode: chunk = np.load(chunk_path, mmap_mode='r')
                else: chunk = np.load(chunk_path) # shape: (BENCHMARK_CHUNK_SIZE, n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
                # Since the nonverbal chunks are just the gap between words, we can cut them to the same length as the word chunks
                # XXX: this is a hack, but it works for now -- split the gap chunk into 2, because it is 5 seconds long and we have n_time_bins = 24 (3 seconds)
                #chunk = chunk.view().reshape(-1, self.n_time_bins, chunk.shape[2], chunk.shape[3]) #XXX
                self.nonverbal_chunk_size = chunk.shape[0]
                self.nonverbal_chunks.append(chunk[:, :self.n_time_bins, :, :])
                self.nonverbal_chunks.append(chunk[:, -self.n_time_bins:, :, :])
            self.n_nonverbal_chunks = len(self.nonverbal_chunks)


class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject_id, trial_id, dtype, n_time_bins, eval_name, spectrogram=True, backbone=None):
        """
        Args:
            eval_name (str): can be "pitch" or "rms" (rms for volume) or "onset" or "speech"
        """
        assert eval_name in ["pitch", "rms", "onset", "speech"], f"eval_name must be 'pitch' or 'rms' or 'onset' or 'speech', not {eval_name}"
        if backbone is None: backbone = BrainTreebankSubjectTrialBenchmarkDatasetBackbone(subject_id, trial_id, dtype, n_time_bins, spectrogram=spectrogram, load_nonverbal=eval_name in ["onset", "speech"])
        self.backbone = backbone
        self.eval_name = eval_name

        if eval_name in ["pitch", "rms"]:
            # Get indices for words in top and bottom quartiles
            all_labels = self.backbone.all_words_df[self.eval_name].to_numpy()
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            self.extreme_indices = np.where((label_percentiles > 0.75) | (label_percentiles < 0.25))[0]
            self.extreme_labels = (label_percentiles[self.extreme_indices] > 0.75).astype(int)
            self.n_samples = len(self.extreme_indices)
            self.__getitem__ = self._pitch_rms__getitem__
        else:
            self.positive_indices = np.where(self.backbone.all_words_df["is_onset"].to_numpy() == 1)[0] if eval_name == "onset" else np.arange(len(self.backbone.all_words_df))
            self.negative_indices = np.arange(self.backbone.n_nonverbal_chunks * self.backbone.nonverbal_chunk_size)
            min_len = min(len(self.positive_indices), len(self.negative_indices)) # make sure we have an equal number of positive and negative samples
            self.positive_indices = np.sort(np.random.choice(self.positive_indices, size=min_len, replace=False))
            self.negative_indices = np.sort(np.random.choice(self.negative_indices, size=min_len, replace=False))
            self.n_samples = len(self.positive_indices) + len(self.negative_indices)
            self.__getitem__ = self._onset_speech__getitem__

    def _pitch_rms__getitem__(self, idx):
        word_index = self.extreme_indices[idx]
        chunk_index = word_index // self.backbone.word_chunk_size
        chunk_data = self.backbone.word_chunks[chunk_index]
        input = torch.from_numpy(chunk_data[word_index % self.backbone.word_chunk_size].copy()).to(dtype=self.backbone.dtype)
        return input, self.extreme_labels[idx]

    def _onset_speech__getitem__(self, idx):
        if idx % 2 == 0: # even indices are positive samples
            word_index = self.positive_indices[idx//2]
            chunk_index = word_index // self.backbone.word_chunk_size
            chunk_data = self.backbone.word_chunks[chunk_index]
            input = torch.from_numpy(chunk_data[word_index % self.backbone.word_chunk_size].copy()).to(dtype=self.backbone.dtype)
            return input, 1
        else: # odd indices are negative samples
            item_index = self.negative_indices[idx//2]
            chunk_index = item_index // self.backbone.nonverbal_chunk_size
            chunk_data = self.backbone.nonverbal_chunks[chunk_index]
            input = torch.from_numpy(chunk_data[item_index % self.backbone.nonverbal_chunk_size].copy()).to(dtype=self.backbone.dtype)
            return input, 0
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.eval_name in ["pitch", "rms"]:
            return self._pitch_rms__getitem__(idx)
        else:
            return self._onset_speech__getitem__(idx)
        

class MultiSubjectTrialDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.total_batches = sum(len(dl) for dl in dataloaders.values())
        
    def __iter__(self):
        self.batch_count = 0
        # Create new iterators each time we iterate
        self.iterators = {k: iter(dl) for k, dl in self.dataloaders.items()}
        return self
        
    def __len__(self):
        return self.total_batches
        
    def __next__(self):
        if self.batch_count >= self.total_batches:
            raise StopIteration
            
        # Try dataloaders until we get a valid batch
        while self.iterators:
            subject_trial = random.choice(list(self.iterators.keys()))
            try:
                batch = next(self.iterators[subject_trial])
                self.batch_count += 1
                return subject_trial, batch
            except StopIteration:
                del self.iterators[subject_trial]
                
        raise StopIteration