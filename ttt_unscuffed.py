import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, json, glob, argparse, time, sklearn
from braintreebank_config import *
from transformer_architecture_juice import *
from braintreebank_subject import Subject
from muon import Muon
import wandb

if __name__ == "__main__":
    args = argparse.Namespace()
    args.rs = "WC"  # Default random string value
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs', type=str, default=args.rs, help='Random string')
    parser.add_argument('--embedding_dim', type=int, default=192, help='Embedding dimension')
    args = parser.parse_args()
    random_string = args.rs
    embedding_dim = args.embedding_dim
else:
    random_string = "WC"
    embedding_dim = 192

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_config = {
    'n_epochs': 300,
    'save_model_every_n_epochs': 10,
    'save_metadata_every_n_epochs': 10, #TODO: make it actually save the metadata
    'mmap_mode': False, # TODO: add option to keep on gpu

    'wandb_project': "ne_bf2",
    'wandb_commit_every_n_epochs': 1,

    'train_subject_trials': [(3, 1), (3, 2)],
    'eval_subject_trials': [(3, 0)],
    'batch_size': 100, 
    'p_test_chunks': 0.15,

    'lr_max': 0.0015,
    'lr_min': 0.0015,
    'lr_warmup_steps': 0,
    'weight_decay': 0.0,

    'random_string': random_string,
}
transformer_config = {
    'model_name': "wc_bug5",
    'max_n_electrodes': 124,
    'dim_input': 64,
    'max_n_time_bins': 24,
    'd_model': 192,
    'n_heads': 12,
    'n_layers': 10,
    'dropout': 0.2,
    'dtype': torch.bfloat16,
    'device': device,
    'optimizer': 'Muon',
    'electrode_embedding': 'zeros',
    'electrode_embedding_grad': True,
    'spectrogram': True,
    'embedding_dim': embedding_dim,
}
transformer_config['n_layers_electrode'] = transformer_config['n_layers']//2
transformer_config['n_layers_time'] = transformer_config['n_layers'] - transformer_config['n_layers_electrode']
transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']
transformer_config['dim_output'] = transformer_config['d_model']

# Set random seed for reproducibility
if not training_config.get('random_string'):
    training_config['random_string'] = f"{time.time():.5f}"[-4:]
random_seed = hash(training_config['random_string']) % (2**32)
training_config['random_seed'] = random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed) 

def update_dir_name():
    dir_name = f"training_results/{transformer_config['model_name']}"
    if not transformer_config['spectrogram']:
        dir_name += f"_ns"
    dir_name += f"_di{transformer_config['dim_input']}"
    #dir_name += f"_s{args.subjects}"
    dir_name += f"_t{transformer_config['max_n_time_bins']}"
    dir_name += f"_dm{transformer_config['d_model']}"
    dir_name += f"_ed{transformer_config['embedding_dim']}"
    dir_name += f"_nh{transformer_config['n_heads']}"
    dir_name += f"_nl{transformer_config['n_layers']}"
    dir_name += f"_dr{transformer_config['dropout']}"
    dir_name += f"_opt{transformer_config['optimizer']}"
    dir_name += f"_ei{transformer_config['electrode_embedding'][0].upper()}"
    dir_name += f"_bs{training_config['batch_size']}"
    dir_name += f"_wd{training_config['weight_decay']}"
    dir_name += f"_lr{training_config['lr_max']}"
    dir_name += f"_lrm{training_config['lr_min']}"
    #dir_name += f"_lrwm{training_config['lr_warmup_steps']}" if 'lr_warmup_steps' in training_config else f"_lrwf{training_config['lr_warmup_frac']}"
    dir_name += f"_r{training_config['random_string']}"
    return dir_name
dir_name = update_dir_name()
training_config['dir_name'] = dir_name

class BrainTreebankSubjectTrialDataset(Dataset):
    def __init__(self, subject_id, trial_id, dtype, n_time_bins, spectrogram=True):
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
            if training_config['mmap_mode']: chunk = np.load(chunk_path, mmap_mode='r')
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
    def __init__(self, subject_id, trial_id, dtype, n_time_bins, spectrogram=True, load_nonverbal=True):
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
            if training_config['mmap_mode']: chunk = np.load(chunk_path, mmap_mode='r')
            else: chunk = np.load(chunk_path) # shape: (BENCHMARK_CHUNK_SIZE, n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
            chunk = chunk[:, self.cut_time_bins_from:self.cut_time_bins_to]
            self.word_chunks.append(chunk)
            self.word_chunks_df.append(self.all_words_df.iloc[chunk_idx*BENCHMARK_CHUNK_SIZE:(chunk_idx+1)*BENCHMARK_CHUNK_SIZE])

        if load_nonverbal:
            self.nonverbal_chunks = []
            n_nonverbal_chunk_files = len(glob.glob(file_prefix + "gap_chunk*.npy")) # Count number of gap chunk files
            for chunk_idx in range(n_nonverbal_chunk_files):
                chunk_path = file_prefix + f"gap_chunk{chunk_idx}.npy"
                if training_config['mmap_mode']: chunk = np.load(chunk_path, mmap_mode='r')
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
        

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.electrode_transformer = ElectrodeTransformer(config=self.config)
        self.time_transformer = TimeTransformer(config=self.config)
        self.temperature_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, electrode_embedding):
        # x shape: (batch_size, n_time_bins, n_electrodes, dim_input)
        # electrode_embedding shape: (n_electrodes, d_model)

        batch_size, n_time_bins, n_electrodes, dim_input = x.shape
        x = x.unsqueeze(1) # add n_samples dimension
        x = self.electrode_transformer(x, electrode_embedding)
        electrode_output = x[:, :, :, 0:1, :] # just the CLS token
        if "1" in training_config['random_string']: 
            electrode_output = electrode_output.transpose(2, 3) #XXX
        time_output = self.time_transformer(electrode_output)

        # squeeze unnecessary empty dimensions
        electrode_output = electrode_output.view(batch_size, n_time_bins, self.config['dim_output']) 
        time_output = time_output.view(batch_size, n_time_bins, self.config['dim_output']) 
        return time_output, electrode_output

if __name__ == "__main__":
    batch_size = training_config['batch_size']
    dtype = transformer_config['dtype']
    subject_ids = set([subject_id for subject_id, trial_id in training_config['train_subject_trials']+training_config['eval_subject_trials']])
    subjects = {subject_id: Subject(subject_id) for subject_id in subject_ids}

    model = Model(transformer_config).to(device, dtype=dtype)
    electrode_embeddings = ElectrodeEmbeddings(transformer_config, subjects, embedding_dim=transformer_config['embedding_dim']).to(device, dtype=dtype)
    
    train_dataloaders = {}
    test_dataloaders = {}
    for subject_id, trial_id in training_config['train_subject_trials']:
        full_dataset = BrainTreebankSubjectTrialDataset(subject_id=subject_id, trial_id=trial_id, dtype=dtype,
                                                        n_time_bins=transformer_config['max_n_time_bins'], spectrogram=transformer_config['spectrogram'])
        train_size = int((1 - training_config['p_test_chunks']) * len(full_dataset))
        indices = torch.randperm(len(full_dataset))
        train_subset = torch.utils.data.Subset(full_dataset, indices[:train_size])
        test_subset = torch.utils.data.Subset(full_dataset, indices[train_size:])
        train_dataloaders[(subject_id, trial_id)] = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        test_dataloaders[(subject_id, trial_id)] = DataLoader(test_subset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        print("Loaded train and test dataloaders for subject", subject_id, "trial", trial_id)

    _backbone = BrainTreebankSubjectTrialBenchmarkDatasetBackbone(subject_id, trial_id, dtype, 
                                                                   transformer_config['max_n_time_bins'], spectrogram=transformer_config['spectrogram'])
    eval_datasets = [
        BrainTreebankSubjectTrialBenchmarkDataset(
            subject_id=subject_id, trial_id=trial_id, dtype=dtype,
            n_time_bins=transformer_config['max_n_time_bins'], spectrogram=transformer_config['spectrogram'],
            eval_name=eval_name, backbone=_backbone
        ) for subject_id, trial_id in training_config['eval_subject_trials'] for eval_name in ["rms", "pitch", "onset", "speech"]
    ]
    eval_dataloaders = [DataLoader(eval_datasets[i], batch_size=100, shuffle=False, pin_memory=True) for i in range(len(eval_datasets))]
    print("Loaded eval dataloaders")

    def eval_model(model, dataloader, test_size=0.2):
        subject_id = dataloader.dataset.backbone.subject_id
        eo_store, to_store, label_store = [], [], []
        for batch_input, batch_label in dataloader:
            batch_size, n_time_bins, n_electrodes, dim_input = batch_input.shape
            permutation = torch.randperm(n_electrodes) #XXX
            batch_input = batch_input[:, :, permutation, :transformer_config['dim_input']].to(device, dtype=transformer_config['dtype'])
            batch_embeddings = electrode_embeddings(subject_id, permutation=permutation)
            eo, to = model(batch_input, batch_embeddings) # shape: (batch_size, n_time_bins, dim_output)
            eo_store.append(eo.mean(dim=1).detach().cpu().float().numpy())
            to_store.append(to.mean(dim=1).detach().cpu().float().numpy())
            label_store.append(batch_label.detach().cpu().numpy())
        eo_store = np.concatenate(eo_store)
        to_store = np.concatenate(to_store)
        label_store = np.concatenate(label_store)

        eval_results = {}
        train_size = int(len(eo_store) * (1-test_size))
        for feature_name, feature_store in [("electrode", eo_store), ("time", to_store)]:
            regressor = sklearn.linear_model.LogisticRegression(max_iter=10000)
            regressor.fit(feature_store[:train_size], label_store[:train_size])
            regressor_pred = regressor.predict_proba(feature_store[train_size:])[:, 1]
            regressor_pred_class = regressor.predict(feature_store[train_size:])
            eval_results[f"eval/{subject_id}_{dataloader.dataset.eval_name}_{feature_name}_roc"] = sklearn.metrics.roc_auc_score(label_store[train_size:], regressor_pred)
            eval_results[f"eval/{subject_id}_{dataloader.dataset.eval_name}_{feature_name}_acc"] = sklearn.metrics.accuracy_score(label_store[train_size:], regressor_pred_class)
        return eval_results
    
    expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(transformer_config['max_n_time_bins']-1, 1).to(device, dtype=torch.long).reshape(-1)
    def calculate_loss(model, batch, subject_id):
        n_electrodes = subjects[subject_id].get_n_electrodes()
        permutation = torch.randperm(n_electrodes)

        # batch shape: (batch_size, n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
        batch = batch[:, :, permutation, :transformer_config['dim_input']].to(device, dtype=transformer_config['dtype'])
        batch_embeddings = electrode_embeddings(subject_id, permutation)

        # all model outputs shape: (batch_size, n_time_bins, dim_output)
        eo1, to1 = model(batch[:, :-1, :n_electrodes//2, :], batch_embeddings[:n_electrodes//2])
        eo2, to2 = model(batch[:, 1:, n_electrodes//2:, :], batch_embeddings[n_electrodes//2:])

        similarity = torch.matmul(to1[:, :].transpose(0, 1), eo2[:, :].permute(1, 2, 0)) * torch.exp(model.temperature_param)
        return nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
    def test_model(model):
        loss = 0
        n_batches = 0
        for subject_trial, dataloader in test_dataloaders.items():
            subject_id = subject_trial[0]
            for batch_i, batch in enumerate(dataloader):
                loss += calculate_loss(model, batch, subject_id)
            n_batches += len(dataloader)
        return loss / n_batches
    def save_model_and_embeddings(epoch_i):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), f"{dir_name}/model_e{epoch_i+1}.pth")
        torch.save(electrode_embeddings.state_dict(), f"{dir_name}/model_electrode_embeddings_e{epoch_i+1}.pth")

    all_params = list(model.parameters()) + list(electrode_embeddings.parameters())
    optimizers = []
    if transformer_config['optimizer'] == 'Muon':
        matrix_params = [p for p in all_params if p.ndim >= 2]
        other_params = [p for p in all_params if p.ndim < 2]
        optimizers.append(Muon(matrix_params, lr=training_config['lr_max'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5))
        optimizers.append(torch.optim.Adam(other_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay']))
    else:
        optimizers = [torch.optim.Adam(all_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay'])]

    if training_config['wandb_project']:
        wandb.init(project=training_config['wandb_project'], name=dir_name.split('/')[-1], id=dir_name.split('/')[-1],
                   config={"training_config": training_config, "transformer_config": transformer_config,}, settings=wandb.Settings(init_timeout=480))

    # with torch.no_grad():
    #     for dataloader_i, dataloader in enumerate(eval_dataloaders):
    #         if dataloader_i <2 : continue
    #         print(f"EVAL {dataloader_i+1}/{len(eval_dataloaders)}")
    #         eval_model(model, dataloader)
    save_model_and_embeddings(0)
    for epoch_i in range(training_config['n_epochs']):
        print(f"EPOCH {epoch_i+1}/{training_config['n_epochs']}")
        model.train()
        epoch_loss = 0
        epoch_avg_distance = 0

        # Create iterators for each dataloader
        iterators = {subject_trial: iter(dataloader) for subject_trial, dataloader in train_dataloaders.items()}
        active_dataloaders = list(train_dataloaders.keys())
        total_batches = sum(len(dl) for dl in train_dataloaders.values())
        batch_i = 0

        while active_dataloaders:
            # Randomly select a dataloader that still has batches
            subject_trial = active_dataloaders[np.random.randint(len(active_dataloaders))]
            dataloader = train_dataloaders[subject_trial]
            
            try: batch = next(iterators[subject_trial])
            except StopIteration: 
                active_dataloaders.remove(subject_trial) # Remove this dataloader from active set when depleted
                continue
            batch_i += 1

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss = calculate_loss(model, batch, subject_trial[0])
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()


            # XXX : getting avg distance between any two vectors in last dimension
            with torch.no_grad():
                n_electrodes = subjects[subject_id].get_n_electrodes()
                permutation = torch.randperm(n_electrodes)
                # batch shape: (batch_size, n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
                batch = batch[:, :, permutation, :transformer_config['dim_input']].to(device, dtype=transformer_config['dtype'], non_blocking=True)
                batch_embeddings = electrode_embeddings(subject_id, permutation)
                # all shape: (batch_size, n_time_bins, dim_output)
                eo1, to1 = model(batch[:, :, :n_electrodes//2, :], batch_embeddings[:n_electrodes//2])
                eo2, to2 = model(batch[:, :, n_electrodes//2:, :], batch_embeddings[n_electrodes//2:])
                diff = to1.unsqueeze(1) - to1.unsqueeze(2) # shape: (batch_size*1*1, n_time_bins, n_time_bins, d_model)
                distances = torch.norm(diff, dim=-1) # shape: (batch_size*1*1, n_time_bins, n_time_bins)
                mask = ~torch.eye(distances.shape[1], dtype=torch.bool, device=distances.device)
                avg_distance = distances[:, mask].mean().item()
                epoch_avg_distance += avg_distance

            epoch_loss += loss.item()
            gpu_mem_used = torch.cuda.max_memory_allocated() // 1024**2 / 1024  # Convert to GB
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for next batch
            print(f"epoch {epoch_i+1}/{training_config['n_epochs']}\tdataloader {subject_trial}\tbatch {batch_i}/{total_batches}\tloss {loss.item():.4f}\tgpu {gpu_mem_used:.1f}G\ttemp {model.temperature_param.item():.4f}\tavg_distance {avg_distance:.4f}")

        eval_results = {}
        with torch.no_grad():
            test_loss = test_model(model) # test loss calculation is not in eval mode, to be comparable to train loss
            model.eval()
            for dataloader in eval_dataloaders:
                eval_results.update(eval_model(model, dataloader))
            eval_results['test_loss'] = test_loss.item()
            eval_results['train_loss'] = epoch_loss / total_batches
            eval_results['avg_distance'] = epoch_avg_distance / total_batches #XXX

            gpu_mem_used = torch.cuda.max_memory_allocated() // 1024**2 / 1024  # Convert to GB
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for next batch
            print(f"EVAL RESULTS (gpu {gpu_mem_used:.1f}G): ", json.dumps(eval_results, indent=4))

        if ((epoch_i+1)%training_config['save_model_every_n_epochs']==0) or (epoch_i==training_config['n_epochs']-1):
            save_model_and_embeddings(epoch_i+1)

        if training_config['wandb_project']:
            wandb.log(eval_results, step=(epoch_i+1)*total_batches, commit=(epoch_i+1)%training_config['wandb_commit_every_n_epochs']==0)
    if training_config['wandb_project']:
        wandb.finish()