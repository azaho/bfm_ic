import torch, numpy as np, sklearn.linear_model, sklearn.metrics, os, argparse
from braintreebank_dataloaders import BrainTreebankSubjectTrialDataset, BrainTreebankSubjectTrialBenchmarkDataset, BrainTreebankSubjectTrialBenchmarkDatasetBackbone, MultiSubjectTrialDataLoader
from torch.utils.data import DataLoader, Subset
from training_muon import Muon

# Default parameters
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

    'random_string': "X",
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
}
transformer_config['embedding_dim'] = transformer_config['d_model']
transformer_config['n_layers_electrode'] = transformer_config['n_layers']//2
transformer_config['n_layers_time'] = transformer_config['n_layers'] - transformer_config['n_layers_electrode']
transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']
transformer_config['dim_output'] = transformer_config['d_model']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs', type=str, help='Random string')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    args = parser.parse_args()
    
    if args.rs is not None: training_config['random_string'] = args.rs
    if args.embedding_dim is not None: transformer_config['embedding_dim'] = args.embedding_dim


def update_random_seed(training_config):
    random_seed = hash(training_config['random_string']) % (2**32)
    training_config['random_seed'] = random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    return random_seed
def update_dir_name(transformer_config, training_config):
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
    training_config['dir_name'] = dir_name
    return dir_name


def create_traintest_dataloaders(training_config, transformer_config, subject_trials=None, verbose=False):
    if subject_trials is None:
        subject_trials = training_config['train_subject_trials']
    train_dataloaders = {}
    test_dataloaders = {}
    for subject_id, trial_id in subject_trials:
        full_dataset = BrainTreebankSubjectTrialDataset(subject_id=subject_id, trial_id=trial_id, dtype=transformer_config['dtype'],
                                                        n_time_bins=transformer_config['max_n_time_bins'], spectrogram=transformer_config['spectrogram'],
                                                        mmap_mode=training_config['mmap_mode'])
        train_size = int((1 - training_config['p_test_chunks']) * len(full_dataset))
        indices = torch.randperm(len(full_dataset))
        train_subset = torch.utils.data.Subset(full_dataset, indices[:train_size])
        test_subset = torch.utils.data.Subset(full_dataset, indices[train_size:])
        train_dataloaders[(subject_id, trial_id)] = DataLoader(train_subset, batch_size=training_config['batch_size'], shuffle=True, drop_last=True, pin_memory=True)
        test_dataloaders[(subject_id, trial_id)] = DataLoader(test_subset, batch_size=training_config['batch_size'], shuffle=True, drop_last=True, pin_memory=True)
        if verbose: print("Loaded train and test dataloaders for subject", subject_id, "trial", trial_id)
    train_dataloader = MultiSubjectTrialDataLoader(train_dataloaders)
    test_dataloader = MultiSubjectTrialDataLoader(test_dataloaders)
    return train_dataloader, test_dataloader
def create_eval_dataloaders(training_config, transformer_config, subject_trials=None, verbose=False):
    if subject_trials is None: subject_trials = training_config['eval_subject_trials']
        
    backbones = [
        BrainTreebankSubjectTrialBenchmarkDatasetBackbone(
            subject_id, trial_id, transformer_config['dtype'], 
            transformer_config['max_n_time_bins'], spectrogram=transformer_config['spectrogram'],
            mmap_mode=training_config['mmap_mode']
        ) for subject_id, trial_id in subject_trials
    ]
    eval_datasets = [
        BrainTreebankSubjectTrialBenchmarkDataset(
            subject_id=subject_id, trial_id=trial_id, dtype=transformer_config['dtype'],
            n_time_bins=transformer_config['max_n_time_bins'], spectrogram=transformer_config['spectrogram'],
            eval_name=eval_name, backbone=backbones[i]
        ) for i, (subject_id, trial_id) in enumerate(subject_trials) for eval_name in ["rms", "pitch", "onset", "speech"]
    ]
    eval_dataloaders = [DataLoader(eval_datasets[i], batch_size=training_config['batch_size'], shuffle=False, pin_memory=True) for i in range(len(eval_datasets))]
    if verbose: print("Loaded eval dataloaders")
    return eval_dataloaders


def create_optimizers(model, electrode_embeddings):
    all_params = list(model.parameters()) + list(electrode_embeddings.parameters())
    optimizers = []
    if transformer_config['optimizer'] == 'Muon':
        matrix_params = [p for p in all_params if p.ndim >= 2]
        other_params = [p for p in all_params if p.ndim < 2]
        optimizers.append(Muon(matrix_params, lr=training_config['lr_max'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5))
        optimizers.append(torch.optim.Adam(other_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay']))
    else:
        optimizers = [torch.optim.Adam(all_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay'])]
    return optimizers


def eval_model(model, electrode_embeddings, dataloader, test_size=0.2):
    subject_id = dataloader.dataset.backbone.subject_id
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eo_store, to_store, label_store = [], [], []
    for batch_input, batch_label in dataloader:
        batch_size, n_time_bins, n_electrodes, dim_input = batch_input.shape
        batch_input = batch_input[:, :, :, :dim_input].to(device, dtype=dtype)
        batch_embeddings = electrode_embeddings(subject_id)
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


def test_model(model, electrode_embeddings, loss_function, test_dataloader):
    loss = 0
    n_batches = 0
    for batch_i, (subject_trial, batch) in enumerate(test_dataloader):
        subject_id = subject_trial[0]
        loss += loss_function(model, electrode_embeddings, batch, subject_id)
        n_batches += 1
    return loss / n_batches
def save_model_and_embeddings(model, electrode_embeddings, epoch_i):
    dir_name = training_config['dir_name'] if 'dir_name' in training_config else update_dir_name(transformer_config, training_config)
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), f"{dir_name}/model_e{epoch_i+1}.pth")
    torch.save(electrode_embeddings.state_dict(), f"{dir_name}/model_electrode_embeddings_e{epoch_i+1}.pth")