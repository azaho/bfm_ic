import torch, numpy as np, json, os, time
from transformer_architecture_cpc import ElectrodeTransformer, TimeTransformer
import argparse
import wandb
import pandas as pd
from scipy import stats
import scipy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
#all_subject_trials = [(2, 4)] #XXX

args = argparse.Namespace()
args.lrmax = 0.001
args.lrmin = 0.001
args.bs = 100
args.nl = 15
args.dm = 384
args.mt = 'mask-out-none'
args.dtype = 'bfloat16'
args.nh = 12
args.dr = 0.0
args.rs = "" 
args.lrwm = 0
args.wait_n_intervals = 0
args.weight_decay = 0.000
args.optimizer = 'Muon'
args.max_gradient_norm = -1
args.electrode_embedding_init = 'normal'
args.wandb_project = ""
args.subjects = "1"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrmax', type=float, default=args.lrmax, help='Maximum learning rate')
    parser.add_argument('--lrmin', type=float, default=args.lrmin, help='Minimum learning rate') 
    parser.add_argument('--bs', type=int, default=args.bs, help='Batch size')
    parser.add_argument('--nl', type=int, default=args.nl, help='Number of transformer layers')
    parser.add_argument('--dm', type=int, default=args.dm, help='Model dimension')
    parser.add_argument('--mt', type=str, default=args.mt, help='Mask type')
    parser.add_argument('--dtype', type=str, default=args.dtype, choices=['bfloat16', 'float32'], help='Data type')
    parser.add_argument('--nh', type=int, default=args.nh, help='Number of attention heads')
    parser.add_argument('--dr', type=float, default=args.dr, help='Dropout rate')
    parser.add_argument('--rs', type=str, default=args.rs, help='Random string') 
    parser.add_argument('--lrwm', type=int, default=args.lrwm, help='Learning rate warmup steps') 
    parser.add_argument('--wait_n_intervals', type=int, default=args.wait_n_intervals, help='Wait n intervals (for many jobs)')
    parser.add_argument('--weight_decay', type=float, default=args.weight_decay, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default=args.optimizer, choices=['AdamW', 'Muon'], help='Optimizer type') # TODO: add Muon
    parser.add_argument('--max_gradient_norm', type=float, default=args.max_gradient_norm, help='Maximum gradient norm (-1 for no clipping)')
    parser.add_argument('--electrode_embedding_init', type=str, default=args.electrode_embedding_init, choices=['normal', 'zeros'], help='Electrode embedding initialization')
    parser.add_argument('--wandb_project', type=str, default=args.wandb_project, help='Weights & Biases project name')
    parser.add_argument('--subjects', type=str, default=args.subjects, help='Subject numbers (digits only)')
    args = parser.parse_args()
    assert args.lrmax >= args.lrmin, "Maximum learning rate must be greater than or equal to minimum learning rate"
    assert args.subjects.isdigit() or args.subjects == "", "Subjects parameter must contain only numbers and commas"
    assert len(args.subjects) > 0, "Subjects parameter must contain at least one subject"
    if args.wait_n_intervals > 0:
        print(f"Waiting {args.wait_n_intervals} intervals")
        for i in range(args.wait_n_intervals):
            print(f"Waiting {i+1} of {args.wait_n_intervals}")
            time.sleep(1)

train_subject_trials = []
for subject in args.subjects:
    if subject == '0': subject = 10
    else: subject = int(subject)
    train_subject_trials.extend((subject_id, trial_id) for subject_id, trial_id in all_subject_trials if subject_id == subject)

training_config = {
    'n_epochs': 200,
    'save_network_every_n_epochs': 20,
    'save_losses_every_n_batches': 20,
    'save_test_losses_every_n_batches': 1, # XXX
    'save_eval_every_n_batches': 1,
    'p_test_chunks': 0.1,

    'batch_size': args.bs,
    'train_subject_trials': train_subject_trials, #[(2, 4)], #[(2, 4), (1, 1), (3, 1)],
    'lr_max': args.lrmax,
    'lr_min': args.lrmin,
    #'lr_warmup_frac': 0.01, # need to specify either warmup frac or steps
    'lr_warmup_steps': args.lrwm,
    'weight_decay': args.weight_decay,
    'random_string': args.rs,
    'max_gradient_norm': args.max_gradient_norm,
    'wandb_project': args.wandb_project,
}
assert ('lr_warmup_frac' in training_config) != ('lr_warmup_steps' in training_config), "Need to specify either lr_warmup_frac or lr_warmup_steps, not both"
wandb_log = (len(args.wandb_project) > 0)

transformer_config = {
    'model_name': "sim",
    'max_n_electrodes': 128,#158,
    'n_freq_features': 37,
    'max_n_time_bins': 24, # 3 second of time (every bin is 125 ms)
    'd_model': args.dm,
    'n_heads': args.nh,
    'n_layers': args.nl,
    'dropout': args.dr,
    'mask_type': args.mt,
    'dtype': getattr(torch, args.dtype),
    'device': device,
    'optimizer': args.optimizer,
    'electrode_embedding_init': args.electrode_embedding_init,
}
transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']
transformer_config['dim_output'] = transformer_config['d_model']

# Set all random seeds for reproducibility
if (not ('random_string' in training_config)) or (len(training_config['random_string']) == 0):
    training_config['random_string'] = str(time.time())[-5:]
random_seed = int(training_config['random_string'], 36) * 1000000 + 123456
random_seed **= 2
random_seed %= 2**32
training_config['random_seed'] = random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

def update_dir_name():
    dir_name = f"training_results/{transformer_config['model_name']}"
    dir_name += f"_s{args.subjects}"
    dir_name += f"_t{transformer_config['max_n_time_bins']}"
    dir_name += f"_dm{transformer_config['d_model']}"
    dir_name += f"_nh{transformer_config['n_heads']}"
    dir_name += f"_nl{transformer_config['n_layers']}"
    dir_name += f"_dr{transformer_config['dropout']}"
    dir_name += f"_{str(transformer_config['dtype']).split('.')[1].replace('float', 'f')}"
    dir_name += f"_mt{''.join([x[0] for x in transformer_config['mask_type'].split('-')]).upper()}"
    dir_name += f"_opt{transformer_config['optimizer']}"
    dir_name += f"_ei{transformer_config['electrode_embedding_init'][0].upper()}"
    dir_name += f"_bs{training_config['batch_size']}"
    dir_name += f"_wd{training_config['weight_decay']}"
    dir_name += f"_mg{training_config['max_gradient_norm']}"
    dir_name += f"_lrmax{training_config['lr_max']}"
    dir_name += f"_lrmin{training_config['lr_min']}"
    dir_name += f"_lrwm{training_config['lr_warmup_steps']}" if 'lr_warmup_steps' in training_config else f"_lrwf{training_config['lr_warmup_frac']}"
    dir_name += f"_r{training_config['random_string']}"
    return dir_name
dir_name = update_dir_name()
training_config['dir_name'] = dir_name

def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                backend='newtonschulz5', backend_steps=5):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps
        )
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeroth_power_via_newtonschulz5

            for i, p in enumerate(group['params']):
                g = p.grad
                assert g is not None
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_backend(g, steps=group['backend_steps'])
                p.data.add_(g, alpha=-lr)

class BrainTreebankSubjectTrialDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda', randomize_chunk_order=True, p_test_chunks=0.0):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.n_time_bins = transformer_config['max_n_time_bins']
        # Load metadata
        metadata_path = f"braintreebank_data_chunks/subject{self.subject_id}_trial{self.trial_id}.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        # Store metadata fields
        self.n_electrodes = self.metadata['n_electrodes']
        self.n_time_bins = self.metadata['n_time_bins'] 
        self.total_samples = self.metadata['total_samples']
        self.n_chunks = self.metadata['n_chunks']
        self.laplacian_rereferenced = self.metadata['laplacian_rereferenced']
        self.n_freq_features = self.metadata['n_freq_features'] if 'n_freq_features' in self.metadata else transformer_config['n_freq_features']

        self.test_chunk_ids = np.random.choice(self.n_chunks, size=int(self.n_chunks*p_test_chunks), replace=False)
        self.train_chunk_ids = np.setdiff1d(np.arange(self.n_chunks), self.test_chunk_ids)
        self.randomize_chunk_order = randomize_chunk_order
        if self.randomize_chunk_order:
            np.random.shuffle(self.train_chunk_ids)
            np.random.shuffle(self.test_chunk_ids)
        self.current_train_chunk = 0
        self.current_test_chunk = 0
        
        self.train_chunks = []
        self.test_chunks = []

    def _load_chunk(self, chunk_id):
        chunk_path = f"braintreebank_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path))
        return chunk_data.unsqueeze(0)
    
    def _get_next_chunk_id(self):
        self.current_train_chunk += 1
        return self.train_chunk_ids[self.current_train_chunk-1]
    def _have_next_chunk(self):
        return self.current_train_chunk < len(self.train_chunk_ids)
    
    def _get_next_test_chunk_id(self):
        self.current_test_chunk += 1
        return self.test_chunk_ids[self.current_test_chunk-1]
    def _have_next_test_chunk(self):
        return self.current_test_chunk < len(self.test_chunk_ids)

    def reset(self):
        self.train_chunks = []
        self.current_train_chunk = 0
        if self.randomize_chunk_order:
            np.random.shuffle(self.train_chunk_ids)
    def reset_test(self):
        self.test_chunks = []
        self.current_test_chunk = 0
        if self.randomize_chunk_order:
            np.random.shuffle(self.test_chunk_ids)

    def get_next_batch(self, batch_size):
        # Remove oldest chunk if we have 5 chunks
        self.train_chunks = self.train_chunks[batch_size:]
        # Load next chunk
        while (len(self.train_chunks) < batch_size) and (self._have_next_chunk()):
            new_chunk = self._load_chunk(self._get_next_chunk_id()) # shape: (1, n_electrodes, n_time_bins, n_freq_features)
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, transformer_config['max_n_time_bins'], transformer_config['n_freq_features'])
            for i in range(new_chunk.shape[2]):
                self.train_chunks.append(new_chunk[:, :, i, :, :])
        # Combine chunks
        data = torch.cat(self.train_chunks[0:batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        return data.to(self.device, dtype=transformer_config['dtype'])
    def get_next_test_batch(self, batch_size):
        self.test_chunks = self.test_chunks[batch_size:]
        while (len(self.test_chunks) < batch_size) and (self._have_next_test_chunk()):
            new_chunk = self._load_chunk(self._get_next_test_chunk_id())
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, transformer_config['max_n_time_bins'], transformer_config['n_freq_features'])
            for i in range(new_chunk.shape[2]):
                self.test_chunks.append(new_chunk[:, :, i, :, :])
        data = torch.cat(self.test_chunks[0:batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        return data.to(self.device, dtype=transformer_config['dtype'])

    def length(self, batch_size):
        return (len(self.train_chunk_ids)-1)*(self.n_time_bins//transformer_config['max_n_time_bins'])//batch_size
    def test_length(self, batch_size):
        return (len(self.test_chunk_ids)-1)*(self.n_time_bins//transformer_config['max_n_time_bins'])//batch_size
    
class BrainTreebankDataLoader:
    def __init__(self, subject_trials, trim_electrodes_to=None, device='cuda', randomize_subject_trials=True, randomize_chunk_order=True, p_test_chunks=0.0):
        self.subject_trials = subject_trials
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.randomize_subject_trials = randomize_subject_trials
        self.randomize_chunk_order = randomize_chunk_order

        self.dataloader_store = []
        for subject_id, trial_id in self.subject_trials:
            dataloader = BrainTreebankSubjectTrialDataLoader(subject_id, trial_id, trim_electrodes_to=self.trim_electrodes_to, 
                                                             device=device, randomize_chunk_order=self.randomize_chunk_order, p_test_chunks=p_test_chunks)
            self.dataloader_store.append(dataloader)

        self.subject_electrode_emb_store = {}
        for i in range(len(self.subject_trials)):
            subject_id, trial_id = self.subject_trials[i]
            if subject_id not in self.subject_electrode_emb_store:
                torch_fun = torch.randn if transformer_config['electrode_embedding_init'] == 'normal' else torch.zeros
                embedding = torch_fun(min(self.dataloader_store[i].n_electrodes, self.trim_electrodes_to), transformer_config['d_model'])
                embedding = embedding.to(device, dtype=transformer_config['dtype']) / np.sqrt(transformer_config['d_model'])
                self.subject_electrode_emb_store[subject_id] = torch.nn.Parameter(embedding)
        #self.electrode_embeddings_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=transformer_config['dtype'], device=device))

        self.total_steps = self.__len__()
        self.current_step = 0
        self.total_steps_dataloaders = [dataloader.length(training_config['batch_size']) for dataloader in self.dataloader_store]
        self.current_step_dataloaders = [0 for _ in range(len(self.dataloader_store))]
    
    def get_next_subject_trial_id(self):
        non_empty_dataloaders = [i for i in range(len(self.dataloader_store)) if self.current_step_dataloaders[i] < self.total_steps_dataloaders[i]]
        if not self.randomize_subject_trials:
            selected_dataloader_i = non_empty_dataloaders[0]
        else:
            selected_dataloader_i = non_empty_dataloaders[np.random.randint(len(non_empty_dataloaders))]
        self.current_step_dataloaders[selected_dataloader_i] += 1
        return selected_dataloader_i
    def have_next_subject_trial(self):
        return self.current_step < self.total_steps
    def reset(self):
        self.current_step = 0
        self.current_step_dataloaders = [0 for _ in range(len(self.dataloader_store))]
        for dataloader in self.dataloader_store:
            dataloader.reset()

    def get_n_embedding_params(self):
        num_emb_params = sum(p.numel() for p in self.subject_electrode_emb_store.values())
        return num_emb_params
    def parameters(self):
        return list(self.subject_electrode_emb_store.values())# + [self.electrode_embeddings_scale]
    def __len__(self):
        return np.sum([dataloader.length(training_config['batch_size']) for dataloader in self.dataloader_store])
    
    def get_next_batch(self, batch_size):
        subject_trial_id = self.get_next_subject_trial_id()
        subject_id, trial_id = self.subject_trials[subject_trial_id]
        self.current_step += 1
        return self.dataloader_store[subject_trial_id].get_next_batch(batch_size), self.subject_electrode_emb_store[subject_id], (subject_id, trial_id)

class BrainTreebankSubjectTrialBenchmarkDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda'):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.n_time_bins = transformer_config['max_n_time_bins']

    def get_chunk_input(self, chunk_id):
        chunk_path = f"braintreebank_benchmark_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path)).to(self.device, dtype=transformer_config['dtype']) # data_chunk shape: (n_chunks, n_electrodes, n_time_bins, n_freqs)
        return chunk_data.unsqueeze(1) 
    
    def get_chunk_labels(self, chunk_id, label_type='rms'):
        chunk_path = f"braintreebank_benchmark_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.csv"
        chunk_labels = pd.read_csv(chunk_path)[label_type].to_numpy() # shape: (n_chunks)
        return chunk_labels

# if __name__ == "__main__":
#     subject_id = 2
#     trial_id = 5
#     dataloader = BrainTreebankSubjectTrialBenchmarkDataLoader(subject_id, trial_id, device=device)
#     dataloader.get_chunk_input(2)
#     dataloader.get_chunk_labels(2)
#     exit()

if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Create electrode embeddings as part of the model
    electrode_transformer = ElectrodeTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])
    time_transformer = TimeTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])

    dataloader = BrainTreebankDataLoader(training_config['train_subject_trials'], 
                                         trim_electrodes_to=transformer_config['max_n_electrodes'], device=device,
                                         p_test_chunks=training_config['p_test_chunks'])

    total_steps = int(training_config['n_epochs'] * len(dataloader))
    training_config['total_steps'] = total_steps
    print(f"Total steps: {total_steps}")
    if 'lr_warmup_steps' in training_config:
        training_config['lr_warmup_frac'] = training_config['lr_warmup_steps'] / total_steps
    else:
        training_config['lr_warmup_steps'] = training_config['lr_warmup_frac'] * total_steps

    all_model_params = list(electrode_transformer.parameters()) + list(time_transformer.parameters())
    all_embedding_params = dataloader.parameters()
    all_params = all_model_params + all_embedding_params

    transformer_config['n_emb_params'] = dataloader.get_n_embedding_params()
    print(f"Number of electrode embedding parameters: {transformer_config['n_emb_params']}")
    num_model_params = sum(p.numel() for p in all_model_params if p.requires_grad)
    transformer_config['n_params'] = num_model_params
    print(f"Number of model parameters: {num_model_params}")

    optimizers = []
    if transformer_config['optimizer'] == 'Muon':
        matrix_params = [p for p in all_params if p.ndim >= 2]
        other_params = [p for p in all_params if p.ndim < 2]
        optimizers.append(Muon(matrix_params, lr=training_config['lr_max'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5))
        optimizers.append(torch.optim.Adam(other_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay']))
    else:
        optimizers = [torch.optim.Adam(all_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay'])]

    # Learning rate schedule
    warmup_steps = int(training_config['lr_warmup_steps'])
    total_steps = training_config['total_steps']
    lr_max = training_config['lr_max']
    lr_min = training_config['lr_min']
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
            return cosine_decay * (1 - lr_min / lr_max) + lr_min / lr_max
    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    # start a new wandb run to track this script
    if wandb_log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            name=dir_name.split('/')[-1],
            id=dir_name.split('/')[-1],

            # track hyperparameters and run metadata
            config={
                "training_config": training_config,
                "transformer_config": transformer_config,
            },
            settings=wandb.Settings(init_timeout=120)
        )

    avg_distance_store = []
    test_loss_store = []
    loss_store = []
    gradient_norm_store = []
    epoch_batch_store = []
    subject_trial_store = []
    training_start_time = time.time()
    overall_batch_i = -1
    for epoch_i in range(training_config['n_epochs']):
        dataloader.reset()
        for batch_i in range(len(dataloader)):
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.zero_grad()

            overall_batch_i += 1
            data, electrode_emb, subject_trial = dataloader.get_next_batch(training_config['batch_size'])
            subject_i, trial_i = subject_trial
            # data shape: (batch_size, 1, n_electrodes, n_time_bins, n_freq_features)
            # electrode_emb shape: (n_electrodes, d_model)

            electrode_output = electrode_transformer(data[:, :, :, :, :], electrode_emb) 
            # electrode_output shape: (batch_size, 1, n_electrodes+1, n_time_bins, d_model)
            electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
            time_output = time_transformer(electrode_output) # shape: (batch_size, 1, 1, n_time_bins, d_model)

            with torch.no_grad():
                # Calculate average distance between any two vectors in last dimension
                # Reshape to combine first 3 dimensions (batch_size, 1, 1)
                reshaped = time_output.reshape(-1, time_output.shape[3], time_output.shape[4]) # shape: (batch_size*1*1, n_time_bins, d_model)
                # Calculate pairwise distances between all time steps
                # Using broadcasting to compute differences
                diff = reshaped.unsqueeze(1) - reshaped.unsqueeze(2) # shape: (batch_size*1*1, n_time_bins, n_time_bins, d_model)
                distances = torch.norm(diff, dim=-1) # shape: (batch_size*1*1, n_time_bins, n_time_bins)
                # Get average distance (excluding diagonal which is 0)
                mask = ~torch.eye(distances.shape[1], dtype=torch.bool, device=distances.device)
                avg_distance = distances[:, mask].mean().item()
                avg_distance_store.append(avg_distance)

            loss = ((time_output[:, :, :, :-1, :] - time_output[:, :, :, 1:, :].detach())**2).mean()

            # Calculate time remaining
            steps_done = overall_batch_i + 1
            time_per_step = (time.time() - training_start_time) / max(steps_done, 1)
            time_remaining = time_per_step * (training_config['total_steps'] - steps_done)
            time_str = f"{int(time_remaining//3600):02d}:{int((time_remaining%3600)//60):02d}:{int(time_remaining%60):02d}"
            current_time_str = f"{int(time.time()//3600):02d}:{int((time.time()%3600)//60):02d}:{int(time.time()%60):02d}"
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**2 # Convert to MB
            
            loss.backward()
            for param in all_params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
            gradient_norm = torch.norm(torch.tensor([torch.norm(p.grad, 2).item() for p in all_params if p.grad is not None]), 2)
            if training_config['max_gradient_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(all_params, training_config['max_gradient_norm'])

            overall_test_loss = None
            if (overall_batch_i+1) % training_config['save_test_losses_every_n_batches'] == 0:
                # Calculate test loss
                subject_trial_test_loss_store = []
                with torch.no_grad():
                    for subject_trial_id in range(len(dataloader.subject_trials)):
                        subject_id, trial_id = dataloader.subject_trials[subject_trial_id]
                        subject_trial_dataloader = dataloader.dataloader_store[subject_trial_id]
                        electrode_emb = dataloader.subject_electrode_emb_store[subject_id]
                        subject_trial_dataloader.reset_test()
                        batch_test_loss_store = []
                        for test_batch_i in range(subject_trial_dataloader.test_length(training_config['batch_size'])):
                            test_data = subject_trial_dataloader.get_next_test_batch(training_config['batch_size'])
                            
                            electrode_output = electrode_transformer(test_data[:, :, :, :, :], electrode_emb) 
                            # electrode_output shape: (batch_size, 1, n_electrodes+1, n_time_bins, d_model)
                            electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
                            time_output = time_transformer(electrode_output) # shape: (batch_size, 1, 1, n_time_bins, d_model)
                            test_loss = ((time_output[:, :, :, :-1, :] - time_output[:, :, :, 1:, :].detach())**2).mean()
                            batch_test_loss_store.append(test_loss.item())
                            if np.isnan(test_loss.item()):
                                print(f"Test loss is NaN for subject {subject_id} trial {trial_id} test_batch {test_batch_i}")
                        test_loss = np.nanmean(batch_test_loss_store)
                        subject_trial_test_loss_store.append(test_loss)
                overall_test_loss = np.nanmean(subject_trial_test_loss_store).item()
                print(f"Test loss: {overall_test_loss}")

            if (overall_batch_i+1) % training_config['save_eval_every_n_batches'] == 0:
                with torch.no_grad():
                    train_chunks = [10, 11, 12]
                    test_chunks = [13, 14, 15]
                    eval_subject_id = 2
                    eval_trial_id = 1
                    dataloader = BrainTreebankSubjectTrialBenchmarkDataLoader(eval_subject_id, eval_trial_id)
                    # Collect features and labels for training chunks
                    train_features_electrode = []
                    train_features_time = []
                    train_labels = []
                    for train_chunk in train_chunks:
                        eval_input = dataloader.get_chunk_input(train_chunk)# shape: (n_chunks, 1, n_electrodes, n_time_bins, n_freq_features)
                        train_labels.append(dataloader.get_chunk_labels(train_chunk))

                        electrode_output = electrode_transformer(eval_input[:, :, :transformer_config['max_n_electrodes'], :, :], electrode_emb)
                        electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
                        time_output = time_transformer(electrode_output)
                        
                        electrode_output_mean = electrode_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                        time_output_mean = time_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                        train_features_electrode.append(electrode_output_mean)
                        train_features_time.append(time_output_mean)

                    # Collect features and labels for test chunks  
                    test_features_electrode = []
                    test_features_time = []
                    test_labels = []
                    for test_chunk in test_chunks:
                        eval_input = dataloader.get_chunk_input(test_chunk) # shape: (n_chunks, 1, n_electrodes, n_time_bins, n_freq_features)
                        test_labels.append(dataloader.get_chunk_labels(test_chunk))

                        electrode_output = electrode_transformer(eval_input[:, :, :transformer_config['max_n_electrodes'], :, :], electrode_emb)
                        electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
                        time_output = time_transformer(electrode_output) # shape: (n_chunks, 1, 1, n_time_bins, d_model)
                        
                        electrode_output_mean = electrode_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                        time_output_mean = time_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                        test_features_electrode.append(electrode_output_mean)
                        test_features_time.append(time_output_mean)

                    # Convert lists to arrays
                    train_features_electrode = np.concatenate(train_features_electrode)
                    train_features_time = np.concatenate(train_features_time)
                    train_labels = np.concatenate(train_labels)
                    test_features_electrode = np.concatenate(test_features_electrode)
                    test_features_time = np.concatenate(test_features_time)
                    test_labels = np.concatenate(test_labels)

                    # Fit linear regression and evaluate using electrode features
                    slope_e, intercept_e, r_value_e, p_value_e, std_err_e = stats.linregress(train_features_electrode, train_labels)
                    train_r_squared_electrode = r_value_e ** 2
                    test_predicted_electrode = slope_e * test_features_electrode + intercept_e
                    test_r_value_electrode = stats.pearsonr(test_labels, test_predicted_electrode)[0]
                    test_r_squared_electrode = test_r_value_electrode ** 2

                    # Fit linear regression and evaluate using time features
                    slope_t, intercept_t, r_value_t, p_value_t, std_err_t = stats.linregress(train_features_time, train_labels)
                    train_r_squared_time = r_value_t ** 2
                    test_predicted_time = slope_t * test_features_time + intercept_t
                    test_r_value_time = stats.pearsonr(test_labels, test_predicted_time)[0]
                    test_r_squared_time = test_r_value_time ** 2

                    print(f"Electrode features -- Train R-squared: {train_r_squared_electrode:.4f} -- Test R-squared: {test_r_squared_electrode:.4f} -- Time features -- Train R-squared: {train_r_squared_time:.4f} -- Test R-squared: {test_r_squared_time:.4f}")


            print(f"Batch {overall_batch_i+1}/{training_config['total_steps']} -- {subject_trial} -- epoch {epoch_i+1}/{training_config['n_epochs']} -- Loss: {loss.item():.4f} -- Avg distance: {avg_distance:.4f} -- GPU mem: {gpu_mem_used:.0f}MB -- Time left: {time_str} -- Current time: {current_time_str}s")
            if wandb_log:
                log_dict = {
                    "loss": loss.item(),
                    "gradient_norm": gradient_norm.item(),
                    "avg_distance": avg_distance,
                }
                if overall_test_loss is not None:
                    log_dict['test_loss'] = overall_test_loss
                if train_r_squared_electrode is not None:
                    log_dict['eval_train_r2_electrode'] = train_r_squared_electrode
                    log_dict['eval_test_r2_electrode'] = test_r_squared_electrode
                    log_dict['eval_train_r2_time'] = train_r_squared_time
                    log_dict['eval_test_r2_time'] = test_r_squared_time
                wandb.log(log_dict)#, **loss_per_electrode)

            loss_store.append(loss.item())
            epoch_batch_store.append((epoch_i, batch_i))
            subject_trial_store.append(subject_trial)
            gradient_norm_store.append(gradient_norm.item())
            if overall_test_loss is not None: test_loss_store.append(overall_test_loss)

            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()

            # Save losses every 20 batches
            if (overall_batch_i+1) % training_config['save_losses_every_n_batches'] == 0:
                # Convert dtype and device to strings for JSON serialization
                json_transformer_config = transformer_config.copy()
                json_transformer_config['dtype'] = str(transformer_config['dtype'])
                json_transformer_config['device'] = str(transformer_config['device'])
                with open(f'{dir_name}/metadata.json', 'w') as f:
                    json.dump({
                        'transformer_config': json_transformer_config,
                        'training_config': training_config,
                        }, f, indent=4)
                with open(f'{dir_name}/training_dynamics.json', 'w') as f:
                    json.dump({
                        'losses': loss_store,
                        'epoch_batch_store': epoch_batch_store,
                        'subject_trial_store': subject_trial_store,
                        'gradient_norm_store': gradient_norm_store,
                        'test_losses': test_loss_store,
                        'avg_distance_store': avg_distance_store,
                        }, f, indent=4)
                torch.save(electrode_transformer.state_dict(), f'{dir_name}/model_electrode_state_dict.pth')
                torch.save(time_transformer.state_dict(), f'{dir_name}/model_time_state_dict.pth')
                print(f"Saved losses and model after batch {overall_batch_i+1}")
        # Save model for this epoch
        if (epoch_i + 1) % training_config['save_network_every_n_epochs'] == 0:
            torch.save(electrode_transformer.state_dict(), f'{dir_name}/model_electrode_state_dict_epoch{epoch_i+1}.pth')
            torch.save(time_transformer.state_dict(), f'{dir_name}/model_time_state_dict_epoch{epoch_i+1}.pth')
            print(f"Saved model checkpoint for epoch {epoch_i+1}")
    if wandb_log:
        wandb.finish()