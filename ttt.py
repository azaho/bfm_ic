import torch, numpy as np, json, os, time
from transformer_architecture import SEEGTransformer
import argparse
import wandb

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
subject_2_trials = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]

args = argparse.Namespace()
args.lrmax = 0.001
args.lrmin = 0.001
args.bs = 64
args.nl = 10
args.dm = 120
args.mt = 'mask-out-none'
args.dtype = 'float32'
args.nh = 6
args.dr = 0.2
args.rs = "XX"  # Added random string parameter
args.lrwm = 0  # Added learning rate warmup steps parameter
args.wait_10s_intervals = 0
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
    parser.add_argument('--rs', type=str, default=args.rs, help='Random string')  # Added random string argument
    parser.add_argument('--lrwm', type=int, default=args.lrwm, help='Learning rate warmup steps')  # Added warmup steps argument
    parser.add_argument('--wait_10s_intervals', type=int, default=args.wait_10s_intervals, help='Wait 10s intervals (for many jobs)')
    args = parser.parse_args()
    assert args.lrmax >= args.lrmin, "Maximum learning rate must be greater than or equal to minimum learning rate"
    if args.wait_10s_intervals > 0:
        print(f"Waiting {args.wait_10s_intervals} 10s intervals")
        for i in range(args.wait_10s_intervals):
            print(f"Waiting {i+1} of {args.wait_10s_intervals}")
            time.sleep(10)

training_config = {
    'n_epochs': 240,
    'save_network_every_n_epochs': 20,
    'save_losses_every_n_batches': 20,

    'batch_size': args.bs,
    'train_subject_trials': [(2, 4)],#subject_2_trials, #[(2, 4)], #[(2, 4), (1, 1), (3, 1)],
    'lr_max': args.lrmax,
    'lr_min': args.lrmin,
    #'lr_warmup_frac': 0.01, # need to specify either warmup frac or steps
    'lr_warmup_steps': args.lrwm,
    'weight_decay': 0.000,
    'random_string': args.rs,  # Using random string from args
}
assert ('lr_warmup_frac' in training_config) != ('lr_warmup_steps' in training_config), "Need to specify either lr_warmup_frac or lr_warmup_steps, not both"

transformer_config = {
    'model_name': "trx",
    'max_n_electrodes': 13,#158,
    'n_freq_features': 37,
    'max_n_time_bins': 10, # 1 second of time (every bin is 125 ms)
    'd_model': args.dm,
    'n_heads': args.nh,
    'n_layers': args.nl,
    'dropout': args.dr,
    'mask_type': args.mt,
    'dtype': getattr(torch, args.dtype),
    'device': device,
}
transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']
transformer_config['dim_output'] = transformer_config['n_freq_features']

def update_dir_name():
    dir_name = f"training_results/{transformer_config['model_name']}"
    dir_name += f"_t{transformer_config['max_n_time_bins']}"
    dir_name += f"_dm{transformer_config['d_model']}"
    dir_name += f"_nh{transformer_config['n_heads']}"
    dir_name += f"_nl{transformer_config['n_layers']}"
    dir_name += f"_dr{transformer_config['dropout']}"
    dir_name += f"_{str(transformer_config['dtype']).split('.')[1].replace('float', 'f')}"
    dir_name += f"_mt{''.join([x[0] for x in transformer_config['mask_type'].split('-')]).upper()}"
    dir_name += f"_wd{training_config['weight_decay']}"
    dir_name += f"_lrmax{training_config['lr_max']}"
    dir_name += f"_lrmin{training_config['lr_min']}"
    dir_name += f"_lrwm{training_config['lr_warmup_steps']}" if 'lr_warmup_steps' in training_config else f"_lrwf{training_config['lr_warmup_frac']}"
    dir_name += f"_r{training_config['random_string']}"
    return dir_name
dir_name = update_dir_name()

# Set all random seeds for reproducibility
if ('random_string' in training_config) and (len(training_config['random_string']) > 0):
    random_seed = int(training_config['random_string'], 36) * 1000000 + 123456
    random_seed **= 2
    training_config['random_seed'] = random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config['random_seed'])
        torch.cuda.manual_seed_all(training_config['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
else:
    training_config['random_seed'] = -1

class BrainTreebankSubjectTrialDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda', randomize_chunk_order=True):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.chunks = []
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

        self.all_chunk_ids = np.arange(self.n_chunks)
        self.already_loaded_chunk_ids = []
        self.randomize_chunk_order = randomize_chunk_order
        if self.randomize_chunk_order:
            np.random.shuffle(self.all_chunk_ids)
        self.current_chunk = -1

    def _load_chunk(self, chunk_id):
        chunk_path = f"braintreebank_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path))
        return chunk_data.unsqueeze(0)
    
    def _get_next_chunk_id(self):
        self.current_chunk += 1
        return self.all_chunk_ids[self.current_chunk]
    def _have_next_chunk(self):
        return self.current_chunk < self.n_chunks

    def reset(self):
        self.chunks = []
        self.already_loaded_chunk_ids = []
        self.current_chunk = -1

    def get_next_batch(self, batch_size):
        # Remove oldest chunk if we have 5 chunks
        self.chunks = self.chunks[batch_size:]
        # Load next chunk
        while (len(self.chunks) < batch_size) and (self._have_next_chunk()):
            new_chunk = self._load_chunk(self._get_next_chunk_id()) # shape: (1, n_electrodes, n_time_bins, n_freq_features)
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, transformer_config['max_n_time_bins'], transformer_config['n_freq_features'])
            for i in range(new_chunk.shape[2]):
                self.chunks.append(new_chunk[:, :, i, :, :])
        # Combine chunks
        data = torch.cat(self.chunks[0:batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        return data.to(self.device, dtype=transformer_config['dtype'])

    def length(self, batch_size):
        return (self.n_chunks-1)*(self.n_time_bins//transformer_config['max_n_time_bins'])//batch_size
    
class BrainTreebankDataLoader:
    def __init__(self, subject_trials, trim_electrodes_to=None, device='cuda', randomize_subject_trials=True, randomize_chunk_order=True):
        self.subject_trials = subject_trials
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.randomize_subject_trials = randomize_subject_trials
        self.randomize_chunk_order = randomize_chunk_order
        self.current_subject_trial_i = -1

        self.dataloader_store = []
        for subject_id, trial_id in self.subject_trials:
            dataloader = BrainTreebankSubjectTrialDataLoader(subject_id, trial_id, trim_electrodes_to=self.trim_electrodes_to, 
                                                             device=device, randomize_chunk_order=self.randomize_chunk_order)
            self.dataloader_store.append(dataloader)

        self.subject_electrode_emb_store = {}
        for i in range(len(self.subject_trials)):
            subject_id, trial_id = self.subject_trials[i]
            if subject_id not in self.subject_electrode_emb_store:
                embedding = torch.randn(min(self.dataloader_store[i].n_electrodes, self.trim_electrodes_to), transformer_config['d_model'])
                embedding = embedding.to(device, dtype=transformer_config['dtype']) / np.sqrt(transformer_config['d_model'])
                self.subject_electrode_emb_store[subject_id] = torch.nn.Parameter(embedding)
        #self.electrode_embeddings_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=transformer_config['dtype'], device=device))

        self.total_steps = self.__len__()
        self.current_step = 0
    
    def get_next_subject_trial_id(self):
        if not self.have_next_subject_trial():
            self.current_subject_trial_i += 1
            return self.current_subject_trial_i
        else:
            return np.random.randint(len(self.subject_trials))
    def have_next_subject_trial(self):
        return self.current_step < self.total_steps
    def reset(self):
        self.current_step = 0
        self.current_subject_trial_i = -1
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

if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Create electrode embeddings as part of the model
    model = SEEGTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    transformer_config['n_params'] = num_model_params
    print(f"Number of model parameters: {num_model_params}")

    dataloader = BrainTreebankDataLoader(training_config['train_subject_trials'], trim_electrodes_to=transformer_config['max_n_electrodes'], device=device)
    transformer_config['n_emb_params'] = dataloader.get_n_embedding_params()
    print(f"Number of electrode embedding parameters: {transformer_config['n_emb_params']}")

    total_steps = int(training_config['n_epochs'] * len(dataloader))
    training_config['total_steps'] = total_steps
    print(f"Total steps: {total_steps}")
    if 'lr_warmup_steps' in training_config:
        training_config['lr_warmup_frac'] = training_config['lr_warmup_steps'] / total_steps
    else:
        training_config['lr_warmup_steps'] = training_config['lr_warmup_frac'] * total_steps

    all_params = list(model.parameters()) + dataloader.parameters()
    optimizer = torch.optim.Adam(all_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay'])

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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="bfm",
        name=dir_name.split('/')[-1],
        id=dir_name.split('/')[-1],

        # track hyperparameters and run metadata
        config={
            "training_config": training_config,
            "transformer_config": transformer_config,
        },
        settings=wandb.Settings(init_timeout=120)
    )

    loss_store = []
    epoch_batch_store = []
    subject_trial_store = []
    training_start_time = time.time()
    overall_batch_i = -1
    for epoch_i in range(training_config['n_epochs']):
        dataloader.reset()
        for batch_i in range(len(dataloader)):
            overall_batch_i += 1
            data, electrode_emb, subject_trial = dataloader.get_next_batch(training_config['batch_size'])
            subject_i, trial_i = subject_trial

            # Get model output on full data
            output = model(data[:, :, :, :-1, :], electrode_emb)
            loss = ((output-data[:, :, :, 1:, :])**2).mean()

            with torch.no_grad():
                loss_per_electrode = ((output-data[:, :, :, 1:, :])**2).mean(dim=[0, 1, 3, 4]).to('cpu').tolist()
                loss_per_electrode = {f"_loss_electrode_{i}": loss_per_electrode[i] for i in range(len(loss_per_electrode))}
            
            # Calculate time remaining
            steps_done = overall_batch_i + 1
            time_per_step = (time.time() - training_start_time) / max(steps_done, 1)
            time_remaining = time_per_step * (training_config['total_steps'] - steps_done)
            time_str = f"{int(time_remaining//3600):02d}:{int((time_remaining%3600)//60):02d}:{int(time_remaining%60):02d}"
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**2 # Convert to MB
            
            print(f"Batch {overall_batch_i+1}/{training_config['total_steps']} -- {subject_trial} -- epoch {epoch_i+1}/{training_config['n_epochs']}\n\tLoss: {loss.item():.4f}\n\tGPU mem: {gpu_mem_used:.0f}MB\n\tTime left: {time_str}")
            wandb.log({"loss": loss})#, **loss_per_electrode})

            loss_store.append(loss.item())
            epoch_batch_store.append((epoch_i, batch_i))
            subject_trial_store.append(subject_trial)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
                        }, f, indent=4)
                torch.save(model.state_dict(), f'{dir_name}/model_state_dict.pth')
                print(f"Saved losses and model after batch {overall_batch_i+1}")
        # Save model for this epoch
        if (epoch_i + 1) % training_config['save_network_every_n_epochs'] == 0:
            torch.save(model.state_dict(), f'{dir_name}/model_state_dict_epoch{epoch_i+1}.pth')
            print(f"Saved model checkpoint for epoch {epoch_i+1}")
    wandb.finish()