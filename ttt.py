import torch, numpy as np, json, os, time
from transformer_architecture import SEEGTransformer
import argparse

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

args = argparse.Namespace()
args.lrmax = 0.001
args.lrmin = 0.001
args.bs = 116
args.nl = 10
args.dm = 256
args.mt = 'mask-out-one'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrmax', type=float, default=0.001, help='Maximum learning rate')
    parser.add_argument('--lrmin', type=float, default=0.001, help='Minimum learning rate') 
    parser.add_argument('--bs', type=int, default=116, help='Batch size')
    parser.add_argument('--nl', type=int, default=10, help='Number of transformer layers')
    parser.add_argument('--dm', type=int, default=256, help='Model dimension')
    parser.add_argument('--mt', type=str, default='mask-out-one', help='Mask type')
    args = parser.parse_args()

training_config = {
    'n_epochs': 4,
    'save_network_every_n_epochs': 1,

    'batch_size': args.bs,
    'train_subject_trials': all_subject_trials, #[(2, 4)], #[(2, 4), (1, 1), (3, 1)],
    'lr_max': args.lrmax,
    'lr_min': args.lrmin,
    #'lr_warmup_frac': 0.01, # need to specify either warmup frac or steps
    'lr_warmup_steps': 100,
    'weight_decay': 0.001,
    'random_string': "XX",
}
assert ('lr_warmup_frac' in training_config) != ('lr_warmup_steps' in training_config), "Need to specify either lr_warmup_frac or lr_warmup_steps, not both"

transformer_config = {
    'model_name': "trx",
    'max_n_electrodes': 130,
    'n_freq_features': 37,
    'max_n_time_bins': 10,
    'd_model': args.dm,
    'n_heads': 8,
    'n_layers': args.nl,
    'dropout': 0.1,
    'mask_type': args.mt,
    'dtype': torch.bfloat16,
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
training_config['random_seed'] = int(training_config['random_string'], 36)
torch.manual_seed(training_config['random_seed'])
np.random.seed(training_config['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(training_config['random_seed'])
    torch.cuda.manual_seed_all(training_config['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BrainTreebankDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda'):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.current_chunk = 0
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

    def _load_chunk(self, chunk_id):
        chunk_path = f"braintreebank_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path))
        return chunk_data.unsqueeze(0)

    def get_next_batch(self, batch_size):
        # Remove oldest chunk if we have 5 chunks
        self.chunks = self.chunks[batch_size:]
        # Load next chunk
        while (len(self.chunks) < batch_size) and (self.current_chunk < self.n_chunks):
            new_chunk = self._load_chunk(self.current_chunk) # shape: (1, n_electrodes, n_time_bins, n_freq_features)
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, transformer_config['max_n_time_bins'], transformer_config['n_freq_features'])
            for i in range(new_chunk.shape[2]):
                self.chunks.append(new_chunk[:, :, i, :, :])
            self.current_chunk += 1
        # Combine chunks
        data = torch.cat(self.chunks[0:batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        return data.to(self.device, dtype=transformer_config['dtype'])

    def length(self, batch_size):
        return (self.n_chunks-1)*(self.n_time_bins//transformer_config['max_n_time_bins'])//batch_size
    
    def reset(self):
        self.chunks = []
        self.current_chunk = 0

if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Create electrode embeddings as part of the model
    model = SEEGTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    transformer_config['n_params'] = num_model_params
    
    print(f"Number of model parameters: {num_model_params}")

    dataloader_store = []
    for subject_id, trial_id in training_config['train_subject_trials']:
        dataloader = BrainTreebankDataLoader(subject_id, trial_id, trim_electrodes_to=transformer_config['max_n_electrodes'], device=device)
        dataloader_store.append(dataloader)

    subject_electrode_emb_store = {}
    electrode_emb_store = []
    for i in range(len(training_config['train_subject_trials'])):
        subject_id, trial_id = training_config['train_subject_trials'][i]
        if subject_id not in subject_electrode_emb_store:
            subject_electrode_emb_store[subject_id] = torch.nn.Parameter(torch.randn(dataloader_store[i].n_electrodes, transformer_config['d_model']).to(device, dtype=transformer_config['dtype']) / np.sqrt(transformer_config['d_model']))
        electrode_emb_store.append(subject_electrode_emb_store[subject_id])
    electrode_embeddings_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=transformer_config['dtype'], device=device))
    num_emb_params = sum(p.numel() for p in electrode_emb_store + [electrode_embeddings_scale])
    transformer_config['n_emb_params'] = num_emb_params
    print(f"Number of electrode embedding parameters: {num_emb_params}")
    
    total_steps = training_config['n_epochs'] * np.sum([dataloader.length(training_config['batch_size']) for dataloader in dataloader_store])
    total_steps = int(total_steps)
    training_config['total_steps'] = total_steps
    print(f"Total steps: {total_steps}")
    if 'lr_warmup_steps' in training_config:
        training_config['lr_warmup_frac'] = training_config['lr_warmup_steps'] / total_steps
    else:
        training_config['lr_warmup_steps'] = training_config['lr_warmup_frac'] * total_steps

    all_params = list(model.parameters()) + electrode_emb_store + [electrode_embeddings_scale]
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

    loss_store = []
    emb_scale_store = []
    inner_batch_i_store = []
    subject_trial_i_store = []

    training_start_time = time.time()
    overall_batch_i = -1
    for epoch_i in range(training_config['n_epochs']):
        subject_i = -1  
        for electrode_emb, dataloader in zip(electrode_emb_store, dataloader_store):
            subject_i += 1
            print(f"Subject {subject_i+1} of {len(training_config['train_subject_trials'])} ({training_config['train_subject_trials'][subject_i]})")
            dataloader.reset()
            for i in range(dataloader.length(training_config['batch_size'])):
                overall_batch_i += 1
                data = dataloader.get_next_batch(training_config['batch_size']) # shape: (batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features)
                
                # Get model output on full data
                output = model(data[:, :, :, :-1, :], electrode_emb)
                
                loss = ((output-data[:, :, :, 1:, :])**2).mean()
                
                # Calculate time remaining
                steps_done = overall_batch_i + 1
                steps_total = training_config['total_steps']
                steps_remaining = steps_total - steps_done
                time_per_step = (time.time() - training_start_time) / steps_done if steps_done > 0 else 0
                time_remaining = time_per_step * steps_remaining
                
                # Convert to hours:minutes:seconds
                hours = int(time_remaining // 3600)
                minutes = int((time_remaining % 3600) // 60)
                seconds = int(time_remaining % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Get GPU memory usage
                gpu_mem_used = torch.cuda.memory_allocated() / 1024**2 # Convert to MB
                
                print(f"Batch {overall_batch_i+1}\n\tLoss: {loss.item():.4f}\n\temb_scale: {electrode_embeddings_scale.item()*10:.4f}\n\tGPU mem: {gpu_mem_used:.0f}MB\n\tTime left: {time_str}")
                
                loss_store.append(loss.item())
                emb_scale_store.append(electrode_embeddings_scale.item())
                inner_batch_i_store.append(i)
                subject_trial_i_store.append(training_config['train_subject_trials'][subject_i])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # Save losses every 20 batches
                if (overall_batch_i+1) % 5 == 0:
                    # Convert dtype and device to strings for JSON serialization
                    json_transformer_config = transformer_config.copy()
                    json_transformer_config['dtype'] = str(transformer_config['dtype'])
                    json_transformer_config['device'] = str(transformer_config['device'])

                    for key, value in json_transformer_config.items():
                        print(f"{key}: {value}")
                        with open(f'{dir_name}/{key}.txt', 'w') as f:
                            json.dump({key: value}, f, indent=4)
                    for key, value in training_config.items():
                        print(f"{key}: {value}")
                        with open(f'{dir_name}/{key}.txt', 'w') as f:
                            json.dump({key: value}, f, indent=4)

                    with open(f'{dir_name}/metadata.json', 'w') as f:
                        json.dump({
                            'transformer_config': json_transformer_config,
                            'training_config': training_config,
                            }, f, indent=4)
                    with open(f'{dir_name}/training_dynamics.json', 'w') as f:
                        json.dump({
                            'losses': loss_store,
                            'emb_scale': emb_scale_store,
                            'inner_batch_i': inner_batch_i_store,
                            'subject_trial_i': subject_trial_i_store,
                            }, f, indent=4)
                    torch.save(model.state_dict(), f'{dir_name}/model_state_dict.pth')
                    print(f"Saved losses and model after batch {overall_batch_i+1}")
        # Save model for this epoch
        if (epoch_i + 1) % training_config['save_network_every_n_epochs'] == 0:
            torch.save(model.state_dict(), f'{dir_name}/model_state_dict_epoch{epoch_i+1}.pth')
            print(f"Saved model checkpoint for epoch {epoch_i+1}")