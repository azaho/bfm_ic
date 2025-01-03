import torch, numpy as np, json
from transformer_architecture import SEEGTransformer
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

trim_electrodes_to = 13 # TODO: make this a variable not always 100
train_subject_trials = [(2, 4)]#[(2, 4), (1, 1), (3, 1)]

batch_size = 1
n_electrodes = trim_electrodes_to
n_freq_features = 37
n_time_bins = 80
d_model = 120# Assuming this is the model dimension
n_samples = 5
n_layers = 5
n_heads = 6

class BrainTreebankDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda'):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.current_chunk = 0
        self.chunks = []

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

    def _load_chunk(self, chunk_id):
        chunk_path = f"braintreebank_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path))
        return chunk_data.unsqueeze(0)

    def get_next_batch(self):
        # Remove oldest chunk if we have 5 chunks
        if len(self.chunks) == n_samples:
            self.chunks.pop()
            
        # Load next chunk
        while self.current_chunk < n_samples:
            new_chunk = self._load_chunk(self.current_chunk)
            self.chunks.insert(0, new_chunk)
            self.current_chunk += 1

        # Combine chunks
        data = torch.cat(self.chunks, dim=0).unsqueeze(0)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        return data.to(self.device)

    def __len__(self):
        return self.n_chunks-n_samples+1

# Initialize model and dataloader
#torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

# Create electrode embeddings as part of the model
model = SEEGTransformer(n_electrodes=n_electrodes, n_freq_features=n_freq_features, n_time_bins=n_time_bins,
                 d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.5).to(device)

electrode_emb_store = []
for i in range(len(train_subject_trials)):
    subject_id, trial_id = train_subject_trials[i]
    electrode_emb = torch.nn.Parameter(torch.randn(n_electrodes, d_model).to(device) / np.sqrt(d_model))
    electrode_emb_store.append(electrode_emb)
electrode_embeddings_scale = torch.nn.Parameter(torch.tensor(1.0))

dataloader_store = []
for subject_id, trial_id in train_subject_trials:
    dataloader = BrainTreebankDataLoader(subject_id, trial_id, trim_electrodes_to=trim_electrodes_to, device=device)
    dataloader_store.append(dataloader)

optimizer = torch.optim.Adam(list(model.parameters()) + electrode_emb_store + [electrode_embeddings_scale], lr=0.001, weight_decay=0.01)
L2_output_penalty = 0.1
L2_electrode_penalty = 0.01

loss_store = []
pos_energy_store = []
neg_energy_store = []
emb_scale_store = []

# Example usage - get first 5 batches
for electrode_emb, dataloader in zip(electrode_emb_store, dataloader_store):
    for i in range(len(dataloader)):
        data = dataloader.get_next_batch() # shape: (batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features)
        output = model(data, electrode_emb / torch.norm(electrode_emb) * electrode_embeddings_scale) # shape: (batch_size, n_samples, n_electrodes, n_time_bins, 1)
        
        #loss = output[:, 0].mean() + torch.maximum(torch.tensor(0.0), 0.1 + output[:, 0:1] - output[:, 1:]).mean() + L2_output_penalty * (output**2).mean()# + L2_electrode_penalty * (electrode_emb**2).mean()
        loss = output[:, 0].mean() - output[:, 1:].mean() + L2_output_penalty * (output**2).mean()# + L2_electrode_penalty * (electrode_emb**2).mean()
        loss_store.append(loss.item())
        pos_energy = output[:, 0:1].mean()
        neg_energy = output[:, 1:].mean()
        pos_energy_store.append(pos_energy.item())
        neg_energy_store.append(neg_energy.item())
        emb_scale_store.append(electrode_embeddings_scale.item())
        print(f"Batch {i}  loss: {loss.item():.4f}, pos_energy: {pos_energy.item():.4f}, neg_energy: {neg_energy.item():.4f}, emb_scale: {electrode_embeddings_scale.item():.4f}")
        #print(output[0, 0:2, :, 0, 0])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save losses every 100 batches
        if (i + 1) % 20 == 0:
            with open(f'training_losses_multisubject.json', 'w') as f:
                json.dump({'losses': loss_store, 'pos_energy': pos_energy_store, 'neg_energy': neg_energy_store, 'emb_scale': emb_scale_store}, f)
            print(f"Saved losses and energies after batch {i+1}")