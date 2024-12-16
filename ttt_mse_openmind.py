import torch, numpy as np, json
from transformer_architecture import SEEGTransformer
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

trim_electrodes_to = 130 # TODO: make this a variable not always 100
train_subject_trials = [(2, 4)] #[(2, 4), (1, 1), (3, 1)]

n_epochs = 20
batch_size = 64
n_electrodes = trim_electrodes_to
n_freq_features = 37
n_time_bins = 10
d_model = 120# Assuming this is the model dimension
n_samples = 1
n_layers = 5
n_heads = 6
dim_output = n_freq_features

class BrainTreebankDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda', batch_size=1):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.current_chunk = 0
        self.chunks = []
        self.batch_size = batch_size
        self.n_time_bins = n_time_bins
        
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
        self.n_freq_features = self.metadata['n_freq_features'] if 'n_freq_features' in self.metadata else n_freq_features

    def _load_chunk(self, chunk_id):
        chunk_path = f"braintreebank_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path))
        return chunk_data.unsqueeze(0)

    def get_next_batch(self):
        # Remove oldest chunk if we have 5 chunks
        self.chunks = self.chunks[self.batch_size:]
            
        # Load next chunk
        while (len(self.chunks) < self.batch_size) and (self.current_chunk < self.n_chunks):
            new_chunk = self._load_chunk(self.current_chunk) # shape: (1, n_electrodes, n_time_bins, n_freq_features)
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, n_time_bins, self.n_freq_features)
            for i in range(new_chunk.shape[2]):
                self.chunks.append(new_chunk[:, :, i, :, :])
            self.current_chunk += 1

        # Combine chunks
        data = torch.cat(self.chunks[0:self.batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        return data.to(self.device)

    def __len__(self):
        return (self.n_chunks-1)*(self.n_time_bins//n_time_bins)//self.batch_size
    
    def reset(self):
        self.chunks = []
        self.current_chunk = 0

class DummyDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda', batch_size=1):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.n_time_bins = n_time_bins
        self.n_freq_features = n_freq_features
        self.n_electrodes = n_electrodes
        self.batch_size = batch_size
        self.device = device

        # Create random matrix and make it orthogonal using QR decomposition
        random_matrix = torch.randn(self.n_electrodes, self.n_electrodes)
        q, r = torch.linalg.qr(random_matrix)
        self.forward_matrix = q  # q is guaranteed to be orthogonal (rotation matrix)
        #self.forward_matrix = torch.eye(self.n_electrodes, self.n_electrodes)

    def reset(self):
        pass

    def generate_item(self):
        # Generate initial random vector
        x = torch.randn(self.n_electrodes, self.n_freq_features)
        
        # Create tensor to store the time series
        time_series = torch.zeros(1, 1, self.n_electrodes, self.n_time_bins, self.n_freq_features)
        
        # Fill first timestep with initial vector
        time_series[:, 0, :, 0, :] = x
        
        # Generate subsequent timesteps by repeatedly applying rotation
        for t in range(1, self.n_time_bins):
            x = torch.matmul(self.forward_matrix, x)  # Apply rotation
            time_series[:, 0, :, t, :] = x
            
        return time_series


    def get_next_batch(self):
        batch = torch.zeros((self.batch_size, 1, self.n_electrodes, self.n_time_bins, self.n_freq_features))
        for i in range(self.batch_size):
            batch[i, :, :, :, :] = self.generate_item()
        return batch.to(self.device)

    def __len__(self):
        return 1000

# Initialize model and dataloader
#torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

# Create electrode embeddings as part of the model
model = SEEGTransformer(n_electrodes=n_electrodes, n_freq_features=n_freq_features, n_time_bins=n_time_bins,
                 d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.2, dim_output=dim_output).to(device)

electrode_emb_store = []
for i in range(len(train_subject_trials)):
    subject_id, trial_id = train_subject_trials[i]
    electrode_emb = torch.nn.Parameter(torch.randn(n_electrodes, d_model).to(device) / np.sqrt(d_model))
    electrode_emb_store.append(electrode_emb)
electrode_embeddings_scale = torch.nn.Parameter(torch.tensor(0.1))

dataloader_store = []
for subject_id, trial_id in train_subject_trials:
    #dataloader = BrainTreebankDataLoader(subject_id, trial_id, trim_electrodes_to=trim_electrodes_to, device=device, batch_size=batch_size)
    dataloader = DummyDataLoader(subject_id, trial_id, trim_electrodes_to=trim_electrodes_to, device=device, batch_size=batch_size)
    dataloader_store.append(dataloader)

optimizer = torch.optim.Adam(list(model.parameters()) + electrode_emb_store + [electrode_embeddings_scale], lr=0.01)

loss_store = []
emb_scale_store = []
inner_batch_i_store = []
subject_trial_i_store = []

# Example usage - get first 5 batches
overall_batch_i = 0
for epoch_i in range(n_epochs):
    subject_i = -1  
    for electrode_emb, dataloader in zip(electrode_emb_store, dataloader_store):
        subject_i += 1
        print(f"Subject {subject_i+1} of {len(train_subject_trials)} ({train_subject_trials[subject_i]})")
        dataloader.reset()
        for i in range(len(dataloader)):
            overall_batch_i += 1
            data = dataloader.get_next_batch() # shape: (batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features)
            
            # Get model output on full data
            output = model(data[:, :, :, :-1, :], electrode_emb)
            
            loss = ((output-data[:, :, :, 1:, :])**2).mean()
            print(f"Batch {overall_batch_i+1}  loss: {loss.item():.4f}, emb_scale: {electrode_embeddings_scale.item()*10:.4f}")
            
            loss_store.append(loss.item())
            emb_scale_store.append(electrode_embeddings_scale.item())
            inner_batch_i_store.append(i)
            subject_trial_i_store.append(train_subject_trials[subject_i])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save losses every 20 batches
            if (overall_batch_i+1) % 20 == 0:
                with open(f'training_losses_multisubject_mse.json', 'w') as f:
                    json.dump({'losses': loss_store, 'emb_scale': emb_scale_store}, f)
                torch.save(model.state_dict(), f'model_state_dict_multisubject_mse.pth')
                print(f"Saved losses and model after batch {overall_batch_i+1}")