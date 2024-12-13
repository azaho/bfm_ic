import torch, numpy as np, json
from transformer_architecture import SEEGTransformer
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

subject_id = 2
trial_id = 4
chunk_from = 0
chunk_to = 4

trim_electrodes_to = 130 // 2

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
                 d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1).to(device)
electrode_emb = torch.nn.Parameter(torch.randn(n_electrodes, d_model).to(device) / np.sqrt(d_model))
model.register_parameter('electrode_embeddings', electrode_emb)

dataloader = BrainTreebankDataLoader(subject_id, trial_id, trim_electrodes_to=trim_electrodes_to, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Example usage - get first 5 batches
for i in range(len(dataloader)):
    data = dataloader.get_next_batch() # shape: (batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features)
    output = model(data, electrode_emb) # shape: (batch_size, n_samples, n_electrodes, n_time_bins, 1)
    
    loss = output[:, 0].mean() + torch.maximum(torch.tensor(0.0), 0.1 + output[:, 0:1] - output[:, 1:]).mean()
    print(f"Batch {i} data shape: {data.shape} , output shape: {output.shape} , loss: {loss.item()}")

    print(output[0, 0:1, :, 0, 0])
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()