import torch, numpy as np, json
from transformer_architecture import SEEGTransformer
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

subject_id = 2
trial_id = 4
chunk_from = 0
chunk_to = 4

trim_electrodes_to = None

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
electrode_emb = torch.randn(n_electrodes, d_model).to(device)
model = SEEGTransformer(n_electrodes=n_electrodes, n_freq_features=n_freq_features, n_time_bins=n_time_bins,
                 d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1).to(device)

dataloader = BrainTreebankDataLoader(subject_id, trial_id, trim_electrodes_to=trim_electrodes_to, device=device)

# Example usage - get first 5 batches
for i in range(len(dataloader)):
    data = dataloader.get_next_batch()
    print(f"Batch {i} shape:", data.shape)
    output = model(data, electrode_emb) 
    print(f"Output {i} shape:", output.shape)