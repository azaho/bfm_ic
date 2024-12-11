from transformer_architecture import *
import torch.cuda as cuda
import psutil
import os
from torch.profiler import profile, record_function, ProfilerActivity

# Example usage
batch_size = 1
n_electrodes = 13
n_freq_features = 37
n_time_bins = 160
d_model = 128  # Assuming this is the model dimension
n_samples = 5
n_layers = 6
n_heads = 8
no_grad = False
profile = True

if no_grad:
    torch.set_grad_enabled(False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_memory_usage():
    if device.type == 'cuda':
        # GPU memory in MB
        memory_allocated = cuda.memory_allocated(device) / 1024**2
        memory_reserved = cuda.memory_reserved(device) / 1024**2
        return f"GPU Memory: {memory_allocated:.2f}MB allocated, {memory_reserved:.2f}MB reserved"
    else:
        # CPU memory in MB
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1024**2
        return f"CPU Memory: {memory:.2f}MB"

print("Initial memory usage:")
print(get_memory_usage())

# Create random input data
x = torch.randn(batch_size, n_samples, n_electrodes, n_freq_features, n_time_bins).to(device)
print("\nAfter creating input tensor:")
print(get_memory_usage())

# Create random electrode embeddings 
electrode_emb = torch.randn(n_electrodes, d_model).to(device)
print("\nAfter creating electrode embeddings:")
print(get_memory_usage())

# Initialize model
model = SEEGTransformer(n_electrodes=n_electrodes, n_freq_features=n_freq_features, n_time_bins=n_time_bins, 
                 d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1).to(device)
print("\nAfter initializing model:")
print(get_memory_usage())

# Forward pass with profiling
if profile:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True) as prof:
        with record_function("forward_pass"):
            output = model(x, electrode_emb)
else:
    output = model(x, electrode_emb)

print("\nAfter forward pass:")
print(get_memory_usage())
print(f"Output shape: {output.shape}")

if profile:
    print("\nProfiling results:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))