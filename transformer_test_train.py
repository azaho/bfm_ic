import torch, numpy as np

subject_id = 2
trial_id = 4
chunk_from = 0
chunk_to = 4

# Load chunks and concatenate them
chunks = []
for chunk_id in range(chunk_from, chunk_to + 1):
    chunk_path = f"braintreebank_data_chunks/subject{subject_id}_trial{trial_id}_chunk{chunk_id}.npy"
    chunk_data = torch.from_numpy(np.load(chunk_path))
    chunks.append(chunk_data.unsqueeze(0))

# Combine all chunks
data = torch.cat(chunks, dim=0).unsqueeze(0)
print(data.shape)