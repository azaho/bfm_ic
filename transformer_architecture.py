import torch
import torch.nn as nn
import math

class SEEGTransformer(nn.Module):
    def __init__(self, n_electrodes=130, n_freq_features=37, n_time_bins=160, 
                 d_model=128, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        
        # Project frequency features to model dimension
        self.freq_projection = nn.Linear(n_freq_features, d_model)
        
        # RoPE parameters
        self.max_seq_len = n_time_bins
        self.d_model = d_model
        
        # Custom transformer encoder that applies RoPE in each layer
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ] + [
            nn.Linear(d_model, 1) # output layer to transform to 1D Energy output
        ])

    def forward(self, x, electrode_emb):
        # x shape: (batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features)
        # electrode_emb shape: (n_electrodes, d_model)
        batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features = x.shape
        
        # Project frequency features and combine with electrode embeddings
        #x = x.reshape(batch_size, -1, self.freq_projection.in_features)
        #x = self.freq_projection(x)  # (batch, samples * electrodes * time_bins, d_model)
        #x = x.reshape(batch_size, n_samples, n_electrodes, n_time_bins, self.d_model)
        x = self.freq_projection(x)

        # Add electrode embeddings
        electrode_emb_unsqueezed = electrode_emb.unsqueeze(1).unsqueeze(0).unsqueeze(0)
        x = x + electrode_emb_unsqueezed # x.shape: (batch_size, n_samples, n_electrodes, n_time_bins, d_model)
        
        # Pass through transformer layers with RoPE applied in each layer
        for layer in self.layers:
            x = layer(x)
            
        # Reshape output back
        #output = x.reshape(batch_size, n_samples, n_electrodes, n_time_bins, self.d_model)
        output = x
        return output

class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x.shape: (batch_size, n_samples, n_electrodes, n_time_bins, d_model)
        # the n_samples dimension: 0 = positive samples, the rest = negative samples
        x2 = self.norm1(x)
        # process the two parts separately: positive with self-attention, negative with cross-attention
        # for cross attention, the query is the negative samples, and the key and value are the positive samples
        x_pos = self.dropout1(self.self_attn(x2[:, :1], x2[:, :1], x2[:, :1]))
        x_neg = self.dropout1(self.self_attn(x2[:, 1:], x2[:, :1], x2[:, :1]))
        x = torch.cat([x[:, :1] + x_pos, x[:, 1:] + x_neg], dim=1)

        x2 = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x2)))))
        return x

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, max_seq_len=160, max_n_electrodes=130):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.batch_first = batch_first
        assert d_model % nhead == 0
        
        self.head_dim = d_model // nhead
        self.scaling = float(self.head_dim) ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.max_seq_len = max_seq_len
        self.rope_encoding_scale = max_seq_len
        self._precompute_rope_qk()

        self.max_n_electrodes = max_n_electrodes
        self._make_electrode_mask()
        self._make_causal_mask()

    def _precompute_rope_qk(self):
        theta = self.rope_encoding_scale ** (-torch.arange(0, self.head_dim//2, 2).float() / self.head_dim//2)
        pos_enc = torch.arange(self.max_seq_len).unsqueeze(-1) * theta
        pos_enc_unsqueezed = pos_enc.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        pos_enc_unsqueezed_sin = torch.sin(pos_enc_unsqueezed)
        pos_enc_unsqueezed_cos = torch.cos(pos_enc_unsqueezed)
        self.register_buffer('pos_enc_unsqueezed_sin', pos_enc_unsqueezed_sin)
        self.register_buffer('pos_enc_unsqueezed_cos', pos_enc_unsqueezed_cos)
    def _make_causal_mask(self):
        # causal_mask shape: (seq_len, seq_len) with True for disallowed positions
        mask = torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)
    def _make_electrode_mask(self):
        # electrode_mask shape: (n_electrodes, n_electrodes) True on diagonal (same electrode)
        indices = torch.arange(self.max_n_electrodes)
        #mask = (indices.unsqueeze(0) == indices.unsqueeze(1))
        mask = (indices.unsqueeze(0) >= indices.unsqueeze(1)) # TODO: make this only mask at the same timestep, currently masking out half of it
        self.register_buffer('electrode_mask', mask)

    def apply_rope_qk(self, x):
        device = x.device
        batch_size, n_samples, n_electrodes, n_time_bins, n_heads, head_dim = x.shape
        
        cos = self.pos_enc_unsqueezed_cos[:, :, :, :n_time_bins, :, :]
        sin = self.pos_enc_unsqueezed_sin[:, :, :, :n_time_bins, :, :]
        
        x_left = x[..., :head_dim//4]
        x_right = x[..., head_dim//4:head_dim//2]
        x_left_unchanged = x[..., head_dim//2:]

        x_right_rotated = x_right * cos - x_left * sin
        x_left_rotated = x_left * cos + x_right * sin
        return torch.cat([x_left_rotated, x_right_rotated, x_left_unchanged], dim=-1)
        
    def forward(self, query, key, value):
        # query, key, value: (batch_size, n_samples, n_electrodes, n_time_bins, d_model)
        batch_size, n_samples_q, n_electrodes, n_time_bins, d_model = query.shape
        n_samples_k = key.shape[1]

        q = self.q_proj(query).view(batch_size, n_samples_q, n_electrodes, n_time_bins, self.nhead, self.head_dim)
        k = self.k_proj(key).view(batch_size, n_samples_k, n_electrodes, n_time_bins, self.nhead, self.head_dim)
        v = self.v_proj(value).view(batch_size, n_samples_k, n_electrodes, n_time_bins, self.nhead, self.head_dim)

        # Apply RoPE
        q = self.apply_rope_qk(q)
        k = self.apply_rope_qk(k)

        # Reshape for attention: (batch, nhead, n_tokens, head_dim)
        # n_tokens = n_samples * n_electrodes * n_time_bins
        q = q.permute(0, 4, 1, 2, 3, 5).reshape(batch_size, self.nhead, -1, self.head_dim)
        k = k.permute(0, 4, 1, 2, 3, 5).reshape(batch_size, self.nhead, -1, self.head_dim)
        v = v.permute(0, 4, 1, 2, 3, 5).reshape(batch_size, self.nhead, -1, self.head_dim)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # attn shape: (batch, nhead, n_tokens_q, n_tokens_k)

        # Build combined mask from cached causal and electrode masks
        # We know that each token corresponds to a triple (sample, electrode, time_bin).
        # Indexing scheme:
        # token_index = sample * (n_electrodes * n_time_bins) + electrode * n_time_bins + time
        # We can rearrange attn back into:
        # (batch, nhead, n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)
        # Then apply masks by broadcasting.

        attn = attn.view(batch_size, self.nhead,
                         n_samples_q, n_electrodes, n_time_bins,
                         n_samples_k, n_electrodes, n_time_bins)

        # Causal mask: shape (n_time_bins, n_time_bins)
        # We want to mask future time steps. Broadcast across samples, electrodes, heads, batch.
        # We'll create a mask for all pairs of tokens that are invalid.
        # causal_mask: (n_time_bins, n_time_bins)
        # Expand causal_mask to match (n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)
        # We can do this by using unsqueeze and relying on broadcasting:
        causal_mask = self.causal_mask[None, None, :n_time_bins, None, None, :n_time_bins]  # shape: (1,1,n_time_bins,1,1,n_time_bins)
        # Create time equality mask: True where time indices are equal
        time_eq_mask = torch.eye(n_time_bins, dtype=torch.bool)[None, None, :, None, None, :]  # (1,1,n_time_bins,1,1,n_time_bins)
        
        # electrode_mask: (n_electrodes, n_electrodes), mask same electrode attention
        # Only apply electrode mask where times are equal
        electrode_mask = self.electrode_mask[None, :n_electrodes, None, None, :n_electrodes, None]  # (1, n_electrodes, 1, 1, n_electrodes, 1)
        electrode_mask = electrode_mask & time_eq_mask  # Broadcasting will handle expansion

        # Both masks are True where we must mask (i.e. set to -inf).
        # Combine them with logical_or:
        combined_mask = causal_mask | electrode_mask  # relies on broadcasting
        # combined_mask shape now effectively: (1, n_electrodes, n_time_bins, 1, n_electrodes, n_time_bins)
        # We need to also expand along the sample dimensions:
        combined_mask = combined_mask.expand(n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)
        # Now shape matches the last 6 dims of attn.

        # Move dimensions to match attn layout:
        # attn shape: (batch_size, nhead, n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)
        # combined_mask shape matches (n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)
        # Just unsqueeze batch_size and nhead:
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1,1,n_samples_q,n_electrodes,n_time_bins,n_samples_k,n_electrodes,n_time_bins)

        attn = attn.masked_fill(combined_mask, float('-inf'))
        # Replace NaN values with 0 while preserving gradients

        # Now back to standard attention softmax:
        attn = attn.view(batch_size, self.nhead, n_samples_q*n_electrodes*n_time_bins, n_samples_k*n_electrodes*n_time_bins)
        attn = torch.softmax(attn, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        #attn = torch.dropout(attn, self.dropout, self.training)
        #print(attn.reshape(self.nhead, n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)[0, 0, :2, :10, :, :2, :10])

        # Compute output
        output = torch.matmul(attn, v.view(batch_size, self.nhead, -1, self.head_dim))
        output = output.view(batch_size, self.nhead, n_samples_q, n_electrodes, n_time_bins, self.head_dim)
        output = output.permute(0, 2, 3, 4, 1, 5).reshape(batch_size, n_samples_q, n_electrodes, n_time_bins, self.d_model)
        output = self.out_proj(output)
        return output