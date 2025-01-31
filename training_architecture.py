import torch
import torch.nn as nn
import math

transformer_config = {
    'max_n_electrodes': 130,
    'n_freq_features': 37,
    'max_n_time_bins': 10,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1,
    'dim_output': 1,
    'mask_type': 'mask-out-one',
    'dtype': torch.bfloat16,  # Changed dtype to bfloat16 to match error
}
transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']

class ElectrodeEmbeddings(nn.Module):
    def __init__(self, config, subjects, device=None, embedding_dim=None):
        """
        Args:
            subjects: Dictionary of Subject objects, keyed by subject_id
            device (torch.device): Device to store the embeddings on
            config (dict): Transformer configuration
        """
        super().__init__()
        self.config = config
        self.subjects = subjects
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.config['d_model']

        self.electrode_emb = nn.ParameterDict({
            str(subject_id): nn.Parameter(
                self._get_electrode_embedding(subject_id),
                requires_grad=self.config['electrode_embedding_grad']
            ) for subject_id in subjects
        })
        if self.embedding_dim < self.config['d_model']:
            self.linear_embed = nn.Linear(self.embedding_dim, self.config['d_model'], 
                                          requires_grad=self.config['electrode_embedding_grad'])
        else: 
            self.linear_embed = lambda x: x # just identity function if embedding dim is already at d_model
    
    def _get_electrode_embedding(self, subject_id):
        if self.config['electrode_embedding'] == 'zeros':
            return torch.zeros(self.subjects[subject_id].get_n_electrodes(), self.embedding_dim)
        elif self.config['electrode_embedding'] == 'normal':
            return torch.randn(self.subjects[subject_id].get_n_electrodes(), self.embedding_dim) / self.embedding_dim ** 0.5
        
    def forward(self, subject_id, permutation=None):
        electrode_emb = self.electrode_emb[str(subject_id)]
        if permutation is not None: electrode_emb = electrode_emb[permutation]
        electrode_emb = self.linear_embed(electrode_emb)
        return electrode_emb

class ElectrodeTransformer(nn.Module):
    def __init__(self, device=None, config=transformer_config):
        super().__init__()
        self.config = config
        self.dtype = config.get('dtype', torch.bfloat16)  
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.electrode_mask = self._make_electrode_mask()
        self.cls_token = nn.Parameter(torch.randn(config['d_model']) / config['d_model'] ** 0.5)
        self.freq_projection = nn.Linear(config['dim_input'], config['d_model'])
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=config['d_model'], dim_feedforward=config['d_model']*4, nhead=config['n_heads'],
                                                                    batch_first=True, norm_first=True, dropout=config['dropout'], activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=config['n_layers']//2)
    
    def forward(self, x, electrode_emb):
        # x shape: (batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features)
        # electrode_emb shape: (n_electrodes, d_model)

        x = x.transpose(2, 3)
        batch_size, n_samples, n_electrodes, n_time_bins, n_freq_features = x.shape
        #x = x.permute(0, 1, 3, 2, 4) # (batch_size, n_samples, n_time_bins, n_electrodes, n_freq_features)
        x = x.transpose(3, 2) # (batch_size, n_samples, n_time_bins, n_electrodes, n_freq_features)
        x = x.reshape(-1, n_electrodes, n_freq_features) # (batch_size*n_samples*n_time_bins, n_electrodes, n_freq_features)
        # TODO: fix those reshapes for efficiency

        x = self.freq_projection(x) # (batch_size*n_samples*n_time_bins, n_electrodes, d_model)
        x = x + electrode_emb.unsqueeze(0)
        # Add CLS token to sequence
        cls_tokens = self.cls_token.unsqueeze(0).expand(x.size(0), -1).unsqueeze(1)  # (batch_size*n_samples*n_time_bins, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size*n_samples*n_time_bins, n_electrodes+1, d_model)
        x = self.transformer_encoder(x)

        # TODO: fix this, permuting twice is not efficient back and forth
        x = x.reshape(batch_size, n_samples, n_time_bins, n_electrodes+1, self.config['d_model'])
        #x = x.transpose(3, 2) # (batch_size, n_samples, n_electrodes+1, n_time_bins, d_model)
        return x
    
    def _make_electrode_mask(self):
        return None
    
class TimeTransformer(nn.Module):
    def __init__(self, device=None, config=transformer_config):
        super().__init__()
        self.config = config
        self.dtype = config.get('dtype', torch.bfloat16) 
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.causal_mask = self._make_causal_mask()
        self.pos_enc_unsqueezed_sin, self.pos_enc_unsqueezed_cos = self._precompute_rope_qk()
        self.precomputed_masks = (self.causal_mask, self.pos_enc_unsqueezed_sin, self.pos_enc_unsqueezed_cos) 

        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(self.precomputed_masks, config=config)
            for _ in range(config['n_layers']//2)
        ] + [
            nn.Linear(config['d_model'], config['dim_output']) # output layer to transform to 1D Energy output
        ])  

    def _precompute_rope_qk(self):
        d_head = self.config['d_model'] // self.config['n_heads']
        theta = self.config['rope_encoding_scale'] ** (-torch.arange(0, d_head//2, 2, dtype=self.dtype, device=self.device) / d_head//2)
        pos_enc = torch.arange(self.config['max_n_time_bins'], dtype=self.dtype, device=self.device).unsqueeze(-1) * theta
        pos_enc_unsqueezed = pos_enc.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        pos_enc_unsqueezed_sin = torch.sin(pos_enc_unsqueezed)
        pos_enc_unsqueezed_cos = torch.cos(pos_enc_unsqueezed)
        return pos_enc_unsqueezed_sin, pos_enc_unsqueezed_cos
    def _make_causal_mask(self):
        # causal_mask shape: (seq_len, seq_len) with True for disallowed positions
        causal_mask = torch.triu(torch.ones(self.config['max_n_time_bins'], self.config['max_n_time_bins'], dtype=torch.bool, device=self.device), diagonal=1)
        return causal_mask
    
    def forward(self, x):
        # x shape: (batch_size, n_samples, n_electrodes, n_time_bins, d_model)
        for layer in self.layers:
            x = layer(x)
        return x


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, precomputed_masks, config=transformer_config):
        super().__init__()
        self.config = config
        self.dtype = config.get('dtype', torch.bfloat16)  # Changed default to bfloat16
        self.self_attn = RoPEMultiheadAttention(precomputed_masks, config)
        self.linear1 = nn.Linear(config['d_model'], 4*config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        self.linear2 = nn.Linear(4*config['d_model'], config['d_model'])
        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm2 = nn.LayerNorm(config['d_model'])
        self.dropout1 = nn.Dropout(config['dropout'])
        self.dropout2 = nn.Dropout(config['dropout'])
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x.shape: (batch_size, n_samples, n_electrodes, n_time_bins, d_model)
        # the n_samples dimension: 0 = positive samples, the rest = negative samples
        #x = x.to(dtype=self.dtype)
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
    def __init__(self, precomputed_masks, config=transformer_config):
        super().__init__()
        self.config = config
        self.dtype = config.get('dtype', torch.bfloat16)  # Changed default to bfloat16
        self.causal_mask, self.pos_enc_unsqueezed_sin, self.pos_enc_unsqueezed_cos = precomputed_masks
        self.d_model = config['d_model']
        self.nhead = config['n_heads']
        self.dropout = config['dropout']
        assert self.d_model % self.nhead == 0
        self.head_dim = self.d_model // self.nhead
        self.scaling = float(self.head_dim) ** -0.5
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def apply_rope_qk(self, x):
        device = x.device
        batch_size, n_samples, n_electrodes, n_time_bins, n_heads, head_dim = x.shape
        
        cos = self.pos_enc_unsqueezed_cos[:, :, :, :n_time_bins, :, :].to(device=device, dtype=self.dtype)
        sin = self.pos_enc_unsqueezed_sin[:, :, :, :n_time_bins, :, :].to(device=device, dtype=self.dtype)
        
        x_left = x[..., :head_dim//4]
        x_right = x[..., head_dim//4:head_dim//2]
        x_left_unchanged = x[..., head_dim//2:]

        x_right_rotated = x_right * cos - x_left * sin
        x_left_rotated = x_left * cos + x_right * sin
        return torch.cat([x_left_rotated, x_right_rotated, x_left_unchanged], dim=-1)
        
    def forward(self, query, key, value):
        # query, key, value: (batch_size, n_samples, n_electrodes, n_time_bins, d_model)
        # query = query.to(dtype=self.dtype)
        # key = key.to(dtype=self.dtype)
        # value = value.to(dtype=self.dtype)
        
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
        attn = attn.view(batch_size, self.nhead,
                         n_samples_q, n_electrodes, n_time_bins,
                         n_samples_k, n_electrodes, n_time_bins)
        
        causal_mask = self.causal_mask[None, None, :n_time_bins, None, None, :n_time_bins]  # shape: (1,1,n_time_bins,1,1,n_time_bins)
        combined_mask = causal_mask 
        # combined_mask shape now effectively: (1, n_electrodes, n_time_bins, 1, n_electrodes, n_time_bins)
        combined_mask = combined_mask.expand(n_samples_q, n_electrodes, n_time_bins, n_samples_k, n_electrodes, n_time_bins)
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1,1,n_samples_q,n_electrodes,n_time_bins,n_samples_k,n_electrodes,n_time_bins)
        attn = attn.masked_fill(combined_mask, float('-inf'))

        attn = attn.view(batch_size, self.nhead, n_samples_q*n_electrodes*n_time_bins, n_samples_k*n_electrodes*n_time_bins)
        attn = torch.softmax(attn, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        #attn = torch.dropout(attn, self.dropout, self.training) # removed attention dropout for now to lower memory usage 0, :2, :10, :, :2, :10])

        # Compute output
        output = torch.matmul(attn, v.view(batch_size, self.nhead, -1, self.head_dim))
        output = output.view(batch_size, self.nhead, n_samples_q, n_electrodes, n_time_bins, self.head_dim)
        output = output.permute(0, 2, 3, 4, 1, 5).reshape(batch_size, n_samples_q, n_electrodes, n_time_bins, self.d_model)
        output = self.out_proj(output)
        return output