import argparse
import torch
import numpy as np
import json
import os
import time
import wandb
import pandas as pd
from scipy import stats
import sklearn

# Import your original modules/classes:
#   - ElectrodeTransformer, TimeTransformer (from transformer_architecture_cpc)
#   - Subject, BrainTreebankDataLoader, Muon, etc. (from braintreebank_utils)
# Adjust these imports as needed for your actual file structure:
from transformer_architecture_cpc import ElectrodeTransformer, TimeTransformer

def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                backend='newtonschulz5', backend_steps=5):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps
        )
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeroth_power_via_newtonschulz5

            for i, p in enumerate(group['params']):
                g = p.grad
                #assert g is not None
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_backend(g, steps=group['backend_steps'])
                p.data.add_(g, alpha=-lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[fine_tune.py] Using device: {device}")

class BrainTreebankSubjectTrialBenchmarkDataLoader:
    def __init__(self, subject_id, trial_id, transformer_config, trim_electrodes_to=None, device='cuda', randomize_electrode_order=True):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.transformer_config = transformer_config
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.n_time_bins = transformer_config['max_n_time_bins']
        self.randomize_electrode_order = randomize_electrode_order

        all_path = f"braintreebank_benchmark_data_chunks/subject{self.subject_id}_trial{self.trial_id}_words_df.csv"
        self.words_df = pd.read_csv(all_path)

    def get_chunk_input(self, chunk_id, permutation=None):
        chunk_path = f"braintreebank_benchmark_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path)).to(self.device, dtype=self.transformer_config['dtype']) # data_chunk shape: (n_chunks, n_electrodes, n_time_bins, n_freqs)
        if permutation is not None:
            chunk_data = chunk_data[:, permutation, :, :]
        return chunk_data.unsqueeze(1) 
    
    def get_chunk_labels(self, chunk_id, label_type='rms', percentiles=True):
        chunk_path = f"braintreebank_benchmark_data_chunks/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.csv"
        chunk_labels = pd.read_csv(chunk_path)[label_type].to_numpy() # shape: (n_chunks)
        if percentiles: 
            overall_labels = self.words_df[label_type].to_numpy()
            if percentiles:
                chunk_labels = np.array([np.mean(overall_labels < x) for x in chunk_labels])
        return chunk_labels

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained model on an evaluation dataset.")
    parser.add_argument('--dir_name', type=str, required=True,
                        help='Path to directory containing metadata.json and pretrained model state_dicts.')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs (default: 5).')
    parser.add_argument('--p_test_chunks', type=float, default=0.5,
                        help='Proportion of chunks to use for testing (default: 0.5).')
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 1) Load original metadata and training configuration from dir_name
    # ---------------------------------------------------------------------
    metadata_path = os.path.join('training_results', args.dir_name, 'metadata.json')
    assert os.path.exists(metadata_path), f"No metadata.json found in {args.dir_name}"
    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    transformer_config = meta['transformer_config']
    training_config = meta['training_config']
    transformer_config['dtype'] = getattr(torch, str(transformer_config['dtype']).replace('torch.', ''), torch.float32)
    transformer_config['device'] = device
    training_config['n_epochs'] = args.n_epochs  # override with our fine-tune epochs
    electrode_transformer = ElectrodeTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])
    time_transformer = TimeTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])
    electrode_sd_path = os.path.join('training_results', args.dir_name, 'model_electrode_state_dict.pth')
    time_sd_path = os.path.join('training_results', args.dir_name, 'model_time_state_dict.pth')
    electrode_emb_path = os.path.join('training_results', args.dir_name, 'subject_electrode_embeddings.pth')
    assert os.path.exists(electrode_sd_path), f"No model_electrode_state_dict.pth found in {args.dir_name}"
    assert os.path.exists(time_sd_path), f"No model_time_state_dict.pth found in {args.dir_name}" 
    assert os.path.exists(electrode_emb_path), f"No subject_electrode_embeddings.pth found in {args.dir_name}"
    electrode_transformer.load_state_dict(torch.load(electrode_sd_path, map_location=device, weights_only=True))
    time_transformer.load_state_dict(torch.load(time_sd_path, map_location=device, weights_only=True))
    subject_electrode_emb_store = torch.load(electrode_emb_path, map_location=device, weights_only=True)
    print("[fine_tune.py] Successfully loaded pretrained model weights and electrode embeddings.")

    print("[fine_tune.py] Building data loaders for evaluation sets...")

    # Create benchmark dataloaders for train/test
    train_chunks = np.arange(48)
    test_chunks = np.arange(48, 96)
    eval_subject_id = 3  # Using same defaults as in clip code
    eval_trial_id = 0

    eval_data_loader = BrainTreebankSubjectTrialBenchmarkDataLoader(eval_subject_id, eval_trial_id, transformer_config)

    # ---------------------------------------------------------------------
    # 6) Create the same optimizer(s) used originally
    # ---------------------------------------------------------------------
    linear_layer = torch.nn.Linear(transformer_config['d_model'], 1).to(device, dtype=transformer_config['dtype'])
    all_model_params = list(electrode_transformer.parameters()) + list(time_transformer.parameters()) + list(linear_layer.parameters())
    all_params = all_model_params + [subject_electrode_emb_store[eval_subject_id]]

    # Example from your code: if optimizer == 'Muon', etc.
    # We'll replicate the same logic. Let's assume Muon + Adam for demonstration:
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]

    # We can read lr_max, weight_decay, etc. from the stored training_config
    lr_max = float(training_config.get('lr_max', 1e-3))
    weight_decay = float(training_config.get('weight_decay', 0.0))
    optimizer_type = training_config.get('optimizer', 'Muon')
    optimizer_type = 'AdamW'

    if optimizer_type == 'Muon':
        optimizers = [
            Muon(matrix_params, lr=lr_max, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5),
            torch.optim.Adam(other_params, lr=lr_max, weight_decay=weight_decay)
        ]
    else:
        optimizers = [
            torch.optim.Adam(all_params, lr=lr_max, weight_decay=weight_decay)
        ]

    schedulers = []

    # ---------------------------------------------------------------------
    # 7) Fine-tuning loop using benchmark evaluation approach
    # ---------------------------------------------------------------------
    print("[fine_tune.py] Starting fine-tuning...")
    fine_tune_start = time.time()

    n_epochs = training_config['n_epochs']
    for epoch_idx in range(-1, n_epochs):
        print(f"\n===> Fine-tune epoch {epoch_idx+1}/{n_epochs}")

        train_features_time = []
        train_labels = []
        for train_chunk in train_chunks:
            for opt in optimizers:
                opt.zero_grad()

            eval_input = eval_data_loader.get_chunk_input(train_chunk)
            chunk_labels = eval_data_loader.get_chunk_labels(train_chunk)
            train_labels.append(chunk_labels)
            chunk_labels = torch.tensor(chunk_labels, device=device, dtype=transformer_config['dtype'])

            n_electrodes = eval_input.shape[2]
            permutation = torch.randperm(n_electrodes)
            eval_input = eval_input[:, :, permutation, :, :]

            electrode_output = electrode_transformer(eval_input[:, :, :n_electrodes//2, :-1, :], 
                                                  subject_electrode_emb_store[eval_subject_id][permutation][:n_electrodes//2])
            electrode_output = electrode_output[:, :, 0:1, :, :]  # just the CLS token
            time_output = time_transformer(electrode_output)
            time_output_mean = time_output.mean(dim=[1, 2, 3])
            model_output = linear_layer(time_output_mean)

            train_features_time.append(time_output_mean.detach().cpu().float().numpy())
            # Use MSE loss between features and labels for fine-tuning
            loss = torch.mean((model_output - chunk_labels)**2)

            if epoch_idx >= 0:
                loss.backward()
                for opt in optimizers:
                    opt.step()

            if (len(train_features_time) + 1) % 8 == 0:
                elapsed = time.time() - fine_tune_start
                print(f"Chunk {len(train_features_time)+1}/{len(train_chunks)}, loss={loss.item():.4f}, elapsed={elapsed:.1f}s")
                # if wandb_log:
                #     wandb.log({"finetune_loss": loss.item()})

        # Evaluation on test chunks
        test_features_electrode = []
        test_features_time = []
        test_labels = []

        with torch.no_grad():
            for test_chunk in test_chunks:
                eval_input = eval_data_loader.get_chunk_input(test_chunk)
                test_labels.append(eval_data_loader.get_chunk_labels(test_chunk))

                n_electrodes = eval_input.shape[2]
                permutation = torch.randperm(n_electrodes)
                eval_input = eval_input[:, :, permutation, :, :]

                electrode_output = electrode_transformer(eval_input[:, :, :n_electrodes//2, :-1, :],
                                                      subject_electrode_emb_store[eval_subject_id][permutation][:n_electrodes//2])
                electrode_output = electrode_output[:, :, 0:1, :, :]
                time_output = time_transformer(electrode_output)

                time_output_mean = time_output.mean(dim=[1, 2, 3]).cpu().float().numpy()
                test_features_time.append(time_output_mean)

        train_features_time = np.concatenate(train_features_time, axis=0)
        test_features_time = np.concatenate(test_features_time, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        # Time features evaluation
        time_regressor = sklearn.linear_model.LinearRegression()
        time_regressor.fit(train_features_time, train_labels)
        train_pred_time = time_regressor.predict(train_features_time)
        test_pred_time = time_regressor.predict(test_features_time)
        train_r_squared_time = sklearn.metrics.r2_score(train_labels, train_pred_time)
        test_r_squared_time = sklearn.metrics.r2_score(test_labels, test_pred_time)
        train_r_time = np.corrcoef(train_labels, train_pred_time)[0, 1]
        test_r_time = np.corrcoef(test_labels, test_pred_time)[0, 1]

        print(f"Time features -- Train R2: {train_r_squared_time:.4f} (R: {train_r_time:.4f}) -- "
              f"Test R2: {test_r_squared_time:.4f} (R: {test_r_time:.4f})")

        # if wandb_log:
        #     wandb.log({
        #         "train_r2_time": train_r_squared_time,
        #         "test_r2_time": test_r_squared_time
        #     })

    print("[fine_tune.py] Fine-tuning complete.")
    # if wandb_log:
    #     wandb.finish()

    # Optionally, save your fine-tuned model:
    torch.save(electrode_transformer.state_dict(), os.path.join(args.dir_name, 'finetuned_electrode_transformer.pth'))
    torch.save(time_transformer.state_dict(), os.path.join(args.dir_name, 'finetuned_time_transformer.pth'))
    torch.save(subject_electrode_emb_store, os.path.join(args.dir_name, 'finetuned_subject_electrode_embeddings.pth'))
    print(f"[fine_tune.py] Saved fine-tuned model to {args.dir_name}")

if __name__ == '__main__':
    main()
