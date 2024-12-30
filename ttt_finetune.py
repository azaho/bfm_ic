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
from braintreebank_utils import Subject
from braintreebank_utils import BrainTreebankDataLoader, BrainTreebankSubjectTrialDataLoader, BrainTreebankSubjectTrialBenchmarkDataLoader
from braintreebank_utils import Muon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[fine_tune.py] Using device: {device}")

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
    metadata_path = os.path.join(args.dir_name, 'metadata.json')
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
    electrode_sd_path = os.path.join(args.dir_name, 'model_electrode_state_dict.pth')
    time_sd_path = os.path.join(args.dir_name, 'model_time_state_dict.pth')
    assert os.path.exists(electrode_sd_path), f"No model_electrode_state_dict.pth found in {args.dir_name}"
    assert os.path.exists(time_sd_path), f"No model_time_state_dict.pth found in {args.dir_name}"
    electrode_transformer.load_state_dict(torch.load(electrode_sd_path, map_location=device))
    time_transformer.load_state_dict(torch.load(time_sd_path, map_location=device))
    print("[fine_tune.py] Successfully loaded pretrained model weights.")

    print("[fine_tune.py] Building data loaders for evaluation sets...")

    # Create benchmark dataloaders for train/test
    train_chunks = np.arange(48)
    test_chunks = np.arange(48, 96)
    eval_subject_id = 3  # Using same defaults as in clip code
    eval_trial_id = 0

    eval_data_loader = BrainTreebankSubjectTrialBenchmarkDataLoader(eval_subject_id, eval_trial_id)

    # ---------------------------------------------------------------------
    # 6) Create the same optimizer(s) used originally
    # ---------------------------------------------------------------------
    all_model_params = list(electrode_transformer.parameters()) + list(time_transformer.parameters())
    all_embedding_params = eval_data_loader.parameters()  # electrode embeddings from the loader
    all_params = all_model_params + all_embedding_params

    # Example from your code: if optimizer == 'Muon', etc.
    # We'll replicate the same logic. Let's assume Muon + Adam for demonstration:
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]

    # We can read lr_max, weight_decay, etc. from the stored training_config
    lr_max = float(training_config.get('lr_max', 1e-3))
    weight_decay = float(training_config.get('weight_decay', 0.0))
    optimizer_type = training_config.get('optimizer', 'Muon')

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
    for opt in optimizers:
        schedulers.append(torch.optim.lr_scheduler.StepLR(opt, step_size=999999, gamma=1.0))

    wandb_log = (len(training_config.get('wandb_project', '')) > 0)
    if wandb_log:
        wandb.init(project=training_config['wandb_project'], name="fine_tune_run", config=training_config)

    # ---------------------------------------------------------------------
    # 7) Fine-tuning loop using benchmark evaluation approach
    # ---------------------------------------------------------------------
    print("[fine_tune.py] Starting fine-tuning...")
    fine_tune_start = time.time()

    n_epochs = training_config['n_epochs']
    for epoch_idx in range(n_epochs):
        print(f"\n===> Fine-tune epoch {epoch_idx+1}/{n_epochs}")

        # Training loop
        train_features_electrode = []
        train_features_time = []
        train_labels = []

        for train_chunk in train_chunks:
            for opt in optimizers:
                opt.zero_grad()

            eval_input = eval_train_loader.get_chunk_input(train_chunk)
            chunk_labels = eval_train_loader.get_chunk_labels(train_chunk)
            train_labels.append(chunk_labels)

            n_electrodes = eval_input.shape[2]
            permutation = torch.randperm(n_electrodes)
            eval_input = eval_input[:, :, permutation, :, :]

            electrode_output = electrode_transformer(eval_input[:, :, :n_electrodes//2, :-1, :], 
                                                  eval_train_loader.subject_electrode_emb_store[eval_subject_id][permutation][:n_electrodes//2])
            electrode_output = electrode_output[:, :, 0:1, :, :]  # just the CLS token
            time_output = time_transformer(electrode_output)

            electrode_output_mean = electrode_output.mean(dim=[1, 2, 3])
            time_output_mean = time_output.mean(dim=[1, 2, 3])

            train_features_electrode.append(electrode_output_mean.detach().cpu().float().numpy())
            train_features_time.append(time_output_mean.detach().cpu().float().numpy())

            # Use MSE loss between features and labels for fine-tuning
            loss = F.mse_loss(electrode_output_mean, torch.tensor(chunk_labels, device=device)) + \
                   F.mse_loss(time_output_mean, torch.tensor(chunk_labels, device=device))

            loss.backward()
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()

            if (len(train_features_electrode) + 1) % 10 == 0:
                elapsed = time.time() - fine_tune_start
                print(f"Chunk {len(train_features_electrode)+1}/{len(train_chunks)}, loss={loss.item():.4f}, elapsed={elapsed:.1f}s")
                if wandb_log:
                    wandb.log({"finetune_loss": loss.item()})

        # Evaluation on test chunks
        test_features_electrode = []
        test_features_time = []
        test_labels = []

        with torch.no_grad():
            for test_chunk in test_chunks:
                eval_input = eval_test_loader.get_chunk_input(test_chunk)
                test_labels.append(eval_test_loader.get_chunk_labels(test_chunk))

                n_electrodes = eval_input.shape[2]
                permutation = torch.randperm(n_electrodes)
                eval_input = eval_input[:, :, permutation, :, :]

                electrode_output = electrode_transformer(eval_input[:, :, :n_electrodes//2, :-1, :],
                                                      eval_test_loader.subject_electrode_emb_store[eval_subject_id][permutation][:n_electrodes//2])
                electrode_output = electrode_output[:, :, 0:1, :, :]
                time_output = time_transformer(electrode_output)

                electrode_output_mean = electrode_output.mean(dim=[1, 2, 3]).cpu().float().numpy()
                time_output_mean = time_output.mean(dim=[1, 2, 3]).cpu().float().numpy()

                test_features_electrode.append(electrode_output_mean)
                test_features_time.append(time_output_mean)

        # Calculate metrics
        train_features_electrode = np.concatenate(train_features_electrode)
        train_features_time = np.concatenate(train_features_time)
        train_labels = np.concatenate(train_labels)
        test_features_electrode = np.concatenate(test_features_electrode)
        test_features_time = np.concatenate(test_features_time)
        test_labels = np.concatenate(test_labels)

        # Electrode features evaluation
        electrode_regressor = sklearn.linear_model.LinearRegression()
        electrode_regressor.fit(train_features_electrode, train_labels)
        train_pred_electrode = electrode_regressor.predict(train_features_electrode)
        test_pred_electrode = electrode_regressor.predict(test_features_electrode)
        train_r_squared_electrode = sklearn.metrics.r2_score(train_labels, train_pred_electrode)
        test_r_squared_electrode = sklearn.metrics.r2_score(test_labels, test_pred_electrode)
        train_r_electrode = np.corrcoef(train_labels, train_pred_electrode)[0, 1]
        test_r_electrode = np.corrcoef(test_labels, test_pred_electrode)[0, 1]

        # Time features evaluation
        time_regressor = sklearn.linear_model.LinearRegression()
        time_regressor.fit(train_features_time, train_labels)
        train_pred_time = time_regressor.predict(train_features_time)
        test_pred_time = time_regressor.predict(test_features_time)
        train_r_squared_time = sklearn.metrics.r2_score(train_labels, train_pred_time)
        test_r_squared_time = sklearn.metrics.r2_score(test_labels, test_pred_time)
        train_r_time = np.corrcoef(train_labels, train_pred_time)[0, 1]
        test_r_time = np.corrcoef(test_labels, test_pred_time)[0, 1]

        print(f"Electrode features -- Train R2: {train_r_squared_electrode:.4f} (R: {train_r_electrode:.4f}) -- "
              f"Test R2: {test_r_squared_electrode:.4f} (R: {test_r_electrode:.4f}) -- "
              f"Time features -- Train R2: {train_r_squared_time:.4f} (R: {train_r_time:.4f}) -- "
              f"Test R2: {test_r_squared_time:.4f} (R: {test_r_time:.4f})")

        if wandb_log:
            wandb.log({
                "train_r2_electrode": train_r_squared_electrode,
                "test_r2_electrode": test_r_squared_electrode,
                "train_r2_time": train_r_squared_time,
                "test_r2_time": test_r_squared_time
            })

    print("[fine_tune.py] Fine-tuning complete.")
    if wandb_log:
        wandb.finish()

    # Optionally, save your fine-tuned model:
    torch.save(electrode_transformer.state_dict(), os.path.join(args.dir_name, 'finetuned_electrode_transformer.pth'))
    torch.save(time_transformer.state_dict(), os.path.join(args.dir_name, 'finetuned_time_transformer.pth'))
    print(f"[fine_tune.py] Saved fine-tuned model to {args.dir_name}")

if __name__ == '__main__':
    main()
