import torch, torch.nn as nn
import numpy as np, pandas as pd, json, glob, argparse, time, sklearn
from braintreebank_subject import Subject
import wandb
import random

from training_muon import Muon
from braintreebank_config import *
from braintreebank_dataloaders import *
from training_architecture_juice import *
from training_utils import *

if __name__ == "__main__": parse_args()
update_random_seed(training_config)
update_dir_name(transformer_config, training_config)
        

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.electrode_transformer = ElectrodeTransformer(config=self.config)
        self.time_transformer = TimeTransformer(config=self.config)
        self.temperature_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, electrode_embedding):
        # x shape: (batch_size, n_time_bins, n_electrodes, dim_input)
        # electrode_embedding shape: (n_electrodes, d_model)
        batch_size, n_time_bins, n_electrodes, dim_input = x.shape
        x = x[:, :, :, :transformer_config['dim_input']]

        x = x.unsqueeze(1) # add n_samples dimension
        x = self.electrode_transformer(x, electrode_embedding)
        electrode_output = x[:, :, :, 0:1, :] # just the CLS token
        if "1" in training_config['random_string']: 
            electrode_output = electrode_output.transpose(2, 3) #XXX
        time_output = self.time_transformer(electrode_output)

        # squeeze unnecessary empty dimensions
        electrode_output = electrode_output.view(batch_size, n_time_bins, self.config['dim_output']) 
        time_output = time_output.view(batch_size, n_time_bins, self.config['dim_output']) 
        return time_output, electrode_output


if __name__ == "__main__":
    subject_ids = set([subject_id for subject_id, trial_id in training_config['train_subject_trials']+training_config['eval_subject_trials']])
    subjects = {subject_id: Subject(subject_id) for subject_id in subject_ids}

    model = Model(transformer_config).to(device, dtype=transformer_config['dtype'])
    electrode_embeddings = ElectrodeEmbeddings(transformer_config, subjects, embedding_dim=transformer_config['embedding_dim']).to(device, dtype=transformer_config['dtype'])
    optimizers = create_optimizers(model, electrode_embeddings)

    train_dataloader, test_dataloader = create_traintest_dataloaders(training_config, transformer_config, verbose=True) # a single dataloader for train and test, combining all subject_trials
    eval_dataloaders = create_eval_dataloaders(training_config, transformer_config, verbose=True) # an array of dataloaders for eval, one for each subject_trial
    
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(transformer_config['max_n_time_bins']-1, 1).to(device, dtype=torch.long).reshape(-1)
    def calculate_loss(model, electrode_embeddings, batch, subject_id):
        n_electrodes = subjects[subject_id].get_n_electrodes()
        permutation = torch.randperm(n_electrodes)
        # batch shape: (batch_size, n_time_bins, n_electrodes, SPECTROGRAM_DIMENSIONALITY or N_PER_SEG)
        batch = batch[:, :, permutation, :transformer_config['dim_input']].to(device, dtype=transformer_config['dtype'])
        batch_embeddings = electrode_embeddings(subject_id, permutation)
        # all model outputs shape: (batch_size, n_time_bins, dim_output)
        eo1, to1 = model(batch[:, :-1, :n_electrodes//2, :], batch_embeddings[:n_electrodes//2])
        eo2, to2 = model(batch[:, 1:, n_electrodes//2:, :], batch_embeddings[n_electrodes//2:])
        similarity = torch.matmul(to1[:, :].transpose(0, 1), eo2[:, :].permute(1, 2, 0)) * torch.exp(model.temperature_param)
        return nn.functional.cross_entropy(similarity.view(-1, training_config['batch_size']), expanded_arange)

    if training_config['wandb_project']:
        wandb.init(project=training_config['wandb_project'], name=training_config['dir_name'].split('/')[-1], id=training_config['dir_name'].split('/')[-1],
                   config={"training_config": training_config, "transformer_config": transformer_config,}, settings=wandb.Settings(init_timeout=480))

    save_model_and_embeddings(model, electrode_embeddings, 0)
    for epoch_i in range(training_config['n_epochs']):
        print(f"EPOCH {epoch_i+1}/{training_config['n_epochs']}")
        model.train()
        epoch_loss = 0
        
        for batch_i, (subject_trial, batch) in enumerate(train_dataloader):
            # Training step
            for optimizer in optimizers: optimizer.zero_grad()
            loss = calculate_loss(model, electrode_embeddings, batch, subject_trial[0])
            loss.backward()
            for optimizer in optimizers: optimizer.step()

            epoch_loss += loss.item()
            gpu_mem_used = torch.cuda.max_memory_allocated() // 1024**2 / 1024
            torch.cuda.reset_peak_memory_stats()
            print(f"epoch {epoch_i+1}/{training_config['n_epochs']}\tdataloader {subject_trial}\tbatch {batch_i+1}/{len(train_dataloader)}\tloss {loss.item():.4f}\tgpu {gpu_mem_used:.1f}G\ttemp {model.temperature_param.item():.4f}")

        eval_results = {}
        with torch.no_grad():
            test_loss = test_model(model, electrode_embeddings, calculate_loss, test_dataloader) # test loss calculation is not in eval mode, to be comparable to train loss
            model.eval()
            for dataloader in eval_dataloaders:
                eval_results.update(eval_model(model, electrode_embeddings, dataloader))
            eval_results['test_loss'] = test_loss.item()
            eval_results['train_loss'] = epoch_loss / len(train_dataloader)

            gpu_mem_used = torch.cuda.max_memory_allocated() // 1024**2 / 1024  # Convert to GB
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for next batch
            print(f"EVAL RESULTS (gpu {gpu_mem_used:.1f}G): ", json.dumps(eval_results, indent=4))

        if ((epoch_i+1)%training_config['save_model_every_n_epochs']==0) or (epoch_i==training_config['n_epochs']-1):
            save_model_and_embeddings(model, electrode_embeddings, epoch_i+1)

        if training_config['wandb_project']:
            wandb.log(eval_results, step=(epoch_i+1)*len(train_dataloader), commit=(epoch_i+1)%training_config['wandb_commit_every_n_epochs']==0)
    if training_config['wandb_project']:
        wandb.finish()