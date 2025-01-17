import torch, numpy as np, json, os, time
from transformer_architecture import ElectrodeTransformer, TimeTransformer
from braintreebank_utils import Subject
import argparse
import wandb
import pandas as pd
from scipy import stats
import sklearn
from braintreebank_config import SPECTROGRAM_DIMENSIONALITY

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

args = argparse.Namespace()
args.lrmax = 0.001
args.lrmin = 0.001
args.bs = 100
args.nl = 10
args.dm = 192
args.mt = 'mask-out-none'
args.dtype = 'bfloat16'
args.nh = 12
args.dr = 0.2
args.rs = "" 
args.lrwm = 0
args.wait_n_intervals = 0
args.weight_decay = 0.000
args.optimizer = 'Muon'
args.max_gradient_norm = -1
args.electrode_embedding_init = 'normal'
args.wandb_project = "bfm_clip_bofa"
args.subjects = "3"
args.spectrogram = 1
args.binarize_eval = 1
args.temp_clip_param = 1
args.test_chunks_interleaved = 0
args.multisubj_eval = 0
args.n_freq_features = SPECTROGRAM_DIMENSIONALITY if args.spectrogram else 256
args.symmetric_loss = 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrmax', type=float, default=args.lrmax, help='Maximum learning rate')
    parser.add_argument('--lrmin', type=float, default=args.lrmin, help='Minimum learning rate') 
    parser.add_argument('--bs', type=int, default=args.bs, help='Batch size')
    parser.add_argument('--nl', type=int, default=args.nl, help='Number of transformer layers')
    parser.add_argument('--dm', type=int, default=args.dm, help='Model dimension')
    parser.add_argument('--mt', type=str, default=args.mt, help='Mask type')
    parser.add_argument('--dtype', type=str, default=args.dtype, choices=['bfloat16', 'float32'], help='Data type')
    parser.add_argument('--nh', type=int, default=args.nh, help='Number of attention heads')
    parser.add_argument('--dr', type=float, default=args.dr, help='Dropout rate')
    parser.add_argument('--rs', type=str, default=args.rs, help='Random string') 
    parser.add_argument('--lrwm', type=int, default=args.lrwm, help='Learning rate warmup steps') 
    parser.add_argument('--wait_n_intervals', type=int, default=args.wait_n_intervals, help='Wait n intervals (for many jobs)')
    parser.add_argument('--weight_decay', type=float, default=args.weight_decay, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default=args.optimizer, choices=['AdamW', 'Muon'], help='Optimizer type') # TODO: add Muon
    parser.add_argument('--max_gradient_norm', type=float, default=args.max_gradient_norm, help='Maximum gradient norm (-1 for no clipping)')
    parser.add_argument('--electrode_embedding_init', type=str, default=args.electrode_embedding_init, choices=['normal', 'zeros', 'coordinates_nograd'], help='Electrode embedding initialization')
    parser.add_argument('--wandb_project', type=str, default=args.wandb_project, help='Weights & Biases project name')
    parser.add_argument('--subjects', type=str, default=args.subjects, help='Subject numbers (digits only)')
    parser.add_argument('--spectrogram', type=int, default=args.spectrogram, help='Use spectrogram')
    parser.add_argument('--binarize_eval', type=int, default=args.binarize_eval, help='Binarize evaluation')
    parser.add_argument('--temp_clip_param', type=int, default=args.temp_clip_param, help='Use temperature clip parameter')
    parser.add_argument('--test_chunks_interleaved', type=int, default=args.test_chunks_interleaved, help='Test chunks interleaved')
    parser.add_argument('--multisubj_eval', type=int, default=args.multisubj_eval, help='Multisubject evaluation')
    parser.add_argument('--n_freq_features', type=int, default=args.n_freq_features, help='Number of frequency features')
    parser.add_argument('--symmetric_loss', type=int, default=args.symmetric_loss, help='Symmetric loss')
    args = parser.parse_args()
    assert args.lrmax >= args.lrmin, "Maximum learning rate must be greater than or equal to minimum learning rate"
    assert args.subjects.isdigit() or args.subjects == "", "Subjects parameter must contain only numbers and commas"
    assert len(args.subjects) > 0, "Subjects parameter must contain at least one subject"
    if args.wait_n_intervals > 0:
        print(f"Waiting {args.wait_n_intervals} intervals")
        for i in range(args.wait_n_intervals):
            print(f"Waiting {i+1} of {args.wait_n_intervals}")
            time.sleep(5)

all_eval_subject_trials = [(1, 2), (2, 6), (3, 0), (4, 2), (5, 0), (6, 1), (10, 0)] # made to match PopT paper
train_subject_trials = []
for subject in args.subjects:
    if subject == '0': subject = 10
    else: subject = int(subject)
    train_subject_trials.extend((subject_id, trial_id) for subject_id, trial_id in all_subject_trials if subject_id == subject and (subject_id, trial_id) not in all_eval_subject_trials)

training_config = {
    'n_epochs': 1000,
    'save_network_every_n_epochs': 100,
    'save_losses_every_n_batches': 100,
    'save_test_losses_every_n_batches': 100,
    'save_eval_every_n_batches': 100,
    'p_test_chunks': 0.15,
 
    'batch_size': args.bs,
    'train_subject_trials': train_subject_trials, #[(2, 4)], #[(2, 4), (1, 1), (3, 1)],
    'lr_max': args.lrmax,
    'lr_min': args.lrmin,
    #'lr_warmup_frac': 0.01, # need to specify either warmup frac or steps
    'lr_warmup_steps': args.lrwm,
    'weight_decay': args.weight_decay,
    'random_string': args.rs,
    'max_gradient_norm': args.max_gradient_norm,

    'wandb_project': args.wandb_project,
    'wandb_commit_every_n_batches': 1000,

    'binarize_eval': args.binarize_eval==1,
    'temp_clip_param': args.temp_clip_param==1,
    'test_chunks_interleaved': args.test_chunks_interleaved==1,
    'multisubj_eval': args.multisubj_eval==1,
    'symmetric_loss': args.symmetric_loss==1,
}
assert ('lr_warmup_frac' in training_config) != ('lr_warmup_steps' in training_config), "Need to specify either lr_warmup_frac or lr_warmup_steps, not both"
wandb_log = (len(args.wandb_project) > 0)

transformer_config = {
    'model_name': "tOOD2", # x is for loss addon, c is default clip, t is for testing deep fine tuning (no loss addon) #XXX
    'max_n_electrodes': 158,#158,
    'n_freq_features': args.n_freq_features,
    'max_n_time_bins': 24, # 3 second of time (every bin is 125 ms)
    'd_model': args.dm,
    'n_heads': args.nh,
    'n_layers': args.nl,
    'dropout': args.dr,
    'mask_type': args.mt,
    'dtype': getattr(torch, args.dtype),
    'device': device,
    'optimizer': args.optimizer,
    'electrode_embedding_init': args.electrode_embedding_init,

    'spectrogram': args.spectrogram==1,
}
transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']
transformer_config['dim_output'] = transformer_config['d_model']

# Set all random seeds for reproducibility
if (not ('random_string' in training_config)) or (len(training_config['random_string']) == 0):
    training_config['random_string'] = str(time.time())[-5:]
random_seed = int(training_config['random_string'], 36) * 1000000 + 123456
random_seed **= 2
random_seed %= 2**32
training_config['random_seed'] = random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

def update_dir_name():
    dir_name = f"training_results/{transformer_config['model_name']}"
    if not transformer_config['spectrogram']:
        dir_name += f"_ns"
    # if training_config['binarize_eval']:
    #     dir_name += f"_be"
    # if training_config['temp_clip_param']:
    #     dir_name += f"_tc"
    # if training_config['test_chunks_interleaved']:
    #     dir_name += f"_ti"
    if training_config['multisubj_eval']:
        dir_name += f"_me"
    if training_config['symmetric_loss']:
        dir_name += f"_sl"
    dir_name += f"_nff{transformer_config['n_freq_features']}"
    dir_name += f"_s{args.subjects}"
    dir_name += f"_t{transformer_config['max_n_time_bins']}"
    dir_name += f"_dm{transformer_config['d_model']}"
    dir_name += f"_nh{transformer_config['n_heads']}"
    dir_name += f"_nl{transformer_config['n_layers']}"
    dir_name += f"_dr{transformer_config['dropout']}"
    #dir_name += f"_{str(transformer_config['dtype']).split('.')[1].replace('float', 'f')}"
    #dir_name += f"_mt{''.join([x[0] for x in transformer_config['mask_type'].split('-')]).upper()}"
    dir_name += f"_opt{transformer_config['optimizer']}"
    dir_name += f"_ei{transformer_config['electrode_embedding_init'][0].upper()}"
    dir_name += f"_bs{training_config['batch_size']}"
    dir_name += f"_wd{training_config['weight_decay']}"
    #dir_name += f"_mg{training_config['max_gradient_norm']}"
    dir_name += f"_lrmax{training_config['lr_max']}"
    dir_name += f"_lrmin{training_config['lr_min']}"
    #dir_name += f"_lrwm{training_config['lr_warmup_steps']}" if 'lr_warmup_steps' in training_config else f"_lrwf{training_config['lr_warmup_frac']}"
    dir_name += f"_r{training_config['random_string']}"
    return dir_name
dir_name = update_dir_name()
training_config['dir_name'] = dir_name

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

class BrainTreebankSubjectTrialDataLoader_old:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda', randomize_chunk_order=True, p_test_chunks=0.0, spectrogram=False, cache_in_memory=True):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.n_time_bins = transformer_config['max_n_time_bins']
        self.spectrogram = spectrogram
        # Load metadata
        metadata_path = f"braintreebank_data_chunks{'_raw' if not spectrogram else ''}/subject{self.subject_id}_trial{self.trial_id}.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        # Store metadata fields
        self.n_electrodes = self.metadata['n_electrodes']
        self.n_time_bins = self.metadata['n_time_bins'] 
        self.total_samples = self.metadata['total_samples']
        self.n_chunks = self.metadata['n_chunks']
        self.laplacian_rereferenced = self.metadata['laplacian_rereferenced']
        self.n_freq_features = self.metadata['n_freq_features'] if 'n_freq_features' in self.metadata else transformer_config['n_freq_features']

        self.test_chunk_ids = np.random.choice(self.n_chunks, size=int(self.n_chunks*p_test_chunks), replace=False)
        self.train_chunk_ids = np.setdiff1d(np.arange(self.n_chunks), self.test_chunk_ids)
        self.randomize_chunk_order = randomize_chunk_order
        if self.randomize_chunk_order:
            np.random.shuffle(self.train_chunk_ids)
            np.random.shuffle(self.test_chunk_ids)
        self.current_train_chunk = 0
        self.current_test_chunk = 0
        
        self.train_chunks = []
        self.test_chunks = []

        self.cache_in_memory = cache_in_memory
        if self.cache_in_memory:
            self._preload_all_chunks()

    def _preload_all_chunks(self):
        self.chunk_data_cache = {}
        for chunk_id in range(self.n_chunks):
            chunk_path = f"braintreebank_data_chunks{'_raw' if not self.spectrogram else ''}/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
            # load once
            chunk_data = torch.from_numpy(np.load(chunk_path))
            self.chunk_data_cache[chunk_id] = chunk_data.unsqueeze(0)

    def _load_chunk(self, chunk_id):
        if self.cache_in_memory:
            return self.chunk_data_cache[chunk_id]
        else:
            # Fallback to on-demand loading
            chunk_path = f"braintreebank_data_chunks{'_raw' if not self.spectrogram else ''}/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
            chunk_data = torch.from_numpy(np.load(chunk_path))
            return chunk_data.unsqueeze(0)
    
    def _get_next_chunk_id(self):
        self.current_train_chunk += 1
        return self.train_chunk_ids[self.current_train_chunk-1]
    def _have_next_chunk(self):
        return self.current_train_chunk < len(self.train_chunk_ids)
    
    def _get_next_test_chunk_id(self):
        self.current_test_chunk += 1
        return self.test_chunk_ids[self.current_test_chunk-1]
    def _have_next_test_chunk(self):
        return self.current_test_chunk < len(self.test_chunk_ids)

    def reset(self):
        self.train_chunks = []
        self.current_train_chunk = 0
        if self.randomize_chunk_order:
            np.random.shuffle(self.train_chunk_ids)
    def reset_test(self):
        self.test_chunks = []
        self.current_test_chunk = 0
        if self.randomize_chunk_order:
            np.random.shuffle(self.test_chunk_ids)

    def get_next_batch(self, batch_size, permutation=None):
        # Remove oldest chunk if we have 5 chunks
        self.train_chunks = self.train_chunks[batch_size:]
        # Load next chunk
        while (len(self.train_chunks) < batch_size) and (self._have_next_chunk()):
            new_chunk = self._load_chunk(self._get_next_chunk_id()) # shape: (1, n_electrodes, n_time_bins, n_freq_features)
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, transformer_config['max_n_time_bins'], transformer_config['n_freq_features'])
            for i in range(new_chunk.shape[2]):
                self.train_chunks.append(new_chunk[:, :, i, :, :])
        # Combine chunks
        data = torch.cat(self.train_chunks[0:batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        if permutation is not None:
            data = data[:, :, permutation, :, :]
        return data.to(self.device, dtype=transformer_config['dtype'])
    def get_next_test_batch(self, batch_size, permutation=None):
        self.test_chunks = self.test_chunks[batch_size:]
        while (len(self.test_chunks) < batch_size) and (self._have_next_test_chunk()):
            new_chunk = self._load_chunk(self._get_next_test_chunk_id())
            new_chunk = new_chunk.reshape(1, self.n_electrodes, -1, transformer_config['max_n_time_bins'], transformer_config['n_freq_features'])
            for i in range(new_chunk.shape[2]):
                self.test_chunks.append(new_chunk[:, :, i, :, :])
        data = torch.cat(self.test_chunks[0:batch_size], dim=0).unsqueeze(1)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        if permutation is not None:
            data = data[:, :, permutation, :, :]
        return data.to(self.device, dtype=transformer_config['dtype'])

    def length(self, batch_size):
        return (len(self.train_chunk_ids)-1)*(self.n_time_bins//transformer_config['max_n_time_bins'])//batch_size
    def test_length(self, batch_size):
        return (len(self.test_chunk_ids)-1)*(self.n_time_bins//transformer_config['max_n_time_bins'])//batch_size

class BrainTreebankSubjectTrialDataLoader:
    def __init__(
        self,
        subject_id,
        trial_id,
        trim_electrodes_to=None,
        device='cuda',
        randomize_chunk_order=True,
        p_test_chunks=0.0,
        spectrogram=False,
        cache_in_memory=True
    ):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.spectrogram = spectrogram
        self.cache_in_memory = cache_in_memory
        self.randomize_chunk_order = randomize_chunk_order
        
        # Load metadata
        metadata_path = (
            f"braintreebank_data_chunks{'_raw' if not spectrogram else ''}/"
            f"subject{self.subject_id}_trial{self.trial_id}.json"
        )
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Unpack metadata
        self.n_electrodes = self.metadata['n_electrodes']
        self.n_time_bins = self.metadata['n_time_bins']
        self.total_samples = self.metadata['total_samples']
        self.n_chunks = self.metadata['n_chunks']
        self.laplacian_rereferenced = self.metadata['laplacian_rereferenced']
        self.n_freq_features = transformer_config['n_freq_features']

        # Train/test split
        self.test_chunk_ids = np.random.choice(
            self.n_chunks,
            size=int(self.n_chunks * p_test_chunks),
            replace=False
        )
        self.train_chunk_ids = np.setdiff1d(
            np.arange(self.n_chunks),
            self.test_chunk_ids
        )
        if self.randomize_chunk_order:
            np.random.shuffle(self.train_chunk_ids)
            np.random.shuffle(self.test_chunk_ids)
        
        # Iteration indices
        self.current_train_chunk = 0
        self.current_test_chunk = 0
        
        # Temp storage for chunk-based iteration
        self.train_chunks = []
        self.test_chunks = []

        # Preload if requested
        if self.cache_in_memory:
            self._preload_all_chunks()

    def _preload_all_chunks(self):
        """
        Preloads and reshapes all chunks directly onto the GPU 
        (and into the correct dtype).
        """
        self.chunk_data_cache = {}
        for chunk_id in range(self.n_chunks):
            chunk_path = (
                f"braintreebank_data_chunks{'_raw' if not self.spectrogram else ''}/"
                f"subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
            )
            # Load the chunk from disk
            chunk_data = np.load(chunk_path)
            chunk_data = torch.from_numpy(chunk_data) # shape: (n_electrodes, n_time_bins, n_freq_features)
            chunk_data = chunk_data[:, :, :self.n_freq_features] # trim to the number of frequency features (in case we're trimming the spectrogram)
            
            # Reshape into (1, n_electrodes, #windows, max_n_time_bins, n_freq_features)
            chunk_data = chunk_data.unsqueeze(0).reshape(
                1,
                self.n_electrodes,
                -1,
                transformer_config['max_n_time_bins'],
                transformer_config['n_freq_features']
            )
            
            # Move to GPU and correct dtype
            chunk_data = chunk_data.to(
                device=self.device,
                dtype=transformer_config['dtype']
            )
            
            # Cache in dictionary
            self.chunk_data_cache[chunk_id] = chunk_data

    def _load_chunk(self, chunk_id):
        """
        If cache_in_memory is True, return the already-GPU-cached chunk. 
        Otherwise, load on-demand.
        """
        if self.cache_in_memory:
            return self.chunk_data_cache[chunk_id]
        else:
            chunk_path = (
                f"braintreebank_data_chunks{'_raw' if not self.spectrogram else ''}/"
                f"subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
            )
            chunk_data = torch.from_numpy(np.load(chunk_path)) # shape: (n_electrodes, n_time_bins, n_freq_features)
            chunk_data = chunk_data[:, :, :self.n_freq_features] # trim to the number of frequency features (in case we're trimming the spectrogram)
            chunk_data = chunk_data.unsqueeze(0).reshape(
                1,
                self.n_electrodes,
                -1,
                transformer_config['max_n_time_bins'],
                transformer_config['n_freq_features']
            )
            # If not cached, we still move it to GPU here
            return chunk_data.to(
                device=self.device,
                dtype=transformer_config['dtype']
            )

    def _get_next_chunk_id(self):
        """
        Returns the next chunk id for training, and advances the pointer.
        """
        chunk_id = self.train_chunk_ids[self.current_train_chunk]
        self.current_train_chunk += 1
        return chunk_id
    
    def _have_next_chunk(self):
        """
        Check if there is still another chunk for training.
        """
        return self.current_train_chunk < len(self.train_chunk_ids)

    def _get_next_test_chunk_id(self):
        """
        Returns the next chunk id for testing, and advances the pointer.
        """
        chunk_id = self.test_chunk_ids[self.current_test_chunk]
        self.current_test_chunk += 1
        return chunk_id

    def _have_next_test_chunk(self):
        """
        Check if there is still another chunk for testing.
        """
        return self.current_test_chunk < len(self.test_chunk_ids)

    def reset(self):
        """
        Reset the training chunk iteration (for new epoch).
        """
        self.train_chunks = []
        self.current_train_chunk = 0
        if self.randomize_chunk_order:
            np.random.shuffle(self.train_chunk_ids)

    def reset_test(self):
        """
        Reset the testing chunk iteration.
        """
        self.test_chunks = []
        self.current_test_chunk = 0
        if self.randomize_chunk_order:
            np.random.shuffle(self.test_chunk_ids)

    def get_next_batch(self, batch_size, permutation=None):
        """
        Returns a batch (batch_size windows), taken from the chunk queue.
        If the queue doesn't have enough windows, load more chunks until we do.
        """
        # Drop the oldest windows if we have leftover
        self.train_chunks = self.train_chunks[batch_size:]
        
        # While we need more windows and have more chunks, load them
        while len(self.train_chunks) < batch_size and self._have_next_chunk():
            new_chunk_id = self._get_next_chunk_id()
            new_chunk = self._load_chunk(new_chunk_id)  # shape: (1, n_electrodes, #windows, max_time, n_freq)
            
            # Split along the #windows dimension and append each window separately
            # shape: (1, n_electrodes, #windows, max_n_time_bins, n_freq_features)
            num_windows = new_chunk.shape[2]
            for i in range(num_windows):
                self.train_chunks.append(new_chunk[:, :, i, :, :])  # shape: (1, n_electrodes, max_time, n_freq)
        
        # Combine the first 'batch_size' windows into one tensor
        data = torch.cat(self.train_chunks[:batch_size], dim=0).unsqueeze(1)
        # data shape is (batch_size, 1, n_electrodes, max_time, n_freq_features)
        
        # Electrode trim (optional)
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        
        # Optional electrode permutation
        if permutation is not None:
            data = data[:, :, permutation, :, :]
        
        # Since everything is already on GPU, we can just return 'data' 
        # (it’s already the correct device/dtype)
        return data

    def get_next_test_batch(self, batch_size, permutation=None):
        """
        Returns a test batch, analogous to get_next_batch.
        """
        self.test_chunks = self.test_chunks[batch_size:]
        
        while len(self.test_chunks) < batch_size and self._have_next_test_chunk():
            new_chunk_id = self._get_next_test_chunk_id()
            new_chunk = self._load_chunk(new_chunk_id)
            
            num_windows = new_chunk.shape[2]
            for i in range(num_windows):
                self.test_chunks.append(new_chunk[:, :, i, :, :])
        
        data = torch.cat(self.test_chunks[:batch_size], dim=0).unsqueeze(1)
        
        if self.trim_electrodes_to:
            data = data[:, :, :self.trim_electrodes_to, :, :]
        
        if permutation is not None:
            data = data[:, :, permutation, :, :]
        
        return data

    def length(self, batch_size):
        """
        How many training batches are left (rough approximation).
        """
        total_windows_per_chunk = self.n_time_bins // transformer_config['max_n_time_bins']
        return (len(self.train_chunk_ids) - 1) * total_windows_per_chunk // batch_size

    def test_length(self, batch_size):
        """
        How many test batches are left (rough approximation).
        """
        total_windows_per_chunk = self.n_time_bins // transformer_config['max_n_time_bins']
        return (len(self.test_chunk_ids) - 1) * total_windows_per_chunk // batch_size

    
class BrainTreebankDataLoader:
    def __init__(self, subject_trials, trim_electrodes_to=None, device='cuda', randomize_subject_trials=True, randomize_chunk_order=True, p_test_chunks=0.0, randomize_electrode_order=True, spectrogram=False):
        self.subject_trials = subject_trials
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.randomize_subject_trials = randomize_subject_trials
        self.randomize_chunk_order = randomize_chunk_order
        self.randomize_electrode_order = randomize_electrode_order
        self.spectrogram = spectrogram
        self.subject_store = {}
        for subject_id, trial_id in self.subject_trials:
            if subject_id not in self.subject_store:
                self.subject_store[subject_id] = Subject(subject_id)

        self.dataloader_store = []
        self.subject_n_electrodes = {}
        for subject_id, trial_id in self.subject_trials:
            dataloader = BrainTreebankSubjectTrialDataLoader(subject_id, trial_id, trim_electrodes_to=self.trim_electrodes_to, 
                                                             device=device, randomize_chunk_order=self.randomize_chunk_order, p_test_chunks=p_test_chunks, spectrogram=self.spectrogram)
            self.subject_n_electrodes[subject_id] = dataloader.n_electrodes
            self.dataloader_store.append(dataloader)

        self.subject_electrode_emb_store = {}
        for i in range(len(self.subject_trials)):
            subject_id, trial_id = self.subject_trials[i]
            if subject_id not in self.subject_electrode_emb_store:
                self.subject_electrode_emb_store[subject_id] = self._make_electrode_embedding(subject_id)

        self.total_steps = self.__len__()
        self.current_step = 0
        self.total_steps_dataloaders = [dataloader.length(training_config['batch_size']) for dataloader in self.dataloader_store]
        self.current_step_dataloaders = [0 for _ in range(len(self.dataloader_store))]

    def _make_electrode_embedding(self, subject_id):
        n_electrodes = min(self.subject_n_electrodes[subject_id], self.trim_electrodes_to)
        if transformer_config['electrode_embedding_init'] in ['normal', 'zeros']:
            torch_fun = torch.randn if transformer_config['electrode_embedding_init'] == 'normal' else torch.zeros
            embedding = torch_fun(n_electrodes, transformer_config['d_model'])
        elif transformer_config['electrode_embedding_init'] == 'coordinates_nograd':
            coordinates = torch.tensor(self.subject_store[subject_id].get_electrode_coordinates()).float()[:n_electrodes]
            if len(coordinates) < n_electrodes:
                padding = torch.zeros(n_electrodes - len(coordinates), 3, device=coordinates.device)
                coordinates = torch.cat([coordinates, padding], dim=0)
            # Create sinusoidal position encoding from 3D coordinates
            d_model = transformer_config['d_model']
            coordinates = (coordinates - 50) / (200 - 50)  # Normalize to [0,1] range given min/max
            
            # Calculate frequencies for each dimension
            freq = 200 ** torch.linspace(0, 1, d_model//6)
            
            # Calculate position encodings for each coordinate dimension
            l_enc = coordinates[:, 0:1] @ freq.unsqueeze(0)  # For L coordinate
            i_enc = coordinates[:, 1:2] @ freq.unsqueeze(0)  # For I coordinate  
            p_enc = coordinates[:, 2:3] @ freq.unsqueeze(0)  # For P coordinate
            padding_zeros = torch.zeros(n_electrodes, d_model-6*(d_model//6)) # padding in case model dimension is not divisible by 6
            # Combine sin and cos encodings
            embedding = torch.cat([torch.sin(l_enc), torch.cos(l_enc), torch.sin(i_enc), torch.cos(i_enc), torch.sin(p_enc), torch.cos(p_enc), padding_zeros], dim=-1)

        embedding = embedding.to(device, dtype=transformer_config['dtype']) / np.sqrt(transformer_config['d_model'])
        embedding = torch.nn.Parameter(embedding, requires_grad=('nograd' not in transformer_config['electrode_embedding_init']))
        return embedding
    
    def get_next_subject_trial_id(self):
        non_empty_dataloaders = [i for i in range(len(self.dataloader_store)) if self.current_step_dataloaders[i] < self.total_steps_dataloaders[i]]
        if not self.randomize_subject_trials:
            selected_dataloader_i = non_empty_dataloaders[0]
        else:
            selected_dataloader_i = non_empty_dataloaders[np.random.randint(len(non_empty_dataloaders))]
        self.current_step_dataloaders[selected_dataloader_i] += 1
        return selected_dataloader_i
    def have_next_subject_trial(self):
        return self.current_step < self.total_steps
    def reset(self):
        self.current_step = 0
        self.current_step_dataloaders = [0 for _ in range(len(self.dataloader_store))]
        for dataloader in self.dataloader_store:
            dataloader.reset()

    def get_n_embedding_params(self):
        num_emb_params = sum(p.numel() for p in self.subject_electrode_emb_store.values() if p.requires_grad)
        return num_emb_params
    def parameters(self):
        return list(self.subject_electrode_emb_store.values())# + [self.electrode_embeddings_scale]
    def __len__(self):
        return np.sum([dataloader.length(training_config['batch_size']) for dataloader in self.dataloader_store])
    
    def get_next_batch(self, batch_size):
        subject_trial_id = self.get_next_subject_trial_id()
        subject_id, trial_id = self.subject_trials[subject_trial_id]
        self.current_step += 1
        n_electrodes = self.subject_electrode_emb_store[subject_id].shape[0]
        if self.randomize_electrode_order:
            permutation = torch.randperm(n_electrodes)
            electrode_emb = self.subject_electrode_emb_store[subject_id][permutation]
            batch = self.dataloader_store[subject_trial_id].get_next_batch(batch_size, permutation=permutation)
        else:
            electrode_emb = self.subject_electrode_emb_store[subject_id]
            batch = self.dataloader_store[subject_trial_id].get_next_batch(batch_size)
        return batch, electrode_emb, (subject_id, trial_id)

class BrainTreebankSubjectTrialBenchmarkDataLoader:
    def __init__(self, subject_id, trial_id, trim_electrodes_to=None, device='cuda', randomize_electrode_order=True, spectrogram=False, cache_in_memory=False, percentiles=True, binarize=True, p_test_chunks=0.2, test_chunks_interleaved=False):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.trim_electrodes_to = trim_electrodes_to
        self.device = device
        self.n_time_bins = transformer_config['max_n_time_bins']
        self.randomize_electrode_order = randomize_electrode_order
        self.spectrogram = spectrogram
        self.percentiles = percentiles
        self.binarize = binarize
        self.n_freq_features = transformer_config['n_freq_features']

        all_path = f"braintreebank_benchmark_data_chunks{'_raw' if not self.spectrogram else ''}/subject{self.subject_id}_trial{self.trial_id}_words_df.csv"
        self.words_df = pd.read_csv(all_path)
        self.n_chunks = len(self.words_df) // 100
        self.p_test_chunks = p_test_chunks
        self.n_test_chunks = int(self.n_chunks * self.p_test_chunks)
        self.n_train_chunks = self.n_chunks - self.n_test_chunks
        if test_chunks_interleaved:
            self.test_chunks = np.random.choice(self.n_chunks, size=self.n_test_chunks, replace=False)
            self.train_chunks = np.array([i for i in range(self.n_chunks) if i not in self.test_chunks])
        else:
            self.test_chunks = np.arange(self.n_test_chunks)
            self.train_chunks = np.arange(self.n_test_chunks, self.n_chunks)

        self.cache_in_memory = cache_in_memory
        if self.cache_in_memory:
            self.cache_input = {}
            self.cache_labels = {}
            for chunk_id in range(self.n_chunks): # preload all chunks into memory
                self.get_chunk_input(chunk_id)
                self.get_chunk_labels(chunk_id)


    def get_chunk_input(self, chunk_id, permutation=None):
        if self.cache_in_memory and chunk_id in self.cache_input:
            return self.cache_input[chunk_id]
        chunk_path = f"braintreebank_benchmark_data_chunks{'_raw' if not self.spectrogram else ''}/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.npy"
        chunk_data = torch.from_numpy(np.load(chunk_path)) # shape: (n_chunks, n_electrodes, n_time_bins, n_freq_features)
        chunk_data = chunk_data[:, :, 8:32, :self.n_freq_features] # trim to the number of frequency features (in case we're trimming the spectrogram)#XXX
        chunk_data = chunk_data.to(self.device, dtype=transformer_config['dtype']) # data_chunk shape: (n_chunks, n_electrodes, n_time_bins, n_freqs)
        if permutation is not None:
            chunk_data = chunk_data[:, permutation, :, :]
        if self.cache_in_memory:
            self.cache_input[chunk_id] = chunk_data.unsqueeze(1)
        return chunk_data.unsqueeze(1) 
    
    def get_chunk_labels(self, chunk_id, label_type='rms'):
        if self.cache_in_memory and chunk_id in self.cache_labels:
            return self.cache_labels[chunk_id]
        chunk_path = f"braintreebank_benchmark_data_chunks{'_raw' if not self.spectrogram else ''}/subject{self.subject_id}_trial{self.trial_id}_chunk{chunk_id}.csv"
        chunk_labels = pd.read_csv(chunk_path)[label_type].to_numpy() # shape: (n_chunks)
        if self.percentiles: 
            overall_labels = self.words_df[label_type].to_numpy()
            chunk_labels = np.array([np.mean(overall_labels < x) for x in chunk_labels])
        if self.binarize:
            chunk_labels = np.where(chunk_labels > 0.75, 1, np.where(chunk_labels < 0.25, 0, -1))
            # -1 is the no-label class
        if self.cache_in_memory:
            self.cache_labels[chunk_id] = chunk_labels
        return chunk_labels

# if __name__ == "__main__":
#     subject_id = 2
#     trial_id = 5
#     dataloader = BrainTreebankSubjectTrialBenchmarkDataLoader(subject_id, trial_id, device=device)
#     dataloader.get_chunk_input(2)
#     dataloader.get_chunk_labels(2)
#     exit()

if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Create electrode embeddings as part of the model
    electrode_transformer = ElectrodeTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])
    time_transformer = TimeTransformer(config=transformer_config, device=device).to(device, dtype=transformer_config['dtype'])

    dataloader = BrainTreebankDataLoader(training_config['train_subject_trials'], 
                                         trim_electrodes_to=transformer_config['max_n_electrodes'], device=device,
                                         p_test_chunks=training_config['p_test_chunks'], spectrogram=transformer_config['spectrogram'])

    total_steps = int(training_config['n_epochs'] * len(dataloader))
    training_config['total_steps'] = total_steps
    print(f"Total steps: {total_steps}")
    if 'lr_warmup_steps' in training_config:
        training_config['lr_warmup_frac'] = training_config['lr_warmup_steps'] / total_steps
    else:
        training_config['lr_warmup_steps'] = training_config['lr_warmup_frac'] * total_steps

    all_model_params = list(electrode_transformer.parameters()) + list(time_transformer.parameters())
    all_embedding_params = dataloader.parameters()
    all_params = all_model_params + all_embedding_params

    if training_config['temp_clip_param']:
        temp_clip_param = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=transformer_config['dtype']))
        all_params.append(temp_clip_param)
    else:
        temp_clip_param = torch.tensor(0.0, device=device, dtype=transformer_config['dtype'])

    transformer_config['n_emb_params'] = dataloader.get_n_embedding_params()
    print(f"Number of electrode embedding parameters: {transformer_config['n_emb_params']}")
    num_model_params = sum(p.numel() for p in all_model_params if p.requires_grad)
    transformer_config['n_params'] = num_model_params
    print(f"Number of model parameters: {num_model_params}")

    optimizers = []
    if transformer_config['optimizer'] == 'Muon':
        matrix_params = [p for p in all_params if p.ndim >= 2]
        other_params = [p for p in all_params if p.ndim < 2]
        optimizers.append(Muon(matrix_params, lr=training_config['lr_max'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5))
        optimizers.append(torch.optim.Adam(other_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay']))
    else:
        optimizers = [torch.optim.Adam(all_params, lr=training_config['lr_max'], weight_decay=training_config['weight_decay'])]

    # Learning rate schedule
    warmup_steps = int(training_config['lr_warmup_steps'])
    total_steps = training_config['total_steps']
    lr_max = training_config['lr_max']
    lr_min = training_config['lr_min']
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
            return cosine_decay * (1 - lr_min / lr_max) + lr_min / lr_max
    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    # start a new wandb run to track this script
    if wandb_log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            name=dir_name.split('/')[-1],
            id=dir_name.split('/')[-1],

            # track hyperparameters and run metadata
            config={
                "training_config": training_config,
                "transformer_config": transformer_config,
            },
            settings=wandb.Settings(init_timeout=480)
        )

    training_on_subjects = [subject_id for subject_id, trial_id in training_config['train_subject_trials']]
    eval_subject_trials = [(subject_id, trial_id) for subject_id, trial_id in all_eval_subject_trials if subject_id in training_on_subjects]
    if not training_config['multisubj_eval']:
        eval_subject_trials = [(3, 0)] # XXX
    eval_dataloaders = [BrainTreebankSubjectTrialBenchmarkDataLoader(eval_subject_id, eval_trial_id, 
                                                                     spectrogram=transformer_config['spectrogram'], 
                                                                     cache_in_memory=True, 
                                                                     binarize=training_config['binarize_eval'],
                                                                     test_chunks_interleaved=training_config['test_chunks_interleaved']) 
                                                                     for eval_subject_id, eval_trial_id in eval_subject_trials]

    avg_distance_store = []
    test_loss_store = []
    loss_store = []
    gradient_norm_store = []
    epoch_batch_store = []
    subject_trial_store = []
    training_start_time = time.time()
    overall_batch_i = -1
    for epoch_i in range(training_config['n_epochs']):
        dataloader.reset()
        for batch_i in range(len(dataloader)):
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.zero_grad()

            overall_batch_i += 1
            data, electrode_emb, subject_trial = dataloader.get_next_batch(training_config['batch_size'])
            subject_i, trial_i = subject_trial
            # data shape: (batch_size, 1, n_electrodes, n_time_bins, n_freq_features)
            # electrode_emb shape: (n_electrodes, d_model)

            batch_size, _, n_electrodes, n_time_bins, n_freq_features = data.shape

            electrode_output = electrode_transformer(data[:, :, :n_electrodes//2, :-1, :], electrode_emb[:n_electrodes//2]) #XXX
            # electrode_output shape: (batch_size, 1, n_electrodes+1, n_time_bins, d_model)
            electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
            time_output = time_transformer(electrode_output) # shape: (batch_size, 1, 1, n_time_bins, d_model)

            electrode_output2 = electrode_transformer(data[:, :, n_electrodes//2:, 1:, :], electrode_emb[n_electrodes//2:]) #XXX
            # electrode_output shape: (batch_size, 1, n_electrodes+1, n_time_bins, d_model)
            electrode_output2 = electrode_output2[:, :, 0:1, :, :] # just the CLS token
            time_output2 = time_transformer(electrode_output2) if training_config['random_string'] == 'NT' else electrode_output2 # shape: (batch_size, 1, 1, n_time_bins, d_model)

            time_output_reshaped = time_output.squeeze(1).squeeze(1).transpose(0, 1) # shape: (n_time_bins, batch_size, d_model)
            time_output2_reshaped = time_output2.squeeze(1).squeeze(1).transpose(0, 1) # shape: (n_time_bins, batch_size, d_model)
            similarity = torch.matmul(time_output_reshaped, time_output2_reshaped.transpose(1, 2)) * torch.exp(temp_clip_param) # shape: (n_time_bins, batch_size, batch_size)

            electrode_output_reshaped = electrode_output.squeeze(1).squeeze(1).transpose(0, 1)[1:] # shape: (n_time_bins-2, batch_size, d_model)
            electrode_output2_reshaped = electrode_output2.squeeze(1).squeeze(1).transpose(0, 1)[:-1] # shape: (n_time_bins-2, batch_size, d_model)
            similarity_electrode = torch.matmul(electrode_output_reshaped, electrode_output2_reshaped.transpose(1, 2)) # shape: (n_time_bins-2, batch_size, batch_size)

            loss = 0
            expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_time_bins-1, 1).to(device, dtype=torch.long).reshape(-1)
            expanded_arange_electrode = torch.arange(batch_size).unsqueeze(0).repeat(n_time_bins-2, 1).to(device, dtype=torch.long).reshape(-1)
            loss += torch.nn.functional.cross_entropy(similarity.reshape(-1, batch_size), expanded_arange)
            if training_config['symmetric_loss']:
                loss += torch.nn.functional.cross_entropy(similarity.transpose(1, 2).reshape(-1, batch_size), expanded_arange)
                loss /= 2
            #loss += torch.nn.functional.cross_entropy(similarity_electrode.reshape(-1, batch_size), expanded_arange_electrode)
            #loss += torch.nn.functional.cross_entropy(similarity.transpose(1, 2).reshape(-1, batch_size), expanded_arange)

            # Calculate average distance between any two vectors in last dimension
            # Reshape to combine first 3 dimensions (batch_size, 1, 1)
            reshaped = time_output.reshape(-1, time_output.shape[3], time_output.shape[4]) # shape: (batch_size*1*1, n_time_bins, d_model)
            # Calculate pairwise distances between all time steps
            # Using broadcasting to compute differences
            diff = reshaped.unsqueeze(1) - reshaped.unsqueeze(2) # shape: (batch_size*1*1, n_time_bins, n_time_bins, d_model)
            distances = torch.norm(diff, dim=-1) # shape: (batch_size*1*1, n_time_bins, n_time_bins)
            # Get average distance (excluding diagonal which is 0)
            mask = ~torch.eye(distances.shape[1], dtype=torch.bool, device=distances.device)
            avg_distance = distances[:, mask].mean().item()
            avg_distance_store.append(avg_distance)

            # Calculate time remaining
            steps_done = overall_batch_i + 1
            time_per_step = (time.time() - training_start_time) / max(steps_done, 1)
            time_remaining = time_per_step * (training_config['total_steps'] - steps_done)
            time_str = f"{int(time_remaining//3600):02d}:{int((time_remaining%3600)//60):02d}:{int(time_remaining%60):02d}"
            current_time = time.localtime()
            current_time_str = f"{current_time.tm_hour:02d}:{current_time.tm_min:02d}:{current_time.tm_sec:02d}"
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**2 # Convert to MB
            
            loss.backward()
            # for param in all_params:
            #     if param.grad is None:
            #         param.grad = torch.zeros_like(param)
            gradient_norm = torch.norm(torch.tensor([torch.norm(p.grad, 2).item() for p in all_params if p.grad is not None]), 2)
            if training_config['max_gradient_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(all_params, training_config['max_gradient_norm'])

            overall_test_loss = None
            if (overall_batch_i+1) % training_config['save_test_losses_every_n_batches'] == 0:
                # Calculate test loss
                subject_trial_test_loss_store = []
                with torch.no_grad():
                    for subject_trial_id in range(len(dataloader.subject_trials)):
                        subject_id, trial_id = dataloader.subject_trials[subject_trial_id]
                        subject_trial_dataloader = dataloader.dataloader_store[subject_trial_id]
                        electrode_emb = dataloader.subject_electrode_emb_store[subject_id]
                        subject_trial_dataloader.reset_test()
                        batch_test_loss_store = []
                        for test_batch_i in range(subject_trial_dataloader.test_length(training_config['batch_size'])):
                            n_electrodes = len(electrode_emb)
                            permutation = torch.randperm(n_electrodes)
                            test_data = subject_trial_dataloader.get_next_test_batch(training_config['batch_size'], permutation=permutation)
                            
                            n_electrodes = test_data.shape[2]

                            electrode_output = electrode_transformer(test_data[:, :, :n_electrodes//2, :-1, :], electrode_emb[permutation][:n_electrodes//2]) 
                            # electrode_output shape: (batch_size, 1, n_electrodes+1, n_time_bins, d_model)
                            electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
                            time_output = time_transformer(electrode_output) # shape: (batch_size, 1, 1, n_time_bins, d_model)

                            electrode_output2 = electrode_transformer(test_data[:, :, n_electrodes//2:, 1:, :], electrode_emb[permutation][n_electrodes//2:]) 
                            # electrode_output shape: (batch_size, 1, n_electrodes+1, n_time_bins, d_model)
                            electrode_output2 = electrode_output2[:, :, 0:1, :, :] # just the CLS token
                            time_output2 = time_transformer(electrode_output2) if training_config['random_string'] == 'NT' else electrode_output2 # shape: (batch_size, 1, 1, n_time_bins, d_model)

                            time_output_reshaped = time_output.squeeze(1).squeeze(1).transpose(0, 1) # shape: (n_time_bins, batch_size, d_model)
                            time_output2_reshaped = time_output2.squeeze(1).squeeze(1).transpose(0, 1) # shape: (n_time_bins, batch_size, d_model)
                            similarity = torch.matmul(time_output_reshaped, time_output2_reshaped.transpose(1, 2)) * torch.exp(temp_clip_param) # shape: (n_time_bins, batch_size, batch_size)

                            electrode_output_reshaped = electrode_output.squeeze(1).squeeze(1).transpose(0, 1)[1:] # shape: (n_time_bins-2, batch_size, d_model)
                            electrode_output2_reshaped = electrode_output2.squeeze(1).squeeze(1).transpose(0, 1)[:-1] # shape: (n_time_bins-2, batch_size, d_model)
                            similarity_electrode = torch.matmul(electrode_output_reshaped, electrode_output2_reshaped.transpose(1, 2)) # shape: (n_time_bins, batch_size, batch_size)

                            test_loss = 0
                            expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_time_bins-1, 1).to(device, dtype=torch.long).reshape(-1)
                            expanded_arange_electrode = torch.arange(batch_size).unsqueeze(0).repeat(n_time_bins-2, 1).to(device, dtype=torch.long).reshape(-1)
                            test_loss += torch.nn.functional.cross_entropy(similarity.reshape(-1, batch_size), expanded_arange)
                            if training_config['symmetric_loss']:
                                test_loss += torch.nn.functional.cross_entropy(similarity.transpose(1, 2).reshape(-1, batch_size), expanded_arange)
                                test_loss /= 2
                            #test_loss += torch.nn.functional.cross_entropy(similarity_electrode.reshape(-1, batch_size), expanded_arange_electrode)
                            #test_loss += torch.nn.functional.cross_entropy(similarity.transpose(1, 2).reshape(-1, batch_size), expanded_arange)

                            batch_test_loss_store.append(test_loss.item())
                            if np.isnan(test_loss.item()):
                                print(f"Test loss is NaN for subject {subject_id} trial {trial_id} test_batch {test_batch_i}")
                        test_loss = np.nanmean(batch_test_loss_store)
                        subject_trial_test_loss_store.append(test_loss)
                overall_test_loss = np.nanmean(subject_trial_test_loss_store).item()
                print(f"Test loss: {overall_test_loss}")

            train_r_squared_electrode = 0
            train_r_squared_time = 0
            test_r_squared_electrode = 0
            test_r_squared_time = 0
            train_acc_electrode = 0
            train_acc_time = 0
            test_acc_electrode = 0
            test_acc_time = 0
            train_roc_electrode = 0
            train_roc_time = 0
            test_roc_electrode = 0
            test_roc_time = 0
            train_r_electrode = 0
            train_r_time = 0
            test_r_electrode = 0
            test_r_time = 0
            to_save_eval = (overall_batch_i+1) % training_config['save_eval_every_n_batches'] == 0
            if to_save_eval:
                with torch.no_grad():
                    

                    for eval_dataloader in eval_dataloaders:
                        eval_train_chunks = eval_dataloader.train_chunks
                        eval_test_chunks = eval_dataloader.test_chunks
                        eval_subject_id = eval_dataloader.subject_id
                        electrode_emb = dataloader.subject_electrode_emb_store[eval_subject_id]
                        # Collect features and labels for training chunks
                        train_features_electrode = []
                        train_features_time = []
                        train_labels = []
                        for train_chunk in eval_train_chunks:
                            eval_input = eval_dataloader.get_chunk_input(train_chunk)# shape: (n_words_per_chunk, 1, n_electrodes, n_time_bins, n_freq_features)
                            train_labels.append(eval_dataloader.get_chunk_labels(train_chunk))
                            n_electrodes = len(electrode_emb)
                            permutation = torch.randperm(n_electrodes)
                            eval_input = eval_input[:, :, :n_electrodes, :, :]
                            eval_input = eval_input[:, :, permutation, :, :]

                            electrode_output = electrode_transformer(eval_input[:, :, :n_electrodes//2, :-1, :], electrode_emb[permutation][:n_electrodes//2])
                            electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
                            time_output = time_transformer(electrode_output)
                            
                            electrode_output_mean = electrode_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                            time_output_mean = time_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                            
                            train_features_electrode.append(electrode_output_mean)
                            train_features_time.append(time_output_mean)

                        # Collect features and labels for test chunks  
                        test_features_electrode = []
                        test_features_time = []
                        test_labels = []
                        for test_chunk in eval_test_chunks:
                            eval_input = eval_dataloader.get_chunk_input(test_chunk) # shape: (n_words_per_chunk, 1, n_electrodes, n_time_bins, n_freq_features)
                            test_labels.append(eval_dataloader.get_chunk_labels(test_chunk))
                            n_electrodes = len(electrode_emb)
                            permutation = torch.randperm(n_electrodes)
                            eval_input = eval_input[:, :, :n_electrodes, :, :]
                            eval_input = eval_input[:, :, permutation, :, :]

                            electrode_output = electrode_transformer(eval_input[:, :, :n_electrodes//2, :-1, :], electrode_emb[permutation][:n_electrodes//2])
                            electrode_output = electrode_output[:, :, 0:1, :, :] # just the CLS token
                            time_output = time_transformer(electrode_output) # shape: (n_words_per_chunk, 1, 1, n_time_bins, d_model)
                            
                            electrode_output_mean = electrode_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                            time_output_mean = time_output.mean(dim=[1, 2, 3]).detach().cpu().float().numpy()
                            test_features_electrode.append(electrode_output_mean) # shape: (n_words_per_chunk, d_model)
                            test_features_time.append(time_output_mean) # shape: (n_words_per_chunk, d_model)

                        # Convert lists to arrays
                        train_features_electrode = np.concatenate(train_features_electrode)
                        train_features_time = np.concatenate(train_features_time)
                        train_labels = np.concatenate(train_labels)
                        test_features_electrode = np.concatenate(test_features_electrode)
                        test_features_time = np.concatenate(test_features_time)
                        test_labels = np.concatenate(test_labels)
                        if training_config['binarize_eval']:
                            # Filter out labels that are -1 (no-label class)
                            train_mask = train_labels >= 0
                            test_mask = test_labels >= 0
                            train_features_electrode = train_features_electrode[train_mask]
                            train_features_time = train_features_time[train_mask] 
                            train_labels = train_labels[train_mask]
                            test_features_electrode = test_features_electrode[test_mask]
                            test_features_time = test_features_time[test_mask]
                            test_labels = test_labels[test_mask]

                        if training_config['binarize_eval']:
                            # Fit logistic regression for electrode features
                            electrode_regressor = sklearn.linear_model.LogisticRegression(max_iter=10000)
                            electrode_regressor.fit(train_features_electrode, train_labels)
                            train_pred_electrode = electrode_regressor.predict_proba(train_features_electrode)[:, 1]
                            test_pred_electrode = electrode_regressor.predict_proba(test_features_electrode)[:, 1]
                            train_pred_electrode_class = electrode_regressor.predict(train_features_electrode)
                            test_pred_electrode_class = electrode_regressor.predict(test_features_electrode)
                            train_r_squared_electrode += sklearn.metrics.r2_score(train_labels, train_pred_electrode)
                            test_r_squared_electrode += sklearn.metrics.r2_score(test_labels, test_pred_electrode)
                            train_r_electrode += np.corrcoef(train_labels, train_pred_electrode)[0, 1]
                            test_r_electrode += np.corrcoef(test_labels, test_pred_electrode)[0, 1]
                            train_roc_electrode += sklearn.metrics.roc_auc_score(train_labels, train_pred_electrode)
                            test_roc_electrode += sklearn.metrics.roc_auc_score(test_labels, test_pred_electrode)
                            train_acc_electrode += sklearn.metrics.accuracy_score(train_labels, train_pred_electrode_class)
                            test_acc_electrode += sklearn.metrics.accuracy_score(test_labels, test_pred_electrode_class)

                            # Fit logistic regression for time features
                            time_regressor = sklearn.linear_model.LogisticRegression(max_iter=10000)
                            time_regressor.fit(train_features_time, train_labels)
                            train_pred_time = time_regressor.predict_proba(train_features_time)[:, 1]
                            test_pred_time = time_regressor.predict_proba(test_features_time)[:, 1]
                            train_pred_time_class = time_regressor.predict(train_features_time)
                            test_pred_time_class = time_regressor.predict(test_features_time)
                            train_r_squared_time += sklearn.metrics.r2_score(train_labels, train_pred_time)
                            test_r_squared_time += sklearn.metrics.r2_score(test_labels, test_pred_time)
                            train_r_time += np.corrcoef(train_labels, train_pred_time)[0, 1]
                            test_r_time += np.corrcoef(test_labels, test_pred_time)[0, 1]
                            train_roc_time += sklearn.metrics.roc_auc_score(train_labels, train_pred_time)
                            test_roc_time += sklearn.metrics.roc_auc_score(test_labels, test_pred_time)
                            train_acc_time += sklearn.metrics.accuracy_score(train_labels, train_pred_time_class)
                            test_acc_time += sklearn.metrics.accuracy_score(test_labels, test_pred_time_class)
                        else:
                            # Fit linear regression for electrode features
                            electrode_regressor = sklearn.linear_model.LinearRegression()
                            electrode_regressor.fit(train_features_electrode, train_labels)
                            train_pred_electrode = electrode_regressor.predict(train_features_electrode)
                            test_pred_electrode = electrode_regressor.predict(test_features_electrode)
                            train_r_squared_electrode += sklearn.metrics.r2_score(train_labels, train_pred_electrode)
                            test_r_squared_electrode += sklearn.metrics.r2_score(test_labels, test_pred_electrode)
                            train_r_electrode += np.corrcoef(train_labels, train_pred_electrode)[0, 1]
                            test_r_electrode += np.corrcoef(test_labels, test_pred_electrode)[0, 1]

                            # Fit linear regression for time features
                            time_regressor = sklearn.linear_model.LinearRegression()
                            time_regressor.fit(train_features_time, train_labels)
                            train_pred_time = time_regressor.predict(train_features_time)
                            test_pred_time = time_regressor.predict(test_features_time)
                            train_r_squared_time += sklearn.metrics.r2_score(train_labels, train_pred_time)
                            test_r_squared_time += sklearn.metrics.r2_score(test_labels, test_pred_time)
                            train_r_time += np.corrcoef(train_labels, train_pred_time)[0, 1]
                            test_r_time += np.corrcoef(test_labels, test_pred_time)[0, 1]

                    # Divide by number of dataloaders to get mean metrics
                    n_dataloaders = len(eval_dataloaders)
                    train_r_squared_electrode /= n_dataloaders
                    train_r_squared_time /= n_dataloaders
                    test_r_squared_electrode /= n_dataloaders
                    test_r_squared_time /= n_dataloaders
                    train_acc_electrode /= n_dataloaders
                    train_acc_time /= n_dataloaders
                    test_acc_electrode /= n_dataloaders
                    test_acc_time /= n_dataloaders
                    train_roc_electrode /= n_dataloaders
                    train_roc_time /= n_dataloaders
                    test_roc_electrode /= n_dataloaders
                    test_roc_time /= n_dataloaders
                    train_r_electrode /= n_dataloaders
                    train_r_time /= n_dataloaders
                    test_r_electrode /= n_dataloaders
                    test_r_time /= n_dataloaders

                    # calculate mean norm of features
                    train_features_electrode_norm = np.linalg.norm(train_features_electrode, axis=1).mean()
                    print(f"Train electrode features norm: {train_features_electrode_norm:.4f}")
                    if not training_config['binarize_eval']:
                        print(f"Electrode -- Train R2: {train_r_squared_electrode:.4f} (R: {train_r_electrode:.4f}) -- Test R2: {test_r_squared_electrode:.4f} (R: {test_r_electrode:.4f}) -- "
                            f"Time -- Train R2: {train_r_squared_time:.4f} (R: {train_r_time:.4f}) -- Test R2: {test_r_squared_time:.4f} (R: {test_r_time:.4f})")
                    else:
                        print(f"Electrode -- Train AUC: {train_roc_electrode:.4f} (R2: {train_r_squared_electrode:.4f}, R: {train_r_electrode:.4f}, Acc: {train_acc_electrode:.4f}) -- Test AUC: {test_roc_electrode:.4f} (R2: {test_r_squared_electrode:.4f}, R: {test_r_electrode:.4f}, Acc: {test_acc_electrode:.4f}) -- "
                              f"Time -- Train AUC: {train_roc_time:.4f} (R2: {train_r_squared_time:.4f}, R: {train_r_time:.4f}, Acc: {train_acc_time:.4f}) -- Test AUC: {test_roc_time:.4f} (R2: {test_r_squared_time:.4f}, R: {test_r_time:.4f}, Acc: {test_acc_time:.4f})")


            print(f"Batch {overall_batch_i+1}/{training_config['total_steps']} -- {subject_trial} -- epoch {epoch_i+1}/{training_config['n_epochs']} -- Loss: {loss.item():.4f} -- Avg distance: {avg_distance:.4f} -- GPU mem: {gpu_mem_used:.0f}MB -- Time left: {time_str} -- Current time: {current_time_str}s -- Temp param: {temp_clip_param.item():.4f}")
            if wandb_log:
                log_dict = {
                    "loss": loss.item(),
                    "gradient_norm": gradient_norm.item(),
                    "avg_distance": avg_distance,
                    "gpu_mem_used": gpu_mem_used,
                    "temp_clip_param": temp_clip_param.item(),
                }
                if overall_test_loss is not None:
                    log_dict['test_loss'] = overall_test_loss
                if to_save_eval:
                    # log_dict['eval/train_r2_electrode'] = train_r_squared_electrode
                    # log_dict['eval/test_r2_electrode'] = test_r_squared_electrode
                    # log_dict['eval/train_r2_time'] = train_r_squared_time
                    # log_dict['eval/test_r2_time'] = test_r_squared_time

                    # log_dict['eval/train_r_electrode'] = train_r_electrode
                    # log_dict['eval/test_r_electrode'] = test_r_electrode
                    # log_dict['eval/train_r_time'] = train_r_time
                    # log_dict['eval/test_r_time'] = test_r_time

                    if training_config['binarize_eval']:
                        #log_dict['eval/train_roc_electrode'] = train_roc_electrode
                        log_dict['eval/test_roc_electrode'] = test_roc_electrode
                        #log_dict['eval/train_roc_time'] = train_roc_time
                        log_dict['eval/test_roc_time'] = test_roc_time
                        #log_dict['eval/train_acc_electrode'] = train_acc_electrode
                        log_dict['eval/test_acc_electrode'] = test_acc_electrode
                        #log_dict['eval/train_acc_time'] = train_acc_time
                        log_dict['eval/test_acc_time'] = test_acc_time
                wandb.log(log_dict, step=overall_batch_i+1, commit=(overall_batch_i+1) % training_config['wandb_commit_every_n_batches'] == 0)#, **loss_per_electrode)

            loss_store.append(loss.item())
            epoch_batch_store.append((epoch_i, batch_i))
            subject_trial_store.append(subject_trial)
            gradient_norm_store.append(gradient_norm.item())
            if overall_test_loss is not None: test_loss_store.append(overall_test_loss)

            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()

            # Save losses every 20 batches
            if (overall_batch_i+1) % training_config['save_losses_every_n_batches'] == 0:
                # Convert dtype and device to strings for JSON serialization
                json_transformer_config = transformer_config.copy()
                json_transformer_config['dtype'] = str(transformer_config['dtype'])
                json_transformer_config['device'] = str(transformer_config['device'])
                with open(f'{dir_name}/metadata.json', 'w') as f:
                    json.dump({
                        'transformer_config': json_transformer_config,
                        'training_config': training_config,
                        }, f, indent=4)
                with open(f'{dir_name}/training_dynamics.json', 'w') as f:
                    json.dump({
                        'losses': loss_store,
                        'epoch_batch_store': epoch_batch_store,
                        'subject_trial_store': subject_trial_store,
                        'gradient_norm_store': gradient_norm_store,
                        'test_losses': test_loss_store,
                        'avg_distance_store': avg_distance_store,
                        }, f, indent=4)
                torch.save(electrode_transformer.state_dict(), f'{dir_name}/model_electrode_state_dict.pth')
                torch.save(time_transformer.state_dict(), f'{dir_name}/model_time_state_dict.pth')
                torch.save(dataloader.subject_electrode_emb_store, f'{dir_name}/subject_electrode_embeddings.pth')
                print(f"Saved losses, model, and electrode embeddings after batch {overall_batch_i+1}")
        # Save model for this epoch
        if (epoch_i + 1) % training_config['save_network_every_n_epochs'] == 0:
            torch.save(electrode_transformer.state_dict(), f'{dir_name}/model_electrode_state_dict_epoch{epoch_i+1}.pth')
            torch.save(time_transformer.state_dict(), f'{dir_name}/model_time_state_dict_epoch{epoch_i+1}.pth')
            torch.save(dataloader.subject_electrode_emb_store, f'{dir_name}/subject_electrode_embeddings_epoch{epoch_i+1}.pth')
            print(f"Saved model and electrode embeddings checkpoint for epoch {epoch_i+1}")
    if wandb_log:
        wandb.finish()