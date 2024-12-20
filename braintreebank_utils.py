import numpy as np

# all subject trials including the special case for subject 6 which only has trials 0, 1, and 4
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

#####

# Dictionary to store MSE values for each subject
subject_mse_dict = {}

for sub_id, trial_id in all_subject_trials:
    persistence_mse_array = []
    for chunk_id in range(10):
        chunk_path = f"braintreebank_data_chunks/subject{sub_id}_trial{trial_id}_chunk{chunk_id}.npy"
        chunk_data = np.load(chunk_path) # (n_electrodes, n_time_bins, n_freq_features)
        persistence_mse = np.mean((chunk_data[:, 1:, :] - chunk_data[:, :-1, :])**2)
        persistence_mse_array.append(persistence_mse)
    trial_mean = np.mean(persistence_mse_array)
    print(f"Subject {sub_id} trial {trial_id} mean persistence MSE: {trial_mean}")
    
    # Add to subject dictionary
    if sub_id not in subject_mse_dict:
        subject_mse_dict[sub_id] = []
    subject_mse_dict[sub_id].append(trial_mean)
# Print mean for each subject
print("\nMean persistence MSE per subject:")
all_means = []
for sub_id in sorted(subject_mse_dict.keys()):
    subject_mean = np.mean(subject_mse_dict[sub_id])
    print(f"Subject {sub_id}: {subject_mean}")
    all_means.append(subject_mean)
# Print overall mean
print(f"\nOverall mean persistence MSE: {np.mean(all_means)}")

# Dictionary to store mean MSE values for each subject
subject_mean_mse_dict = {}

for sub_id, trial_id in all_subject_trials:
    mean_mse_array = []
    for chunk_id in range(10):
        chunk_path = f"braintreebank_data_chunks/subject{sub_id}_trial{trial_id}_chunk{chunk_id}.npy"
        chunk_data = np.load(chunk_path) # (n_electrodes, n_time_bins, n_freq_features)
        
        # Calculate mean per electrode across time
        electrode_means = np.mean(chunk_data, axis=1, keepdims=True)  # (n_electrodes, 1, n_freq_features)
        
        # Calculate MSE between actual values and mean prediction
        mean_mse = np.mean((chunk_data - electrode_means)**2)
        mean_mse_array.append(mean_mse)
        
    trial_mean = np.mean(mean_mse_array)
    print(f"Subject {sub_id} trial {trial_id} mean MSE using electrode means: {trial_mean}")
    
    # Add to subject dictionary
    if sub_id not in subject_mean_mse_dict:
        subject_mean_mse_dict[sub_id] = []
    subject_mean_mse_dict[sub_id].append(trial_mean)

# Print mean for each subject
print("\nMean MSE per subject (using electrode means):")
all_means = []
for sub_id in sorted(subject_mean_mse_dict.keys()):
    subject_mean = np.mean(subject_mean_mse_dict[sub_id])
    print(f"Subject {sub_id}: {subject_mean}")
    all_means.append(subject_mean)

# Print overall mean
print(f"\nOverall mean MSE using electrode means: {np.mean(all_means)}")

# Now calculate MSE using global mean across all electrodes
subject_global_mean_mse_dict = {}

for sub_id, trial_id in all_subject_trials:
    global_mean_mse_array = []
    for chunk_id in range(10):
        chunk_path = f"braintreebank_data_chunks/subject{sub_id}_trial{trial_id}_chunk{chunk_id}.npy"
        chunk_data = np.load(chunk_path)
        
        # Calculate global mean across all electrodes and time
        global_mean = np.mean(chunk_data, axis=(0,1), keepdims=True)  # (1, 1, n_freq_features)
        
        # Calculate MSE between actual values and global mean prediction
        global_mean_mse = np.mean((chunk_data - global_mean)**2)
        global_mean_mse_array.append(global_mean_mse)
        
    trial_mean = np.mean(global_mean_mse_array)
    print(f"Subject {sub_id} trial {trial_id} mean MSE using global mean: {trial_mean}")
    
    if sub_id not in subject_global_mean_mse_dict:
        subject_global_mean_mse_dict[sub_id] = []
    subject_global_mean_mse_dict[sub_id].append(trial_mean)

# Print mean for each subject
print("\nMean MSE per subject (using global mean):")
all_global_means = []
for sub_id in sorted(subject_global_mean_mse_dict.keys()):
    subject_mean = np.mean(subject_global_mean_mse_dict[sub_id])
    print(f"Subject {sub_id}: {subject_mean}")
    all_global_means.append(subject_mean)

# Print overall mean
print(f"\nOverall mean MSE using global mean: {np.mean(all_global_means)}")


#####

from braintreebank_process_chunks import *

for sub_id, trial_id in all_subject_trials:
    print(f"Processing subject {sub_id} trial {trial_id}")
    subject = Subject(sub_id)
    subject.load_neural_data(trial_id)
    subject.check_electrodes(trial_id)
    subject.close_all_files()
    del subject