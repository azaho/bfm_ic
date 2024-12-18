import numpy as np

all_subject_trials = []
for sub_id in range(1, 11):
    for trial_id in np.arange([3, 7, 3, 3, 1, 3, 2, 1, 1, 2][sub_id-1]):
        all_subject_trials.append((sub_id, trial_id))

print(all_subject_trials)

#####

from braintreebank_process_chunks import *

for sub_id, trial_id in all_subject_trials:
    print(f"Processing subject {sub_id} trial {trial_id}")
    subject = Subject(sub_id)
    subject.load_neural_data(trial_id)
    subject.check_electrodes(trial_id)
    subject.close_all_files()
    del subject