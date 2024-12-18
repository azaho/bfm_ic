import numpy as np

# all subject trials including the special case for subject 6 which only has trials 0, 1, and 4
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

#####

from braintreebank_process_chunks import *

for sub_id, trial_id in all_subject_trials:
    print(f"Processing subject {sub_id} trial {trial_id}")
    subject = Subject(sub_id)
    subject.load_neural_data(trial_id)
    subject.check_electrodes(trial_id)
    subject.close_all_files()
    del subject