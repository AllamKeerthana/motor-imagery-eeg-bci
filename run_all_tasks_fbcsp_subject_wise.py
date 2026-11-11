from pathlib import Path
from adaptive_sparse_fbcsp import seed_everything, run_subject_wise

seed_everything()
run_subject_wise(range(1, 21), Path('eeg-motor-movementimagery-dataset-1.0.0/outputs_all_tasks_fbcsp_subject_wise'), n_channels=16)
