from pathlib import Path
from adaptive_sparse_fbcsp import seed_everything, run_independent

seed_everything()
run_independent(range(1, 110), Path('eeg-motor-movementimagery-dataset-1.0.0/outputs_all_tasks_fbcsp_independent'), n_channels=16)
