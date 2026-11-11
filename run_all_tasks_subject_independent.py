from pathlib import Path
from all_tasks_adaptive_sparse_common import SEED, seed_everything, run_independent

seed_everything()
run_independent(
    subject_ids=range(1, 110),
    output_dir=Path('eeg-motor-movementimagery-dataset-1.0.0/outputs_all_tasks_independent'),
    n_channels=16,
)
