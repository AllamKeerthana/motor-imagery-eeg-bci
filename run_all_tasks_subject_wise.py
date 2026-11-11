from pathlib import Path
from all_tasks_adaptive_sparse_common import seed_everything, run_subject_wise

seed_everything()
run_subject_wise(
    subject_ids=range(1, 21),
    output_dir=Path('eeg-motor-movementimagery-dataset-1.0.0/outputs_all_tasks_subject_wise'),
    n_channels=16,
)
