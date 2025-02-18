import os
import sys
from pathlib import Path

# Add the local spikeinterface and simsort paths to Python path
SPIKEINTERFACE_PATH = Path("/home/v-yimuzhang/spikeinterface/src")
SIMSORT_PATH = SPIKEINTERFACE_PATH / "spikeinterface/sorters/simsortsrc"

if str(SPIKEINTERFACE_PATH) not in sys.path:
    sys.path.insert(0, str(SPIKEINTERFACE_PATH))
if str(SIMSORT_PATH) not in sys.path:
    sys.path.insert(0, str(SIMSORT_PATH))

import numpy as np
from spikeinterface import generate
from spikeinterface.sorters import run_sorter

# Clean up existing test directory
test_dir = Path('test_simsort')
if test_dir.exists():
    import shutil
    shutil.rmtree(test_dir)

# Generate a synthetic recording
recording, sorting_true = generate.generate_ground_truth_recording(
    num_channels=4,
    sampling_frequency=30000,
    durations=[30.0],  # 30 seconds
    num_units=8
)

# Run SimSort with correct parameters
sorting = run_sorter(
    'simsort',
    recording,
    folder='test_simsort',
    yaml_path=str(SIMSORT_PATH / 'SimSort.yaml'),
    verbose=True,
)

print("Sorting completed successfully!")
print(f"Found {len(sorting.unit_ids)} units")