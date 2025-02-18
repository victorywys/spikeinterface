from __future__ import annotations
import os
import sys
import numpy as np
from pathlib import Path
from spikeinterface.core import NumpySorting

# Add all necessary paths
SIMSORT_PATH = Path("/home/v-yimuzhang/spikeinterface/src/spikeinterface/sorters/simsortsrc")
TASK_PATH = SIMSORT_PATH / "task"
UTILS_PATH = SIMSORT_PATH / "utils"
MODEL_PATH = SIMSORT_PATH / "model"

# Add all paths to Python path at module import time
for path in [SIMSORT_PATH, TASK_PATH, UTILS_PATH, MODEL_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Try importing required modules to verify paths are correct
try:
    import custom_sorting
    import tool
    from model import model_detector, model_extractor
except ImportError as e:
    raise ImportError(f"Failed to import SimSort modules: {e}")

from ...core import load
from ...extractors import NpzSortingExtractor
from ..basesorter import BaseSorter

class SimSortSorter(BaseSorter):
    """SimSort sorter wrapper."""

    sorter_name = "simsort"
    requires_locations = False
    gpu_capability = "supported"
    requires_binary_data = True
    compatible_with_parallel = {"loky": True, "multiprocessing": True, "threading": True}

    _default_params = {
        "yaml_path": str(SIMSORT_PATH / "SimSort.yaml"),
        "model_path": str(SIMSORT_PATH / "simsort_pretrained"),
        "cluster_method": "MS",  # MeanShift clustering
        "n_clusters": None,  # For methods that need predefined clusters
        "ms_quantile": 0.12,  # For MeanShift clustering
        "output_folder": None,
        "recording_folder": None,
    }

    _params_description = {
        "yaml_path": "Path to the YAML configuration file",
        "model_path": "Path to pretrained SimSort models",
        "cluster_method": "Clustering method (MS, KMeans, GMM, etc.)",
        "n_clusters": "Number of clusters for methods that need it predefined",
        "ms_quantile": "Quantile for MeanShift clustering",
    }

    sorter_description = "SimSort: A deep learning spike sorter"
    installation_mesg = "SimSort is using the local version from src/spikeinterface/sorters/simsort"

    @classmethod
    def is_installed(cls):
        try:
            # Try basic imports
            import custom_sorting
            import tool
            from model import model_detector, model_extractor
            
            # Check for required files
            if not (SIMSORT_PATH / "SimSort.yaml").exists():
                return False
            if not (SIMSORT_PATH / "simsort_pretrained").exists():
                return False
                
            return True
        except ImportError:
            return False

    @classmethod
    def get_sorter_version(cls):
        return "local"

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        return params

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        # Create recording folder
        recording_folder = sorter_output_folder / "recording"
        recording_folder.mkdir(parents=True, exist_ok=True)
        
        # Save recording to binary file with overwrite=True
        recording.save(folder=recording_folder, overwrite=True)
        
        # Update parameters with correct paths
        params["recording_folder"] = str(recording_folder)
        params["output_folder"] = str(sorter_output_folder)
        
        # Save parameters for debugging
        if verbose:
            print(f"Recording folder: {params['recording_folder']}")
            print(f"Output folder: {params['output_folder']}")

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import custom_sorting
        
        sorter_output_folder = Path(sorter_output_folder)
        recording_folder = sorter_output_folder / "recording"
        
        if "recording_folder" not in params:
            params["recording_folder"] = str(recording_folder)
            
        recording = load(recording_folder)
        sorting_task = custom_sorting.SortingTask(
            root_path=SIMSORT_PATH,
            yaml_path=params.get("yaml_path"),
            recording=recording,
            cluster_method=params.get("cluster_method"),
            dmodel_save_path=str(SIMSORT_PATH / "simsort_pretrained/detector_bbp_L1-L5-8192/saved_models"),
            emodel_save_path=str(SIMSORT_PATH / "simsort_pretrained/extractor_bbp_L1-L5-8192/saved_models"),
            n_clusters=params.get("n_clusters"),
            ms_quantile=params.get("ms_quantile"),
            verbose=params.get("verbose", True)
        )
        
        # Run sorting
        sorting_task.get_data()
        peak_positions, snippets = sorting_task.detect_spike(sorting_task.test_data)
        labels_list, _ = sorting_task.cluster()
        
        # Convert results to spikeinterface format
        sampling_frequency = recording.get_sampling_frequency()
        
        all_spikes = []
        all_labels = []
        
        if len(peak_positions) > 0 and len(peak_positions[0]) > 0:
            spikes = peak_positions[0]
            labels = labels_list[0]
            all_spikes.extend(spikes)
            all_labels.extend(labels)

        all_spikes = np.array(all_spikes, dtype=np.int64)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        # generate unit dict{unit_id: spike_times}
        spike_train = {}
        unique_labels = np.unique(all_labels)
        for unit_id in unique_labels:
            unit_mask = all_labels == unit_id
            spike_train[int(unit_id)] = all_spikes[unit_mask].astype(np.int64)
        
        # use NumpySorting to directly save sorting
        sorting = NumpySorting.from_unit_dict(spike_train, sampling_frequency)
        sorting.save(folder=sorter_output_folder / "output")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        output_folder = sorter_output_folder / "output"
        
        if not output_folder.exists():
            raise FileNotFoundError(f"Results not found in {output_folder}")
        
        # load sorting results from folder
        sorting = load(output_folder)
        return sorting