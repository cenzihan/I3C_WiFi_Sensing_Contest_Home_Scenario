import yaml
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

# Get the absolute path to the project root directory
# This makes path handling robust and independent of the current working directory
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(UTILS_DIR)

def load_config(config_path):
    """Loads the configuration from a YAML file."""
    # Assume config_path is relative to the project root
    abs_config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(abs_config_path, 'r') as file:
        return yaml.safe_load(file)

def get_data_files(data_path, scenario):
    """Gets all data and truth file paths, assuming data_path is relative to project root."""
    abs_data_path = os.path.join(PROJECT_ROOT, data_path)
    data_dir = os.path.join(abs_data_path, scenario, 'data')
    
    room_A_files = sorted(glob.glob(os.path.join(data_dir, 'room_A', 'csi_*.txt')))
    room_B_files = sorted(glob.glob(os.path.join(data_dir, 'room_B', 'csi_*.txt')))
    
    assert len(room_A_files) == len(room_B_files), "Mismatch in number of data files between room_A and room_B"
    
    return list(zip(room_A_files, room_B_files))

class CSIDataset(Dataset):
    def __init__(self, data_path, scenario, sequence_length):
        self.scenario = scenario
        self.sequence_length = sequence_length
        print("Creating CSI Dataset...")

        self.data_files = get_data_files(data_path, scenario)
        self.samples = self._create_samples()

        print(f"Dataset created successfully with {len(self.samples)} samples.")
        
    def _create_samples(self):
        samples = []
        num_files = len(self.data_files)
        print(f"  > Found {num_files} data file pairs to process.")

        for i, (room_a_file, room_b_file) in enumerate(self.data_files):
            if (i+1) % 10 == 0:
                 print(f"    - Processing file {i+1}/{num_files}...")
            
            base_name = os.path.basename(room_a_file)
            truth_base = base_name.replace('.txt', '_truth.txt')
            
            # Since data_files now contains absolute paths, we can derive the truth path robustly
            data_root_for_truth = os.path.dirname(os.path.dirname(os.path.dirname(room_a_file))) # Navigates up to '.../home_scenario_1/'
            truth_root = os.path.join(data_root_for_truth, 'truth')
            
            truth_path_a = os.path.join(truth_root, 'room_A', truth_base)
            truth_path_b = os.path.join(truth_root, 'room_B', truth_base)
            parlor_truth_file = os.path.join(truth_root, 'parlor', truth_base)

            csi_a = self._load_csi_data(room_a_file)
            csi_b = self._load_csi_data(room_b_file)

            if csi_a.size == 0 or csi_b.size == 0:
                continue

            min_len = min(len(csi_a), len(csi_b))
            csi_combined = np.concatenate([csi_a[:min_len], csi_b[:min_len]], axis=1)

            truth_a = self._load_truth_data(truth_path_a)
            truth_b = self._load_truth_data(truth_path_b)
            truth_parlor = self._load_truth_data(parlor_truth_file)

            if not truth_a or not truth_b or not truth_parlor:
                continue

            if not (len(truth_a) == len(truth_b) == len(truth_parlor)):
                print(f"Warning: Skipping {base_name} due to mismatched truth lengths.")
                continue

            num_windows = len(truth_a)
            if num_windows == 0:
                continue

            packets_per_window = len(csi_combined) // num_windows
            if packets_per_window == 0:
                continue

            for j in range(num_windows):
                start_idx = j * packets_per_window
                end_idx = start_idx + packets_per_window
                
                window_csi = csi_combined[start_idx:end_idx]
                
                if len(window_csi) >= self.sequence_length:
                    seq_csi = window_csi[:self.sequence_length]
                else:
                    padding_len = self.sequence_length - len(window_csi)
                    padding = np.zeros((padding_len, window_csi.shape[1]), dtype=np.float32)
                    seq_csi = np.concatenate([window_csi, padding], axis=0)

                label = torch.tensor([truth_a[j], truth_b[j], truth_parlor[j]], dtype=torch.long)
                samples.append((torch.from_numpy(seq_csi), label))

        return samples

    def _load_csi_data(self, file_path):
        """
        Loads CSI data from a single file, correctly parsing the packet structure.
        Packet format for scenario 1: 8 metadata ints + 500 complex numbers.
        Packet format for scenario 2: 8 metadata ints + 496 complex numbers.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: CSI data file not found at {file_path}")
            return np.array([], dtype=np.float32)

        all_elements = content.strip().split()
        if not all_elements:
            return np.array([], dtype=np.float32)

        if self.scenario == "home_scenario_1":
            elements_per_packet = 8 + 500
        elif self.scenario == "home_scenario_2":
            elements_per_packet = 8 + 496
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        num_packets = len(all_elements) // elements_per_packet
        
        if num_packets == 0:
            return np.array([], dtype=np.float32)

        packets = np.array(all_elements[:num_packets * elements_per_packet]).reshape(num_packets, elements_per_packet)
        
        csi_complex_strings = packets[:, 8:]

        all_csi_floats = []
        for packet_strings in csi_complex_strings:
            try:
                complex_numbers = np.array([complex(s.replace('i', 'j')) for s in packet_strings], dtype=np.complex64)
                float_features = np.stack([complex_numbers.real, complex_numbers.imag], axis=-1).flatten().astype(np.float32)
                all_csi_floats.append(float_features)
            except ValueError:
                print(f"Warning: Corrupted CSI data in a packet in {os.path.basename(file_path)}. Skipping packet.")
                continue
        
        if not all_csi_floats:
             print(f"Warning: No valid CSI packets could be parsed from {os.path.basename(file_path)}.")
             return np.array([], dtype=np.float32)

        return np.vstack(all_csi_floats)
        
    def _load_truth_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
        except FileNotFoundError:
            return []
        if not content:
            return []
        
        return [int(p) for p in content.split() if p]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]