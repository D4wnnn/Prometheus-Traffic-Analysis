"""
Multi-modal Protocol Understanding Task Dataset
Includes 5 tasks:
1. Direction Prediction
2. IAT Anomaly Detection
3. Size Sequence Prediction
4. TCP Request-Response Matching
5. Cross-Modal Consistency
"""

import torch
from torch.utils.data import Dataset
import h5py
import random
import numpy as np


class MultiModalProbeDataset(Dataset):
    """
    Dataset for multi-modal protocol understanding tasks.
    
    Args:
        h5_file: Path to the HDF5 data file.
        task_type: Type of the evaluation task.
            - 'direction': Direction Prediction
            - 'iat_anomaly': IAT Anomaly Detection
            - 'size_pred': Packet Size Prediction
            - 'tcp_match': TCP Request-Response Matching
            - 'consistency': Cross-modal Consistency Check
            - 'masked_recovery': Context Completion (Legacy task)
    """
    def __init__(self, h5_file, task_type='direction', max_bytes=None, max_packets=None):
        self.h5_file = h5_file
        self.task_type = task_type
        self.max_bytes = max_bytes
        self.max_packets = max_packets
        
        with h5py.File(h5_file, 'r') as f:
            self.length = f['bytes_nlp_view'].shape[0]
            self.has_tcp = 'tcp_data_view' in f
        
        # Packet size bins (for size_pred task)
        self.size_bins = [0, 64, 128, 256, 512, 768, 1024, 1280, 1400, 1500]
        self.num_size_bins = len(self.size_bins)

    def __len__(self):
        return self.length

    def _load_and_pad(self, idx):
        """Load data and unify sequence length."""
        with h5py.File(self.h5_file, 'r') as f:
            bytes_nlp = f['bytes_nlp_view'][idx]  # (N, L)
            seq_nlp = f['seq_nlp_view'][idx]      # (N, 3)
            tcp_data = f['tcp_data_view'][idx]    # (N, 3)
            label = f['labels'][idx]
        
            # Truncate to teacher view size
            bytes_nlp = torch.from_numpy(bytes_nlp[:self.max_packets, :self.max_bytes]).long()
            seq_nlp = torch.from_numpy(seq_nlp[:self.max_packets]).float()
            if self.has_tcp:
                tcp_data = tcp_data[:self.max_packets]
                tcp_data = torch.from_numpy(tcp_data).float()
            else:
                tcp_data = torch.zeros(seq_nlp.shape[0], 3)
        
        # Calculate valid length
        valid_len = (seq_nlp[:, 0] != 0).sum().item()
        
        return bytes_nlp, seq_nlp, tcp_data, valid_len

    def __getitem__(self, idx):
        if self.task_type == 'direction':
            return self._create_direction_sample(idx)
        elif self.task_type == 'iat_anomaly':
            return self._create_iat_anomaly_sample_v2(idx)
        elif self.task_type == 'size_pred':
            return self._create_size_prediction_sample_v2(idx)
        elif self.task_type == 'tcp_match':
            return self._create_tcp_matching_sample(idx)
        elif self.task_type == 'consistency':
            return self._create_consistency_sample(idx)
        elif self.task_type == 'masked_recovery':
            return self._create_masked_recovery_sample(idx)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    # ==========================================
    # Task 1: Direction Prediction
    # ==========================================
    def _create_direction_sample(self, idx):
        """
        Given the complete features of the first N-1 packets, predict the direction of the N-th packet.
        Requires the model to understand protocol interaction patterns (e.g., Request-Response).
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        # Requires at least 2 valid packets
        if valid_len < 2:
            # Return a dummy sample to be filtered during training
            return self._create_dummy_direction_sample(bytes_nlp, seq_nlp, tcp_data)
        
        # Determine prediction position (the last valid packet)
        pred_idx = min(valid_len - 1, self.max_packets - 1)
        
        # Target direction: OUT=1 -> label=0, IN=-1 -> label=1
        target_dir = seq_nlp[pred_idx, 1].item()
        if target_dir > 0:
            label = 0  # OUT
        elif target_dir < 0:
            label = 1  # IN
        else:
            label = 2  # PADDING (should not occur for valid packets)
        
        # Input: first pred_idx packets (excluding the one to be predicted)
        # Zero out features at the prediction index to simulate "unknown"
        input_bytes = bytes_nlp.clone()
        input_seq = seq_nlp.clone()
        input_tcp = tcp_data.clone()
        
        # Mask the prediction position
        input_bytes[pred_idx] = 0
        input_seq[pred_idx] = 0
        input_tcp[pred_idx] = 0
        
        return {
            'bytes_nlp': input_bytes,
            'seq_nlp': input_seq,
            'tcp_data': input_tcp,
            'pred_idx': torch.tensor(pred_idx).long(),
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()  # Mark as valid sample
        }
    
    def _create_dummy_direction_sample(self, bytes_nlp, seq_nlp, tcp_data):
        """Create an invalid dummy sample."""
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'pred_idx': torch.tensor(0).long(),
            'label': torch.tensor(0).long(),
            'valid': torch.tensor(0).long()
        }

    # ==========================================
    # Task: Protocol Pattern Recognition
    # ==========================================
    def _create_pattern_recognition_sample(self, idx):
        """
        Identify flow interaction patterns:
        - Pattern A: Request-Response (OUT-IN-OUT-IN...)
        - Pattern B: Bulk Transfer (OUT-OUT-OUT... or IN-IN-IN...)
        - Pattern C: Mixed Mode
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 3:
            return self._create_dummy_pattern_sample(bytes_nlp, seq_nlp, tcp_data)
        
        # Analyze direction sequence
        directions = seq_nlp[:valid_len, 1].numpy()
        
        # Calculate pattern features
        alternating_count = 0
        for i in range(1, len(directions)):
            if directions[i] * directions[i-1] < 0:  # Direction changes
                alternating_count += 1
        
        alternating_ratio = alternating_count / (len(directions) - 1) if len(directions) > 1 else 0
        
        # Label: 0=Bulk (low alternation), 1=Req-Resp (high alternation), 2=Mixed
        if alternating_ratio > 0.7:
            label = 1  # Request-Response pattern
        elif alternating_ratio < 0.3:
            label = 0  # Bulk transfer pattern
        else:
            label = 2  # Mixed pattern
        
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }
    
    # ==========================================
    # Task: Burst Detection
    # ==========================================
    def _create_burst_detection_sample(self, idx):
        """
        Detect if a burst exists in the flow.
        Burst definition: multiple consecutive packets with very small IAT.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 4:
            return self._create_dummy_sample()
        
        iats = seq_nlp[1:valid_len, 2].numpy()
        
        # Detect burst: 3 or more consecutive packets with IAT < threshold
        threshold = np.median(iats) * 0.1 if len(iats) > 0 else 0.001
        
        burst_detected = False
        consecutive_small = 0
        for iat in iats:
            if iat < threshold:
                consecutive_small += 1
                if consecutive_small >= 3:
                    burst_detected = True
                    break
            else:
                consecutive_small = 0
        
        label = 1 if burst_detected else 0
        
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }
    
    # ==========================================
    # Task: Flow Ordering
    # ==========================================
    def _create_flow_ordering_sample(self, idx):
        """
        Determine the correct temporal order of shuffled packets.
        Binary classification: given two packets, judge which was sent first.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 3:
            return self._create_dummy_ordering_sample(bytes_nlp)
        
        # Randomly select two distinct positions
        pos_a = random.randint(0, valid_len - 2)
        pos_b = random.randint(pos_a + 1, valid_len - 1)
        
        # Randomly decide whether to swap (50% probability)
        if random.random() < 0.5:
            # Normal order: A before B
            first_idx, second_idx = pos_a, pos_b
            label = 0  # "first" is indeed before "second"
        else:
            # Swapped order: B before A
            first_idx, second_idx = pos_b, pos_a
            label = 1  # "first" is actually after "second"
        
        return {
            'first_bytes': bytes_nlp[first_idx],
            'first_seq': seq_nlp[first_idx],
            'first_tcp': tcp_data[first_idx],
            'second_bytes': bytes_nlp[second_idx],
            'second_seq': seq_nlp[second_idx],
            'second_tcp': tcp_data[second_idx],
            'context_bytes': bytes_nlp,  # Full context
            'context_seq': seq_nlp,
            'context_tcp': tcp_data,
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }

    # ==========================================
    # Task 2: IAT Anomaly Detection
    # ==========================================
    def _create_iat_anomaly_sample(self, idx):
        """
        Detect anomalies in Inter-Arrival Time (IAT).
        Positive (label=1): Original flow, normal IAT.
        Negative (label=0): IAT of a packet has been tampered with.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 2:
            return self._create_dummy_anomaly_sample(bytes_nlp, seq_nlp, tcp_data)
        
        label = 1  # Default to normal
        
        # 50% chance to create an anomalous sample
        if random.random() < 0.5:
            label = 0
            # Choose anomaly position (avoiding first packet and padding)
            anomaly_idx = random.randint(1, valid_len - 1)
            original_iat = seq_nlp[anomaly_idx, 2].item()
            
            # Anomaly type: excessively large or small
            anomaly_type = random.choice(['too_small', 'too_large'])
            if anomaly_type == 'too_small':
                factor = random.uniform(0.001, 0.1)
            else:
                factor = random.uniform(10, 100)
            
            seq_nlp[anomaly_idx, 2] = original_iat * factor
        
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }
    
    def _create_dummy_anomaly_sample(self, bytes_nlp, seq_nlp, tcp_data):
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(1).long(),
            'valid': torch.tensor(0).long()
        }

    # ==========================================
    # Task 3: Size Sequence Prediction
    # ==========================================
    def _create_size_prediction_sample(self, idx):
        """
        Given the first N-1 packets, predict the size bin of the N-th packet.
        Multi-class task, number of classes = len(size_bins).
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 2:
            return self._create_dummy_size_sample(bytes_nlp, seq_nlp, tcp_data)
        
        pred_idx = min(valid_len - 1, self.max_packets - 1)
        target_size = seq_nlp[pred_idx, 0].item()
        
        # Discretize size into bins
        target_bin = np.digitize(target_size, self.size_bins) - 1
        target_bin = min(max(target_bin, 0), self.num_size_bins - 1)
        
        # Mask prediction position
        input_bytes = bytes_nlp.clone()
        input_seq = seq_nlp.clone()
        input_tcp = tcp_data.clone()
        
        input_bytes[pred_idx] = 0
        input_seq[pred_idx] = 0
        input_tcp[pred_idx] = 0
        
        return {
            'bytes_nlp': input_bytes,
            'seq_nlp': input_seq,
            'tcp_data': input_tcp,
            'pred_idx': torch.tensor(pred_idx).long(),
            'label': torch.tensor(target_bin).long(),
            'valid': torch.tensor(1).long()
        }
    
    def _create_dummy_size_sample(self, bytes_nlp, seq_nlp, tcp_data):
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'pred_idx': torch.tensor(0).long(),
            'label': torch.tensor(0).long(),
            'valid': torch.tensor(0).long()
        }

    # ==========================================
    # Task 4: TCP Request-Response Matching
    # ==========================================
    def _create_tcp_matching_sample(self, idx):
        """
        Given a request packet and two candidate response packets, determine the correct response.
        Requires understanding TCP seq/ack semantics and temporal relationships.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if not self.has_tcp or valid_len < 4:
            return self._create_dummy_tcp_sample(bytes_nlp)
        
        # Find OUT->IN packet pairs
        directions = seq_nlp[:valid_len, 1]
        pairs = []
        for i in range(valid_len - 1):
            if directions[i] > 0 and directions[i + 1] < 0:  # OUT -> IN
                pairs.append((i, i + 1))
        
        if len(pairs) < 2:
            return self._create_dummy_tcp_sample(bytes_nlp)
        
        # Randomly choose one pair as positive sample
        pair_idx = random.randint(0, len(pairs) - 1)
        req_idx, resp_idx = pairs[pair_idx]
        
        # Select another response as negative sample
        other_pairs = [p for j, p in enumerate(pairs) if j != pair_idx]
        neg_resp_idx = random.choice(other_pairs)[1]
        
        # Randomly decide the position of the correct response
        if random.random() < 0.5:
            cand_a_idx, cand_b_idx = resp_idx, neg_resp_idx
            label = 0  # A is correct
        else:
            cand_a_idx, cand_b_idx = neg_resp_idx, resp_idx
            label = 1  # B is correct
        
        return {
            'request_bytes': bytes_nlp[req_idx],
            'request_seq': seq_nlp[req_idx],
            'request_tcp': tcp_data[req_idx],
            'cand_a_bytes': bytes_nlp[cand_a_idx],
            'cand_a_seq': seq_nlp[cand_a_idx],
            'cand_a_tcp': tcp_data[cand_a_idx],
            'cand_b_bytes': bytes_nlp[cand_b_idx],
            'cand_b_seq': seq_nlp[cand_b_idx],
            'cand_b_tcp': tcp_data[cand_b_idx],
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }

    def _create_iat_anomaly_sample_v2(self, idx):
        """
        Improved version of IAT Anomaly Detection.
        - Uses more obvious anomaly patterns.
        - Statistics-based anomalies instead of random factors.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 3:
            return self._create_dummy_anomaly_sample(bytes_nlp, seq_nlp, tcp_data)
        
        label = 1  # Normal
        
        # Calculate statistics of original IATs
        valid_iats = seq_nlp[1:valid_len, 2]  # Skip the first packet
        if len(valid_iats) > 0:
            mean_iat = valid_iats.mean().item()
            std_iat = valid_iats.std().item() + 1e-6
        else:
            mean_iat, std_iat = 0.1, 0.1
        
        # 50% chance to create anomaly
        if random.random() < 0.5:
            label = 0
            anomaly_idx = random.randint(1, valid_len - 1)
            
            # Anomaly type: deviation beyond statistics
            anomaly_type = random.choice(['spike', 'drop', 'zero', 'huge'])
            
            if anomaly_type == 'spike':
                # Sudden increase (5-10x mean)
                seq_nlp[anomaly_idx, 2] = mean_iat * random.uniform(5, 10)
            elif anomaly_type == 'drop':
                # Sudden decrease (near 0)
                seq_nlp[anomaly_idx, 2] = mean_iat * random.uniform(0.001, 0.01)
            elif anomaly_type == 'zero':
                # Explicit zero
                seq_nlp[anomaly_idx, 2] = 0
            else:  # huge
                # Extreme outlier
                seq_nlp[anomaly_idx, 2] = random.uniform(10, 100)
        
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }
    
    # ==========================================
    # Task 3 Improved: Size Prediction - Fine-grained
    # ==========================================
    def _create_size_prediction_sample_v2(self, idx):
        """
        Improved version of packet size prediction.
        - Predict actual size values (Regression) instead of bins.
        - Or use finer-grained bin divisions.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 2:
            return self._create_dummy_size_sample(bytes_nlp, seq_nlp, tcp_data)
        
        pred_idx = min(valid_len - 1, self.max_packets - 1)
        target_size = seq_nlp[pred_idx, 0].item()
        
        # Use finer bins (e.g., 10 intervals)
        fine_bins = [0, 40, 80, 160, 256, 400, 600, 900, 1200, 1500]
        
        target_bin = np.digitize(target_size, fine_bins) - 1
        target_bin = min(max(target_bin, 0), len(fine_bins) - 1)
        
        # Mask prediction position
        input_bytes = bytes_nlp.clone()
        input_seq = seq_nlp.clone()
        input_tcp = tcp_data.clone()
        
        input_bytes[pred_idx] = 0
        input_seq[pred_idx] = 0
        input_tcp[pred_idx] = 0
        
        return {
            'bytes_nlp': input_bytes,
            'seq_nlp': input_seq,
            'tcp_data': input_tcp,
            'pred_idx': torch.tensor(pred_idx).long(),
            'label': torch.tensor(target_bin).long(),
            'target_size_raw': torch.tensor(target_size).float(),  # Also return raw value for regression
            'valid': torch.tensor(1).long()
        }

    def _create_dummy_tcp_sample(self, bytes_nlp):
        n_bytes = bytes_nlp.shape[1]
        return {
            'request_bytes': torch.zeros(n_bytes, dtype=torch.long),
            'request_seq': torch.zeros(3),
            'request_tcp': torch.zeros(3),
            'cand_a_bytes': torch.zeros(n_bytes, dtype=torch.long),
            'cand_a_seq': torch.zeros(3),
            'cand_a_tcp': torch.zeros(3),
            'cand_b_bytes': torch.zeros(n_bytes, dtype=torch.long),
            'cand_b_seq': torch.zeros(3),
            'cand_b_tcp': torch.zeros(3),
            'label': torch.tensor(0).long(),
            'valid': torch.tensor(0).long()
        }

    # ==========================================
    # Task 5: Cross-Modal Consistency
    # ==========================================
    def _create_consistency_sample(self, idx):
        """
        Judge whether the features of different modalities in a flow are consistent.
        Positive (label=1): Original flow, modalities are consistent.
        Negative (label=0): One modality is replaced by the corresponding modality of another flow.
        """
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 2:
            return self._create_dummy_consistency_sample(bytes_nlp, seq_nlp, tcp_data)
        
        label = 1  # Consistent by default
        
        # 50% chance to create inconsistent sample
        if random.random() < 0.5:
            label = 0
            # Choose another flow randomly
            other_idx = random.randint(0, self.length - 1)
            while other_idx == idx:
                other_idx = random.randint(0, self.length - 1)
            
            _, other_seq, other_tcp, other_valid = self._load_and_pad(other_idx)
            
            if other_valid >= 2:
                # Randomly choose a modality to replace
                modality = random.choice(['size', 'direction', 'iat'])
                
                if modality == 'size':
                    seq_nlp[:, 0] = other_seq[:, 0]
                elif modality == 'direction':
                    seq_nlp[:, 1] = other_seq[:, 1]
                else:  # iat
                    seq_nlp[:, 2] = other_seq[:, 2]
        
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(label).long(),
            'valid': torch.tensor(1).long()
        }
    
    def _create_dummy_consistency_sample(self, bytes_nlp, seq_nlp, tcp_data):
        return {
            'bytes_nlp': bytes_nlp,
            'seq_nlp': seq_nlp,
            'tcp_data': tcp_data,
            'label': torch.tensor(1).long(),
            'valid': torch.tensor(0).long()
        }

    # ==========================================
    # Task 6: Masked Context Recovery
    # ==========================================
    def _create_masked_recovery_sample(self, idx):
        """Legacy context recovery task."""
        bytes_nlp, seq_nlp, tcp_data, valid_len = self._load_and_pad(idx)
        
        if valid_len < 3:
            mask_idx = 1 if valid_len >= 2 else 0
        else:
            mask_idx = random.randint(1, valid_len - 1)
        
        # Save target values
        target_packet = seq_nlp[mask_idx].clone()
        
        # Construct input
        input_bytes = bytes_nlp.clone()
        input_bytes[mask_idx] = 0
        
        input_seq = seq_nlp.clone()
        input_seq[mask_idx] = 0
        
        input_tcp = tcp_data.clone()
        input_tcp[mask_idx] = 0
        
        # Prepare targets
        target_size = torch.log1p(target_packet[0])
        target_dir = target_packet[1]
        target_iat = torch.log1p(target_packet[2])
        
        if target_dir > 0:
            dir_label = 0
        elif target_dir < 0:
            dir_label = 1
        else:
            dir_label = 2

        return {
            'bytes_nlp': input_bytes,
            'seq_nlp': input_seq,
            'tcp_data': input_tcp,
            'mask_idx': torch.tensor(mask_idx).long(),
            'target_size': target_size,
            'target_dir': torch.tensor(dir_label).long(),
            'target_iat': target_iat,
            'valid': torch.tensor(1).long()
        }


def probe_collate_fn(batch):
    """General collate function for probing tasks."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


def tcp_match_collate_fn(batch):
    """Collate function specifically for TCP matching task."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    return collated