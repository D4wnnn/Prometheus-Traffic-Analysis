"""
Pre-training Dataset
Loading unlabeled HDF5 data
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
# import sys
# sys.path.append('..')
# from pretrain.pretrain_utils import DataAugmentation
from pretrain_utils import DataAugmentation

class PretrainDataset(Dataset):
    """
    Pre-training dataset (Unlabeled)
    
    Args:
        h5_file: Path to the HDF5 file
        max_packets: Maximum number of packets (truncated if exceeded)
        max_bytes: Maximum bytes per packet (truncated if exceeded)
        augmentation: Whether to use data augmentation
        aug_prob: Probability of applying augmentation
    """
    def __init__(self, h5_file, max_packets=None, max_bytes=None, 
                 augmentation=True, aug_prob=0.5):
        self.h5_file = h5_file
        self.max_packets = max_packets
        self.max_bytes = max_bytes
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.data_aug = DataAugmentation()
        
        with h5py.File(h5_file, 'r') as f:
            self.length = f['bytes_nlp_view'].shape[0]
            self.has_tcp_data = 'tcp_data_view' in f
            
            # Get original dimensions from the dataset
            self.data_n_packets = f['bytes_nlp_view'].shape[1]
            self.data_n_bytes = f['bytes_nlp_view'].shape[2]
            
            print(f"[PretrainDataset] Loaded {self.length} samples from {h5_file}")
            print(f"[PretrainDataset] Data shape: packets={self.data_n_packets}, bytes={self.data_n_bytes}")
            
            if self.max_packets is not None:
                print(f"[PretrainDataset] Will truncate/pad packets to {self.max_packets}")
            if self.max_bytes is not None:
                print(f"[PretrainDataset] Will truncate/pad bytes to {self.max_bytes}")
            if self.has_tcp_data:
                print(f"[PretrainDataset] TCP data available")
    
    def __len__(self):
        return self.length
    
    def _truncate_or_pad(self, tensor, target_size, dim, pad_value=0):
        """
        Truncate or pad the tensor to the target size
        
        Args:
            tensor: Input tensor
            target_size: Target size
            dim: Dimension to operate on
            pad_value: Value used for padding
        
        Returns:
            Processed tensor
        """
        current_size = tensor.shape[dim]
        
        if current_size == target_size:
            return tensor
        elif current_size > target_size:
            # Truncation
            indices = [slice(None)] * tensor.dim()
            indices[dim] = slice(0, target_size)
            return tensor[tuple(indices)]
        else:
            # Padding
            pad_shape = list(tensor.shape)
            pad_shape[dim] = target_size - current_size
            
            if tensor.dtype in [torch.float32, torch.float64]:
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
            else:
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
            
            return torch.cat([tensor, padding], dim=dim)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            bytes_nlp = torch.from_numpy(f['bytes_nlp_view'][idx]).long()
            seq_nlp = torch.from_numpy(f['seq_nlp_view'][idx]).float()
            
            # Load TCP data
            if self.has_tcp_data:
                tcp_data = torch.from_numpy(f['tcp_data_view'][idx]).float()
            else:
                num_packets = seq_nlp.shape[0]
                tcp_data = torch.zeros(num_packets, 3, dtype=torch.float)
            
            # ===== Apply max_bytes limit =====
            if self.max_bytes is not None:
                # bytes_nlp: (n_packets, n_bytes) -> truncate/pad on n_bytes dimension
                bytes_nlp = self._truncate_or_pad(bytes_nlp, self.max_bytes, dim=1, pad_value=0)
            
            # ===== Apply max_packets limit =====
            if self.max_packets is not None:
                # bytes_nlp: (n_packets, n_bytes) -> truncate/pad on n_packets dimension
                bytes_nlp = self._truncate_or_pad(bytes_nlp, self.max_packets, dim=0, pad_value=0)
                
                # seq_nlp: (n_packets, 3) -> truncate/pad on n_packets dimension
                seq_nlp = self._truncate_or_pad(seq_nlp, self.max_packets, dim=0, pad_value=0)
                
                # tcp_data: (n_packets, 3) -> truncate/pad on n_packets dimension
                tcp_data = self._truncate_or_pad(tcp_data, self.max_packets, dim=0, pad_value=0)
            
            data = {
                'bytes_nlp': bytes_nlp,
                'seq_nlp': seq_nlp,
                'tcp_data': tcp_data
            }
            
            # Data augmentation (optional)
            if self.augmentation and torch.rand(1).item() < self.aug_prob:
                data = self._apply_augmentation(data)
            
            return data
    
    def _apply_augmentation(self, data):
        """
        Apply data augmentation
        Randomly select one augmentation method
        """
        aug_type = torch.randint(0, 4, (1,)).item()
        
        if aug_type == 0:
            # Byte Jitter
            data['bytes_nlp'] = self.data_aug.byte_jitter(
                data['bytes_nlp'].unsqueeze(0), corruption_prob=0.1
            ).squeeze(0)
        
        elif aug_type == 1:
            # Time Jitter
            data['seq_nlp'] = self.data_aug.time_jitter(
                data['seq_nlp'].unsqueeze(0), iat_noise_std=0.1
            ).squeeze(0)
        
        elif aug_type == 2:
            # Byte Truncation
            data['bytes_nlp'] = self.data_aug.byte_truncation(
                data['bytes_nlp'].unsqueeze(0), min_ratio=0.7
            ).squeeze(0)
        
        elif aug_type == 3:
            # Packet Dropout
            bytes_aug, seq_aug, tcp_aug = self.data_aug.packet_dropout(
                data['bytes_nlp'].unsqueeze(0),
                data['seq_nlp'].unsqueeze(0),
                data['tcp_data'].unsqueeze(0),
                dropout_prob=0.1
            )
            data['bytes_nlp'] = bytes_aug.squeeze(0)
            data['seq_nlp'] = seq_aug.squeeze(0)
            data['tcp_data'] = tcp_aug.squeeze(0)
        
        return data


def collate_fn(batch):
    """Batch processing function"""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    return collated