"""
Pre-training Utility Functions
Includes data augmentation, mask generation, etc.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List


class DataAugmentation:
    """
    Utility class for data augmentation
    """

    @staticmethod
    def byte_jitter(bytes_data, corruption_prob=0.15):
        """
        Byte Jitter: Randomly flip a portion of bytes.

        Args:
            bytes_data: (B, N, L) byte data
            corruption_prob: Probability of flipping a byte
        Returns:
            Augmented byte data
        """
        mask = torch.rand_like(bytes_data.float()) < corruption_prob
        mask = mask & (bytes_data != 0)  # Do not flip padding

        noise = torch.randint_like(bytes_data, 0, 256)
        augmented = torch.where(mask, noise, bytes_data)

        return augmented

    @staticmethod
    def time_jitter(seq_data, iat_noise_std=0.1):
        """
        Time Jitter: Add Gaussian noise to Inter-Arrival Time (IAT).

        Args:
            seq_data: (B, N, 3) [size, direction, iat]
            iat_noise_std: Standard deviation of noise (relative to the original value)
        """
        augmented = seq_data.clone()
        iats = seq_data[:, :, 2]

        # Add noise only to non-zero IATs
        noise = torch.randn_like(iats) * iat_noise_std * iats
        noise = noise * (iats != 0).float()

        augmented[:, :, 2] = torch.clamp(iats + noise, min=0)

        return augmented

    @staticmethod
    def packet_dropout(bytes_data, seq_data, tcp_data, dropout_prob=0.1):
         """
         Packet Dropout: Randomly drop a portion of packets.
         """
         batch_size, n_packets = seq_data.shape[:2]

         # Generate retention mask
         keep_mask = torch.rand(batch_size, n_packets, device=bytes_data.device) > dropout_prob

         # Find valid packets (size != 0)
         valid_packet_mask = (seq_data[:, :, 0] != 0)

         # Ensure at least one *valid* packet is kept for each sample
         for i in range(batch_size):
             valid_kept = (keep_mask[i] & valid_packet_mask[i]).any()
             
             if not valid_kept:
                 # Find indices of all valid packets in this sample
                 valid_indices = valid_packet_mask[i].nonzero(as_tuple=True)[0]
                 
                 if len(valid_indices) > 0:
                     # If valid packets exist, keep the first one
                     keep_mask[i, valid_indices[0]] = True 
                 else:
                     # If the flow is entirely padding, keep the first packet (original logic)
                     keep_mask[i, 0] = True

         # Apply mask
         bytes_aug = bytes_data * keep_mask.unsqueeze(-1).to(bytes_data.dtype)
         seq_aug = seq_data * keep_mask.unsqueeze(-1).float()
         tcp_aug = tcp_data * keep_mask.unsqueeze(-1).float()

         return bytes_aug, seq_aug, tcp_aug

    @staticmethod
    def byte_truncation(bytes_data, min_ratio=0.5, max_ratio=1.0):
        """
        Byte Truncation: Randomly truncate the byte length of packets.

        Args:
            bytes_data: (B, N, L)
            min_ratio: Minimum retention ratio
            max_ratio: Maximum retention ratio
        """
        batch_size, n_packets, max_bytes = bytes_data.shape
        augmented = bytes_data.clone()

        for i in range(batch_size):
            for j in range(n_packets):
                # Find actual length (index of the first zero)
                nonzero = (bytes_data[i, j] != 0).nonzero(as_tuple=True)[0]
                if len(nonzero) == 0:
                    continue

                actual_len = len(nonzero)
                ratio = torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
                new_len = max(1, int(actual_len * ratio))

                # Truncate
                if new_len < actual_len:
                    augmented[i, j, new_len:] = 0

        return augmented


class MaskGenerator:
    """
    Mask Generator (for MPM Task)
    """

    @staticmethod
    def generate_packet_mask(batch_size, n_packets, mask_prob=0.15, min_mask=1, device="cuda"):
        """
        Generate packet-level mask

        Args:
            batch_size: Batch size
            n_packets: Number of packets
            mask_prob: Probability of masking
            min_mask: Minimum number of masked packets
            device: Computing device

        Returns:
            mask: (B, N) bool tensor, True indicates masked
        """
        mask = torch.rand(batch_size, n_packets, device=device) < mask_prob

        # Ensure each sample has at least min_mask packets masked
        for i in range(batch_size):
            if mask[i].sum() < min_mask:
                indices = torch.randperm(n_packets, device=device)[:min_mask]
                mask[i, indices] = True

        return mask

    @staticmethod
    def generate_byte_mask(bytes_data, mask_prob=0.15, mask_token=256):
        """
        Generate byte-level mask (applied to packets not masked at the packet level)

        Args:
            bytes_data: (B, N, L)
            mask_prob: Probability of masking
            mask_token: Value of the mask token

        Returns:
            masked_bytes: Masked byte data
            byte_mask: (B, N, L) bool mask
        """
        batch_size, n_packets, n_bytes = bytes_data.shape

        # Generate mask only for non-zero bytes
        valid_mask = bytes_data != 0
        random_mask = torch.rand_like(bytes_data.float()) < mask_prob
        byte_mask = valid_mask & random_mask

        # Apply mask
        masked_bytes = bytes_data.clone()
        masked_bytes[byte_mask] = mask_token

        return masked_bytes, byte_mask


def create_contrastive_pairs(batch_data, window_size=4):
    """
    Create positive pairs for contrastive learning.
    Strategy: Split the same flow into two non-overlapping segments.

    Args:
        batch_data: dict containing bytes_nlp, seq_nlp, tcp_data
        window_size: Minimum window size

    Returns:
        anchor_data: dict
        positive_data: dict
    """
    bytes_nlp = batch_data["bytes_nlp"]
    seq_nlp = batch_data["seq_nlp"]
    tcp_data = batch_data["tcp_data"]

    batch_size, n_packets = seq_nlp.shape[:2]

    # Find the number of valid packets
    valid_lengths = (seq_nlp[:, :, 0] != 0).sum(dim=1)  # (B,)

    anchor_data = {"bytes_nlp": [], "seq_nlp": [], "tcp_data": []}
    positive_data = {"bytes_nlp": [], "seq_nlp": [], "tcp_data": []}
    data_aug = DataAugmentation()
    for i in range(batch_size):
        valid_len = valid_lengths[i].item()

        if valid_len <= 2 * window_size:
            # If too few packets, use the entire flow for both anchor and positive (with different augmentations)
            anchor_data["bytes_nlp"].append(bytes_nlp[i])
            anchor_data["seq_nlp"].append(seq_nlp[i])
            anchor_data["tcp_data"].append(tcp_data[i])

            # Apply strong augmentation to create variance
            curr_bytes = bytes_nlp[i].unsqueeze(0)
            curr_seq = seq_nlp[i].unsqueeze(0)
            curr_tcp = tcp_data[i].unsqueeze(0)
            
            aug_bytes, aug_seq, aug_tcp = data_aug.packet_dropout(
                curr_bytes, curr_seq, curr_tcp, dropout_prob=0.2
            )
            aug_bytes = data_aug.byte_jitter(aug_bytes, corruption_prob=0.15)

            positive_data["bytes_nlp"].append(aug_bytes.squeeze(0))
            positive_data["seq_nlp"].append(aug_seq.squeeze(0))
            positive_data["tcp_data"].append(aug_tcp.squeeze(0))
        else:
            # Case 2: Long flow segmentation
            split_point = torch.randint(window_size, valid_len - window_size, (1,)).item()

            # --- Anchor (First part) ---
            anchor_bytes = bytes_nlp[i].clone()
            anchor_seq = seq_nlp[i].clone()
            anchor_tcp = tcp_data[i].clone()

            # Zero out the part after the split point
            anchor_bytes[split_point:] = 0
            anchor_seq[split_point:] = 0
            anchor_tcp[split_point:] = 0

            # --- Positive (Second part -> moved to the beginning) ---
            slice_bytes = bytes_nlp[i, split_point:valid_len]
            slice_seq = seq_nlp[i, split_point:valid_len]
            slice_tcp = tcp_data[i, split_point:valid_len]

            pad_len = n_packets - slice_bytes.shape[0]

            pad_bytes = torch.zeros((pad_len, *slice_bytes.shape[1:]), dtype=slice_bytes.dtype, device=slice_bytes.device)
            pad_seq = torch.zeros((pad_len, *slice_seq.shape[1:]), dtype=slice_seq.dtype, device=slice_seq.device)
            pad_tcp = torch.zeros((pad_len, *slice_tcp.shape[1:]), dtype=slice_tcp.dtype, device=slice_tcp.device)

            pos_bytes = torch.cat([slice_bytes, pad_bytes], dim=0)
            pos_seq = torch.cat([slice_seq, pad_seq], dim=0)
            pos_tcp = torch.cat([slice_tcp, pad_tcp], dim=0)

            # Reset first packet IAT as it is moved to the start
            if pos_seq.shape[0] > 0:
                pos_seq[0, 2] = 0.0

            anchor_data["bytes_nlp"].append(anchor_bytes)
            anchor_data["seq_nlp"].append(anchor_seq)
            anchor_data["tcp_data"].append(anchor_tcp)

            positive_data["bytes_nlp"].append(pos_bytes)
            positive_data["seq_nlp"].append(pos_seq)
            positive_data["tcp_data"].append(pos_tcp)

    # Stack
    for key in anchor_data:
        anchor_data[key] = torch.stack(anchor_data[key])
        positive_data[key] = torch.stack(positive_data[key])

    return anchor_data, positive_data


def compute_flow_statistics(seq_nlp):
    """
    Compute flow statistical features (for the statistics reconstruction task)

    Args:
        seq_nlp: (B, N, 3) [size, direction, iat]

    Returns:
        stats: dict containing various statistical measures
    """
    batch_size = seq_nlp.shape[0]
    device = seq_nlp.device

    sizes = seq_nlp[:, :, 0]
    directions = seq_nlp[:, :, 1]
    iats = seq_nlp[:, :, 2]

    # Filter padding
    valid_mask = sizes != 0

    stats = {}

    # 1. Packet size distribution (10-bin histogram)
    size_hist = []
    for i in range(batch_size):
        valid_sizes = sizes[i][valid_mask[i]]
        if len(valid_sizes) == 0:
            size_hist.append(torch.zeros(10, device=device))
        else:
            hist = torch.histc(valid_sizes.float(), bins=10, min=0, max=1500)
            hist = hist / (hist.sum() + 1e-8)  # Normalization
            size_hist.append(hist)

    stats["size_histogram"] = torch.stack(size_hist)  # (B, 10)

    # 2. Upstream/Downstream packet ratios
    up_ratio = []
    down_ratio = []
    for i in range(batch_size):
        valid_dirs = directions[i][valid_mask[i]]
        if len(valid_dirs) == 0:
            up_ratio.append(torch.tensor(0.5, device=device))
            down_ratio.append(torch.tensor(0.5, device=device))
        else:
            up_count = (valid_dirs == 1).sum().float()
            down_count = (valid_dirs == -1).sum().float()
            total = len(valid_dirs)
            up_ratio.append(up_count / total)
            down_ratio.append(down_count / total)

    stats["up_ratio"] = torch.stack(up_ratio)  # (B,)
    stats["down_ratio"] = torch.stack(down_ratio)  # (B,)

    # 3. IAT statistics (mean and standard deviation)
    iat_mean = []
    iat_std = []
    for i in range(batch_size):
        valid_iats = iats[i][valid_mask[i]]
        if len(valid_iats) == 0:
            iat_mean.append(torch.tensor(0.0, device=device))
            iat_std.append(torch.tensor(0.0, device=device))
        else:
            valid_iats_log = torch.log1p(valid_iats)
            iat_mean.append(valid_iats_log.mean())
            iat_std.append(valid_iats_log.std() if len(valid_iats) > 1 else torch.tensor(0.0, device=device))

    stats["iat_mean"] = torch.stack(iat_mean)  # (B,)
    stats["iat_std"] = torch.stack(iat_std)  # (B,)

    # 4. Flow duration (cumulative IAT)
    flow_duration = []
    for i in range(batch_size):
        valid_iats = iats[i][valid_mask[i]]
        if len(valid_iats) == 0:
            flow_duration.append(torch.tensor(0.0, device=device))
        else:
            flow_duration.append(valid_iats.sum())

    stats["flow_duration"] = torch.stack(flow_duration)  # (B,)
    return stats