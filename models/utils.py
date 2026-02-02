"""
Utility Functions Module
Contains general utilities such as positional encoding and mask generation.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding
    
    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def create_padding_mask(seq, pad_value=0):
    """
    Create padding mask
    
    Args:
        seq: (batch_size, seq_len) or (batch_size, seq_len, feature_dim).
        pad_value: Value used for padding.
    
    Returns:
        mask: (batch_size, seq_len), True at positions to be masked.
    """
    if seq.dim() == 3:
        # For (batch, seq_len, feature_dim), check if the first feature is 0
        mask = (seq[:, :, 0] == pad_value)
    else:
        # For (batch, seq_len)
        mask = (seq == pad_value)
    
    return mask


def compute_temporal_bias(timestamps, sigma=1.0):
    """
    Compute temporal bias (Gaussian Kernel)
    
    Args:
        timestamps: (batch_size, n_packets), Timestamps (cumulative time calculated from IAT).
        sigma: Standard deviation of the Gaussian kernel.
    
    Returns:
        bias: (batch_size, n_packets, n_packets)
    """
    # 1. Compute time difference matrix (unit: seconds)
    delta_t_sec = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)   # (B, N, N)
    delta_t_sec = torch.abs(delta_t_sec)
    
    # 2. Convert to milliseconds (ms)
    delta_t_ms = delta_t_sec * 1000.0
    
    # 3. Apply log10(x + 1) transform for scale compression and numerical safety
    delta_t_transformed = torch.log10(delta_t_ms + 1.0)
    
    # 4. Compute Gaussian kernel using transformed time differences
    # Note: sigma=1.0 might need readjustment under the new scale
    bias = -((delta_t_transformed / sigma) ** 2)
    
    return bias


def compute_size_similarity(sizes, window_size=10):
    """
    Compute size similarity bias (local window)
    
    Args:
        sizes: (batch_size, n_packets), Packet sizes.
        window_size: Window size.
    
    Returns:
        bias: (batch_size, n_packets, n_packets)
    """
    batch_size, n_packets = sizes.shape
    
    # Compute similarity matrix
    size_i = sizes.unsqueeze(2)  # (B, N, 1)
    size_j = sizes.unsqueeze(1)  # (B, 1, N)
    
    similarity = torch.min(size_i, size_j) / (torch.max(size_i, size_j) + 1e-8)
    
    # Create window mask
    positions = torch.arange(n_packets, device=sizes.device)
    distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))  # (N, N)
    window_mask = (distance <= window_size).float()
    
    # Apply window mask
    bias = similarity * window_mask.unsqueeze(0)
    
    return bias


def compute_direction_bias(directions, alpha_same=1.0, window_size=30):
    """
    Compute asymmetric direction bias (modified: focuses only on same-direction info)
    
    Args:
        directions: (batch_size, n_packets), Directions (1=OUT, -1=IN, 0=padding).
        alpha_same: Correlation weight for the same direction.
        window_size: Window size.
    """
    batch_size, n_packets = directions.shape
    
    dir_i = directions.unsqueeze(2) # (B, N, 1)
    dir_j = directions.unsqueeze(1) # (B, 1, N)
    
    bias = torch.zeros(batch_size, n_packets, n_packets, device=directions.device)
    
    # Same direction
    mask_same = (dir_i == dir_j) & (dir_i != 0) & (dir_j != 0)
    bias = torch.where(mask_same, torch.tensor(alpha_same, device=bias.device), bias)
    
    # Apply window mask
    positions = torch.arange(n_packets, device=directions.device)
    pos_i = positions.unsqueeze(1) # (N, 1)
    pos_j = positions.unsqueeze(0) # (1, N)
    
    distance = torch.abs(pos_i - pos_j)
    window_mask = (distance <= window_size).float().unsqueeze(0)
    bias = bias * window_mask
    
    return bias


def _compute_tcp_seq_ack_bias(seq, ack, payload_len, mask_temporal, alpha_tcp_match):
    """
    Helper function: Computes exact match bias based on TCP Seq/Ack.
    
    Args:
        seq (B, N): Sequence numbers.
        ack (B, N): Acknowledgment numbers.
        payload_len (B, N): Payload length.
        mask_temporal (N, N): Temporal mask (j > i).
        alpha_tcp_match (float): Weight for exact matches.
    """
    # TCP consumes 1 sequence number even if payload=0 (e.g., SYN/FIN).
    # Use clamp(min=1) to correctly compute expected ACK.
    expected_ack = seq + torch.clamp(payload_len, min=1)
    
    expected_ack_i = expected_ack.unsqueeze(2) # (B, N, 1)
    ack_j = ack.unsqueeze(1) # (B, 1, N)
    
    # Match condition: j's ack == i's expected_ack, j's ack > 0, and j occurs after i.
    mask_match_tcp = (ack_j == expected_ack_i) & (ack_j > 0) & mask_temporal
    
    bias_tcp = torch.where(
        mask_match_tcp, 
        torch.tensor(alpha_tcp_match, device=seq.device), 
        0.0
    )
    return bias_tcp


def _compute_direction_fallback_bias(directions, mask_temporal, alpha_dir_strong, alpha_dir_weak):
    """
    Helper function: Computes fallback bias based on direction.
    
    Args:
        directions (B, N): Directions.
        mask_temporal (N, N): Temporal mask (j > i).
        alpha_dir_strong (float): Strong weight for OUT -> IN.
        alpha_dir_weak (float): Weak weight for IN -> OUT.
    """
    batch_size, n_packets = directions.shape
    device = directions.device
    
    dir_i = directions.unsqueeze(2) # (B, N, 1)
    dir_j = directions.unsqueeze(1) # (B, 1, N)
    
    bias = torch.zeros(batch_size, n_packets, n_packets, device=device)
    
    # Logic 1 (Strong correlation): Request(OUT) -> Response(IN), where j > i.
    mask_strong = (dir_i == 1) & (dir_j == -1) & mask_temporal
    bias = torch.where(
        mask_strong, 
        torch.tensor(alpha_dir_strong, device=device), 
        bias
    )
    
    # Logic 2 (Weak correlation): Response(IN) -> New Request(OUT), where j > i.
    mask_weak = (dir_i == -1) & (dir_j == 1) & mask_temporal
    bias = torch.where(
        mask_weak, 
        torch.tensor(alpha_dir_weak, device=device), 
        bias
    )
    
    return bias


def compute_tcp_bias(tcp_features, directions, 
                     alpha_tcp_match=2.0, 
                     alpha_dir_strong=1.5, 
                     alpha_dir_weak=0.5):
    """
    Compute TCP semantic bias (Request-Response relationship).
    
    Workflow:
     1. Prioritize TCP Seq/Ack exact matching (weight alpha_tcp_match).
     2. If flow is invalid (e.g., UDP or all-zero ACK), fallback to direction-based bias.
    
    Args:
        tcp_features: (batch_size, n_packets, 3) [seq, ack, payload_len].
        directions: (batch_size, n_packets), Directions (1=OUT, -1=IN, 0=padding).
        alpha_tcp_match: Weight for exact Seq/Ack matching.
        alpha_dir_strong: Fallback weight for direction (OUT -> IN).
        alpha_dir_weak: Fallback weight for direction (IN -> OUT).
    """
    batch_size, n_packets, _ = tcp_features.shape
    device = tcp_features.device
    
    # --- 1. Feature Extraction ---
    seq = tcp_features[:, :, 0]
    ack = tcp_features[:, :, 1]
    payload_len = tcp_features[:, :, 2]
    
    # --- 2. Check Flow Validity ---
    # Check if each flow in the batch has at least one ack > 0.
    is_valid_tcp = torch.any(ack > 0, dim=1) # (B,)
    is_valid_tcp_mask = is_valid_tcp.view(batch_size, 1, 1) # (B, 1, 1) for broadcasting

    # --- 3. Create General Temporal Mask (ensure j > i) ---
    positions = torch.arange(n_packets, device=device)
    pos_i = positions.unsqueeze(1) # (N, 1)
    pos_j = positions.unsqueeze(0) # (1, N)
    mask_temporal = (pos_j > pos_i) # (N, N), True means j > i

    # --- 4. Branch 1: Compute TCP Seq/Ack Bias ---
    bias_tcp = _compute_tcp_seq_ack_bias(
        seq, ack, payload_len, 
        mask_temporal, 
        alpha_tcp_match
    )
    
    # --- 5. Branch 2: Compute Direction Fallback Bias ---
    bias_dir_fallback = _compute_direction_fallback_bias(
        directions, 
        mask_temporal,
        alpha_dir_strong,
        alpha_dir_weak
    )

    # --- 6. Merge ---
    # Use bias_tcp for valid TCP flows; otherwise use bias_dir_fallback.
    bias = torch.where(is_valid_tcp_mask, bias_tcp, bias_dir_fallback)
    
    return bias


class LearnableBiasScaler(nn.Module):
    """
    Learnable bias scaling parameters.
    """
    def __init__(self, num_bias_types=4, init_values=None):
        super().__init__()
        if init_values is None:
            init_values = [0.5, 0.3, 0.8, 1.0]  # temporal, size, direction, tcp
        
        self.scalers = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))
    
    def forward(self):
        return self.scalers


class AdaptiveHeadGating(nn.Module):
    """
    Dynamically selects which attention heads to activate based on flow features.
    
    Args:
        d_model: Input feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    def __init__(self, d_model, num_heads=5, dropout=0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_heads),
            nn.Sigmoid()  # Activation probability per head (0-1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, d_model), Global representation of the flow.
        
        Returns:
            gates: (batch_size, num_heads), Weight for each head.
        """
        gates = self.gate_network(x)  # (B, num_heads)
        return gates