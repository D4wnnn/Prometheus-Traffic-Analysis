"""
Packet-level Multi-Head Attention Module (5-Head Design)
Includes multimodal guidance for temporal, size, direction, and TCP features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import (
    PositionalEncoding,
    compute_temporal_bias,
    compute_size_similarity,
    compute_direction_bias,
    compute_tcp_bias,
    LearnableBiasScaler,
    AdaptiveHeadGating,
    create_padding_mask
)


class MultiModalMultiHeadAttention(nn.Module):
    """
    Multimodal Multi-Head Attention (5 heads + Dynamic Head Gating).
    
    Head 0: Content Head (Standard Q-K Attention).
    Head 1: Temporal Head (Guided by time intervals).
    Head 2: Size Head (Guided by size similarity).
    Head 3: Direction Head (Guided by asymmetric direction info).
    Head 4: TCP Head (Guided by TCP semantics).
    
    Args:
        d_model: Model dimension.
        num_heads: Total heads (fixed to 5).
        dropout: Dropout probability.
        use_learnable_bias: Use learnable bias weights.
        use_adaptive_gating: Use dynamic head gating.
    """
    def __init__(self, d_model=260, num_heads=5, dropout=0.1, 
                 use_learnable_bias=True, use_adaptive_gating=True, disabled_head_indices=None):
        super().__init__()
        
        assert num_heads == 5, "Current design is fixed to 5 heads."
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_adaptive_gating = use_adaptive_gating
        self.disabled_head_indices = disabled_head_indices if disabled_head_indices is not None else []
        
        # Independent Q, K, V projections for each head
        self.W_Q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
        # Output projection
        if use_adaptive_gating:
            # Gating mode: weighted sum followed by projection.
            self.W_O = nn.Linear(self.d_k, d_model)
            self.head_gating = AdaptiveHeadGating(d_model, num_heads, dropout)
        else:
            # Standard mode: concatenation followed by projection.
            self.W_O = nn.Linear(d_model, d_model)
        
        # Learnable bias scaling parameters
        self.use_learnable_bias = use_learnable_bias
        if use_learnable_bias:
            self.bias_scaler = LearnableBiasScaler(
                num_bias_types=4,
                init_values=[0.5, 0.3, 0.8, 0.8]
            )
        else:
            self.register_buffer('bias_weights', torch.tensor([0.5, 0.3, 0.8, 0.8]))
        
        self.dropout = nn.Dropout(dropout)
        
        # Window size configurations
        self.temporal_window = float('inf')
        self.size_window = 10
        self.direction_window = 30
        self.tcp_window = 30
    
    def forward(self, x, seq_features, mask=None):
        """
        Args:
            x: (batch_size, n_packets, d_model) Packet representation.
            seq_features: dict containing sequence features.
            mask: (batch_size, n_packets) Padding mask, True at positions to mask.
        
        Returns:
            (batch_size, n_packets, d_model) Attention output.
        """
        batch_size, n_packets, _ = x.shape
        
        # Extract features
        timestamps = seq_features['timestamps']
        sizes = seq_features['sizes']
        directions = seq_features['directions']
        tcp_data = seq_features['tcp_data']
        
        # Compute biases
        bias_temporal = compute_temporal_bias(timestamps, sigma=1.0)
        bias_size = compute_size_similarity(sizes, window_size=self.size_window)
        bias_direction = compute_direction_bias(
            directions, alpha_same=0.5, window_size=self.direction_window
        )
        bias_tcp = compute_tcp_bias(tcp_data, directions)
        
        # Get bias weights
        if self.use_learnable_bias:
            lambda_weights = self.bias_scaler()
        else:
            lambda_weights = self.bias_weights
            
        if self.disabled_head_indices:
            # Create weight mask (avoids in-place operations to preserve gradients)
            weight_mask = torch.ones_like(lambda_weights)
            for h_idx in self.disabled_head_indices:
                # Only apply to heads 1-4 (Head 0 is the Content Head with no external bias)
                if 1 <= h_idx <= 4:
                    weight_mask[h_idx - 1] = 0.0
            lambda_weights = lambda_weights * weight_mask
            
        head_outputs = []
        
        # ========== Head 0: Content Head ==========
        Q0 = self.W_Q[0](x)
        K0 = self.W_K[0](x)
        V0 = self.W_V[0](x)
        score0 = torch.matmul(Q0, K0.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            score0 = score0.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn0 = F.softmax(score0, dim=-1)
        attn0 = self.dropout(attn0)
        output0 = torch.matmul(attn0, V0)
        head_outputs.append(output0)
        
        # ========== Head 1: Temporal Head ==========
        Q1 = self.W_Q[1](x)
        K1 = self.W_K[1](x)
        V1 = self.W_V[1](x)
        score1 = torch.matmul(Q1, K1.transpose(-2, -1)) / math.sqrt(self.d_k)
        score1 = score1 + lambda_weights[0] * bias_temporal
        if mask is not None:
            score1 = score1.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn1 = F.softmax(score1, dim=-1)
        attn1 = self.dropout(attn1)
        output1 = torch.matmul(attn1, V1)
        head_outputs.append(output1)
        
        # ========== Head 2: Size Head ==========
        Q2 = self.W_Q[2](x)
        K2 = self.W_K[2](x)
        V2 = self.W_V[2](x)
        score2 = torch.matmul(Q2, K2.transpose(-2, -1)) / math.sqrt(self.d_k)
        score2 = score2 + lambda_weights[1] * bias_size
        if mask is not None:
            score2 = score2.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn2 = F.softmax(score2, dim=-1)
        attn2 = self.dropout(attn2)
        output2 = torch.matmul(attn2, V2)
        head_outputs.append(output2)
        
        # ========== Head 3: Direction Head ==========
        Q3 = self.W_Q[3](x)
        K3 = self.W_K[3](x)
        V3 = self.W_V[3](x)
        score3 = torch.matmul(Q3, K3.transpose(-2, -1)) / math.sqrt(self.d_k)
        score3 = score3 + lambda_weights[2] * bias_direction
        if mask is not None:
            score3 = score3.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn3 = F.softmax(score3, dim=-1)
        attn3 = self.dropout(attn3)
        output3 = torch.matmul(attn3, V3)
        head_outputs.append(output3)
        
        # ========== Head 4: TCP Head ==========
        Q4 = self.W_Q[4](x)
        K4 = self.W_K[4](x)
        V4 = self.W_V[4](x)
        score4 = torch.matmul(Q4, K4.transpose(-2, -1)) / math.sqrt(self.d_k)
        score4 = score4 + lambda_weights[3] * bias_tcp
        if mask is not None:
            score4 = score4.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn4 = F.softmax(score4, dim=-1)
        attn4 = self.dropout(attn4)
        output4 = torch.matmul(attn4, V4)
        head_outputs.append(output4)
        
        # ========== Multi-Head Fusion ==========
        if self.use_adaptive_gating:
            # Compute global flow representation for gating
            if mask is not None:
                valid_mask = ~mask  # (B, N)
                valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
                x_masked = x * valid_mask.unsqueeze(-1).float()
                global_repr = x_masked.sum(dim=1) / valid_counts.float()
            else:
                global_repr = x.mean(dim=1)
            
            # Compute gate weights (per sample)
            gates = self.head_gating(global_repr)  # (B, num_heads)
            
            # [TRICK 1] Kill weights for ablated heads
            if self.disabled_head_indices:
                gate_mask = torch.ones_like(gates)
                for h_idx in self.disabled_head_indices:
                    if 0 <= h_idx < self.num_heads:
                        gate_mask[:, h_idx] = 0.0
                gates = gates * gate_mask
            
            # Apply gating to each head's output
            weighted_outputs = []
            for i, head_out in enumerate(head_outputs):
                weight = gates[:, i:i+1].unsqueeze(1)  # (B, 1, 1)
                weighted_outputs.append(weight * head_out)  # (B, N, d_k)
            
            # Weighted summation (not concatenation)
            multi_head_output = sum(weighted_outputs)  # (B, N, d_k)
            output = self.W_O(multi_head_output)  # (B, N, d_model)
        else:
            # Standard concatenation
            multi_head_output = torch.cat(head_outputs, dim=-1)  # (B, N, d_model)
            output = self.W_O(multi_head_output)  # (B, N, d_model)
        
        return output

    def reset_ablated_heads_weights(self):
        """
        Trick 1: Reset Q/K/V projection weights for ablated heads to eliminate pre-training memory.
        """
        if not self.disabled_head_indices:
            return

        with torch.no_grad():
            for h_idx in self.disabled_head_indices:
                if 0 <= h_idx < self.num_heads:
                    # Reset Q, K, V; preserve W_O as it is a fusion layer.
                    nn.init.xavier_uniform_(self.W_Q[h_idx].weight)
                    if self.W_Q[h_idx].bias is not None:
                        nn.init.zeros_(self.W_Q[h_idx].bias)
                    
                    nn.init.xavier_uniform_(self.W_K[h_idx].weight)
                    if self.W_K[h_idx].bias is not None:
                        nn.init.zeros_(self.W_K[h_idx].bias)
                    
                    nn.init.xavier_uniform_(self.W_V[h_idx].weight)
                    if self.W_V[h_idx].bias is not None:
                        nn.init.zeros_(self.W_V[h_idx].bias)


class PacketTransformerLayer(nn.Module):
    """
    Packet-level Transformer Layer
    
    Includes:
    1. Multimodal multi-head attention.
    2. FFN.
    3. Residual connections and LayerNorm.
    """
    def __init__(self, d_model=260, num_heads=5, d_ff=1024, dropout=0.1, use_adaptive_gating=True, disabled_head_indices=None):
        super().__init__()
        
        self.attention = MultiModalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_adaptive_gating=use_adaptive_gating,
            disabled_head_indices=disabled_head_indices
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, seq_features, mask=None):
        """
        Args:
            x: (batch_size, n_packets, d_model).
            seq_features: dict, sequence features.
            mask: padding mask.
        
        Returns:
            (batch_size, n_packets, d_model).
        """
        # Multimodal attention + Residual
        attn_output = self.attention(self.norm1(x), seq_features, mask)
        x = x + self.dropout(attn_output)
        
        # FFN + Residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x


class PacketTransformerEncoder(nn.Module):
    """
    Packet-level Transformer Encoder
    
    Stacks multiple PacketTransformerLayers.
    
    Args:
        d_model: Model dimension.
        num_layers: Number of layers.
        num_heads: Heads per layer.
        d_ff: FFN dimension.
        max_packets: Maximum packet count.
        dropout: Dropout probability.
    """
    def __init__(self, 
                 d_model=260, 
                 num_layers=4, 
                 num_heads=5, 
                 d_ff=1024, 
                 max_packets=100, 
                 dropout=0.1,
                 use_adaptive_gating=True,
                 disabled_head_indices=None):
        super().__init__()
        
        self.d_model = d_model
        self.max_packets = max_packets
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_packets, dropout=dropout)
        
        # Stacks Transformer layers
        self.layers = nn.ModuleList([
            PacketTransformerLayer(d_model, num_heads, d_ff, dropout, use_adaptive_gating, disabled_head_indices)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, packet_repr, seq_features, mask=None):
        """
        Args:
            packet_repr: (batch_size, n_packets, d_model).
            seq_features: dict, sequence features.
            mask: padding mask.
        
        Returns:
            (batch_size, n_packets, d_model) encoded packet sequence.
        """
        # Add positional encoding
        x = self.pos_encoder(packet_repr)
        
        # Propagation through layers
        for layer in self.layers:
            x = layer(x, seq_features, mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x