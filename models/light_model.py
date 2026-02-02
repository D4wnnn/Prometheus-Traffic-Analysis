"""
Light Model: Lightweight Traffic Classifier
Designed for fast online inference and as a pre-filter for the Heavy Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightByteEncoder(nn.Module):
    """
    Lightweight Byte Encoder: Embedding + 1D-CNN.
    """
    def __init__(self, vocab_size=257, d_embed=32, d_out=64, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        
        # 1D CNN: Extracts local features from byte sequences.
        # Input: (B*N, d_embed, L) -> Output: (B*N, d_out, L')
        self.cnn = nn.Sequential(
            nn.Conv1d(d_embed, d_out, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
            nn.Conv1d(d_out, d_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_out),
            nn.ReLU()
        )
        self.d_out = d_out

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_packets, n_bytes)
        Returns:
            packet_repr: (batch_size, n_packets, d_out)
        """
        batch_size, n_packets, n_bytes = x.shape
        
        # 1. Reshape for CNN: (B*N, n_bytes)
        x_flat = x.view(-1, n_bytes)
        
        # 2. Embedding: (B*N, n_bytes, d_embed) -> Permute to (B*N, d_embed, n_bytes)
        x_embed = self.embedding(x_flat).permute(0, 2, 1)
        
        # 3. CNN
        feat = self.cnn(x_embed) # (B*N, d_out, n_bytes)
        
        # 4. Global Max Pooling over bytes
        feat = torch.max(feat, dim=2)[0] # (B*N, d_out)
        
        # 5. Reshape back
        return feat.view(batch_size, n_packets, self.d_out)


class DynamicFeatureGating(nn.Module):
    """
    Dynamic Feature Gating Controller (Meta-Controller).
    Dynamically determines weights for byte features and statistical features based on input flow statistics.
    """
    def __init__(self, d_stats=3, hidden_dim=16):
        super().__init__()
        # Input: Global flow statistics (e.g., avg size, avg IAT).
        self.gate_net = nn.Sequential(
            nn.Linear(d_stats, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), # Output two weights [weight_byte, weight_stats]
            nn.Softmax(dim=-1)
        )

    def forward(self, global_stats):
        """
        Args:
            global_stats: (batch_size, d_stats)
        Returns:
            weights: (batch_size, 2)
        """
        return self.gate_net(global_stats)


class LightTrafficClassifier(nn.Module):
    """
    Main Light Model Architecture.
    """
    def __init__(self, num_classes, d_model=64, max_packets=100):
        super().__init__()
        
        # 1. Byte Branch
        self.byte_encoder = LightByteEncoder(d_embed=32, d_out=d_model)
        
        # 2. Statistical Branch (Size, Dir, IAT) -> MLP
        # Removed BatchNorm1d to avoid DDP errors and because log1p transform makes it less critical.
        self.stats_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Dynamic Gating
        self.gating = DynamicFeatureGating(d_stats=3)
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, batch_data, return_feats=False):
        """
        Args:
            batch_data: dict containing 'bytes_nlp', 'seq_nlp'.
        """
        bytes_nlp = batch_data['bytes_nlp'] # (B, N, L)
        seq_nlp = batch_data['seq_nlp']     # (B, N, 3)
        
        # ===== Fix: Log Transform =====
        # Compress raw large values (0~1500) into range (0~7.5) to prevent linear layer saturation.
        size = torch.log1p(seq_nlp[:, :, 0].float())
        direction = seq_nlp[:, :, 1].float()
        iat = torch.log1p(seq_nlp[:, :, 2].float())
        
        # Re-stack features (B, N, 3)
        seq_features = torch.stack([size, direction, iat], dim=-1)
        
        mask = (seq_nlp[:, :, 0] == 0)      # Padding mask (B, N)
        valid_mask = ~mask
        
        # ===== 1. Feature Extraction =====
        # Byte features (B, N, d_model)
        byte_repr = self.byte_encoder(bytes_nlp)
        
        # Statistical features (B, N, d_model)
        stats_repr = self.stats_encoder(seq_features)
        
        # ===== 2. Feature Aggregation (Packet -> Flow) =====
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)
        
        flow_byte_repr = (byte_repr * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        flow_stats_repr = (stats_repr * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        
        # ===== 3. Dynamic Weighting =====
        # Calculate global statistics for gating (B, 3)
        # Using transformed seq_features here.
        global_stats_input = (seq_features * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        
        # Get weights (B, 2)
        weights = self.gating(global_stats_input)
        w_byte = weights[:, 0:1]
        w_stats = weights[:, 1:2]
        
        # Weighted Fusion
        global_repr = w_byte * flow_byte_repr + w_stats * flow_stats_repr
        
        # ===== 4. Classification =====
        logits = self.classifier(global_repr)
        if return_feats:
            return logits, global_repr
        return logits

    def get_gate_weights(self, batch_data):
        """Helper to visualize/analyze gating weights."""
        with torch.no_grad():
            seq_nlp = batch_data['seq_nlp']
            mask = (seq_nlp[:, :, 0] == 0)
            valid_mask = ~mask
            valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            # Note: Must apply same log1p transform for consistent visualization.
            size = torch.log1p(seq_nlp[:, :, 0].float())
            direction = seq_nlp[:, :, 1].float()
            iat = torch.log1p(seq_nlp[:, :, 2].float())
            seq_features = torch.stack([size, direction, iat], dim=-1)
            
            global_stats_input = (seq_features * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
            return self.gating(global_stats_input)