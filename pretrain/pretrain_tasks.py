"""
Pre-training Task Definitions
Contains loss functions for various pre-training tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MPMTask(nn.Module):
    """
    Masked Packet Modeling (MPM) Task
    
    Includes:
    1. Packet-level reconstruction (Primary)
    2. Byte-level reconstruction (Auxiliary)
    """
    def __init__(self, d_packet=256, vocab_size=257):
        """
        Args:
            d_packet: Dimension of packet representation
            vocab_size: Byte vocabulary size (256 + 1 MASK token)
        """
        super().__init__()
        
        # Packet reconstruction head
        self.packet_recon_head = nn.Sequential(
            nn.Linear(d_packet, d_packet),
            nn.GELU(),
            nn.Linear(d_packet, d_packet)
        )
        
        # Byte reconstruction head
        self.byte_recon_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.GELU(),
            nn.Linear(d_packet // 2, vocab_size)
        )
    
    def forward(self, packet_repr, original_repr, packet_mask, 
                byte_logits=None, byte_targets=None, byte_mask=None):
        """
        Args:
            packet_repr: (B, N, d_packet) Current packet representations
            original_repr: (B, N, d_packet) Original packet representations (Target)
            packet_mask: (B, N) bool, True indicates masked packets
            byte_logits: (B, N, L, vocab_size) Byte predictions (Optional)
            byte_targets: (B, N, L) Byte targets (Optional)
            byte_mask: (B, N, L) bool, Masked bytes (Optional)
        
        Returns:
            loss_dict: Dictionary containing various losses
        """
        losses = {}
        
        # ===== 1. Packet-level Reconstruction Loss =====
        pred_repr = self.packet_recon_head(packet_repr)  # (B, N, d_packet)
        
        # Calculate loss only for masked packets
        if packet_mask.any():
            pred_masked = pred_repr[packet_mask] 
            target_masked = original_repr[packet_mask]
            
            # Use Cosine Similarity Loss
            loss_packet = 1 - F.cosine_similarity(pred_masked, target_masked, dim=-1).mean()
            losses['packet_recon'] = loss_packet
        else:
            losses['packet_recon'] = torch.tensor(0.0, device=packet_repr.device)
        
        # ===== 2. Byte-level Reconstruction Loss (Optional) =====
        if byte_logits is not None and byte_targets is not None and byte_mask is not None:
            if byte_mask.any():
                logits_flat = byte_logits[byte_mask]  
                targets_flat = byte_targets[byte_mask]
                
                loss_byte = F.cross_entropy(logits_flat, targets_flat)
                losses['byte_recon'] = loss_byte
            else:
                losses['byte_recon'] = torch.tensor(0.0, device=packet_repr.device)
        
        return losses


class MultiModalPredictionTask(nn.Module):
    """
    Multimodal Feature Prediction Task
    Consists of sub-tasks corresponding to multiple attention heads.
    """
    def __init__(self, d_packet=256):
        super().__init__()
        
        # Head 1: Packet type classification (Data vs. ACK)
        self.packet_type_head = nn.Linear(d_packet, 2)
        
        # Head 2: IAT distribution prediction (10 bins)
        self.iat_dist_head = nn.Linear(d_packet, 10)
        
        # Head 3: Packet size regression
        self.size_head = nn.Linear(d_packet, 1)
        
        # Head 4: Direction sequence prediction (3 classes: OUT, IN, PADDING)
        self.direction_head = nn.Linear(d_packet, 3)
        
        # Head 5: TCP state prediction (Request-Response pairs)
        self.tcp_pair_head = nn.Linear(d_packet * 2, 2)
    
    def forward(self, packet_repr, seq_nlp, tcp_data, mask=None):
        """
        Args:
            packet_repr: (B, N, d_packet)
            seq_nlp: (B, N, 3) [size, direction, iat]
            tcp_data: (B, N, 3) [seq, ack, len]
            mask: (B, N) Padding mask
        """
        losses = {}
        batch_size, n_packets, _ = packet_repr.shape
        device = packet_repr.device
        
        # Create validity mask
        if mask is None:
            valid_mask = (seq_nlp[:, :, 0] != 0)
        else:
            valid_mask = ~mask
        
        # ===== Task 1: Packet Type Classification =====
        packet_type_logits = self.packet_type_head(packet_repr)
        packet_type_targets = (tcp_data[:, :, 2] > 0).long() # payload > 0 is data packet
        
        if valid_mask.any():
            loss_type = F.cross_entropy(
                packet_type_logits[valid_mask],
                packet_type_targets[valid_mask]
            )
            losses['packet_type'] = loss_type
        else:
            losses['packet_type'] = torch.tensor(0.0, device=device)
        
        # ===== Task 2: IAT Distribution Prediction =====
        iats = seq_nlp[:, :, 2]
        iat_bins = torch.clamp((iats * 10).long(), 0, 9) # Simple binning
        iat_logits = self.iat_dist_head(packet_repr)
        
        if valid_mask.any():
            loss_iat = F.cross_entropy(
                iat_logits[valid_mask],
                iat_bins[valid_mask]
            )
            losses['iat_dist'] = loss_iat
        else:
            losses['iat_dist'] = torch.tensor(0.0, device=device)
        
        # ===== Task 3: Packet Size Regression =====
        size_pred = self.size_head(packet_repr).squeeze(-1)
        size_target = seq_nlp[:, :, 0]
        log_size_target = torch.log10(size_target + 1.0)
        
        if valid_mask.any():
            loss_size = F.mse_loss(
                size_pred[valid_mask],
                log_size_target[valid_mask]
            )
            losses['size_pred'] = loss_size
        else:
            losses['size_pred'] = torch.tensor(0.0, device=device)
        
        # ===== Task 4: Direction Prediction =====
        # Predict the direction of the next packet
        dir_logits = self.direction_head(packet_repr[:, :-1])
        dir_targets = seq_nlp[:, 1:, 1]
        # Mapping: 1 -> 0 (OUT), -1 -> 1 (IN), 0 -> 2 (PADDING)
        dir_targets = torch.where(dir_targets == 1, 
                                   torch.tensor(0, device=device), 
                                   torch.where(dir_targets == -1, 
                                               torch.tensor(1, device=device), 
                                               torch.tensor(2, device=device)))
        
        valid_mask_dir = valid_mask[:, :-1]
        if valid_mask_dir.any():
            loss_dir = F.cross_entropy(
                dir_logits[valid_mask_dir],
                dir_targets[valid_mask_dir]
            )
            losses['direction'] = loss_dir
        else:
            losses['direction'] = torch.tensor(0.0, device=device)
        
        # ===== Task 5: TCP Request-Response Pair Prediction =====
        directions = seq_nlp[:, :, 1]
        pair_repr = torch.cat([
            packet_repr[:, :-1],
            packet_repr[:, 1:]
        ], dim=-1)
        
        pair_logits = self.tcp_pair_head(pair_repr)
        # Target: 1 if i is OUT and i+1 is IN, else 0
        pair_targets = ((directions[:, :-1] == 1) & (directions[:, 1:] == -1)).long()
        
        valid_mask_pair = valid_mask[:, :-1]
        if valid_mask_pair.any():
            loss_tcp = F.cross_entropy(
                pair_logits[valid_mask_pair],
                pair_targets[valid_mask_pair]
            )
            losses['tcp_pair'] = loss_tcp
        else:
            losses['tcp_pair'] = torch.tensor(0.0, device=device)
        
        return losses


class ContrastiveLearningTask(nn.Module):
    """
    Contrastive Learning Task (InfoNCE Loss)
    """
    def __init__(self, d_packet=256, projection_dim=128, temperature=0.07):
        super().__init__()
        
        self.temperature = temperature
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(d_packet, d_packet),
            nn.ReLU(),
            nn.Linear(d_packet, projection_dim)
        )
    
    def forward(self, anchor_repr, positive_repr):
        """
        Args:
            anchor_repr: (B, d_packet) Global representation of the anchor
            positive_repr: (B, d_packet) Global representation of the positive
        """
        batch_size = anchor_repr.shape[0]
        
        # Projection
        z_anchor = self.projection(anchor_repr)
        z_positive = self.projection(positive_repr)
        
        # L2 Normalization
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_anchor, z_positive.T) / self.temperature
        
        # Diagonal elements represent positive pairs
        labels = torch.arange(batch_size, device=anchor_repr.device)
        
        # Symmetric InfoNCE loss
        loss = (F.cross_entropy(similarity_matrix, labels) + 
                F.cross_entropy(similarity_matrix.T, labels)) / 2
        
        return loss


class StatisticsReconstructionTask(nn.Module):
    """
    Flow Statistics Reconstruction Task
    """
    def __init__(self, d_packet=256):
        super().__init__()
        
        # Packet size histogram prediction (10 bins)
        self.size_hist_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Linear(d_packet // 2, 10),
            nn.Softmax(dim=-1)
        )
        
        # Direction ratio prediction (Upstream vs. Downstream)
        self.direction_ratio_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Linear(d_packet // 2, 2),
            nn.Sigmoid()
        )
        
        # IAT statistics prediction (Mean and Std)
        self.iat_stats_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Linear(d_packet // 2, 2)
        )
    
    def forward(self, global_repr, target_stats):
        """
        Args:
            global_repr: (B, d_packet) Global flow representation
            target_stats: dict containing ground truth statistics
        """
        losses = {}
        
        # ===== 1. Packet Size Distribution =====
        pred_size_hist = self.size_hist_head(global_repr)
        target_size_hist = target_stats['size_histogram']
        
        loss_size_hist = F.kl_div(
            torch.log(pred_size_hist + 1e-8),
            target_size_hist,
            reduction='batchmean'
        )
        losses['size_histogram'] = loss_size_hist
        
        # ===== 2. Direction Ratio =====
        pred_ratio = self.direction_ratio_head(global_repr)
        target_ratio = torch.stack([
            target_stats['up_ratio'],
            target_stats['down_ratio']
        ], dim=-1)
        
        loss_ratio = F.mse_loss(pred_ratio, target_ratio)
        losses['direction_ratio'] = loss_ratio
        
        # ===== 3. IAT Statistics =====
        pred_iat_stats = self.iat_stats_head(global_repr)
        target_iat_stats = torch.stack([
            target_stats['iat_mean'],
            target_stats['iat_std']
        ], dim=-1)
        
        loss_iat = F.mse_loss(pred_iat_stats, target_iat_stats)
        losses['iat_stats'] = loss_iat
        
        return losses