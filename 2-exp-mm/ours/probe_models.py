"""
Model wrappers for multi-modal protocol understanding tasks.
Provides specialized probe heads for each task.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('../..')
from models import EncryptedTrafficClassifier
from finetune.finetune import load_config_from_folder

class BaseProbeWrapper(nn.Module):
    """
    Base probe wrapper.
    Loads pretrained backbone and freezes it by default.
    """
    def __init__(self, backbone, d_packet=260, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.d_packet = d_packet
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _get_flow_representation(self, bytes_nlp, seq_nlp, tcp_data):
        """Get global representation of the flow"""
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        padding_mask = (seq_nlp[:, :, 0] == 0)
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, padding_mask)
        
        # Global pooling
        valid_mask = ~padding_mask
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        flow_repr_masked = flow_repr * valid_mask.unsqueeze(-1).float()
        global_repr = flow_repr_masked.sum(dim=1) / valid_counts.float()
        
        return global_repr, flow_repr
    
    def _get_position_representation(self, bytes_nlp, seq_nlp, tcp_data, position_idx):
        """Get representation for a specific position"""
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        padding_mask = (seq_nlp[:, :, 0] == 0)
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, padding_mask)
        
        # Extract representation at the specified position
        B = flow_repr.shape[0]
        batch_indices = torch.arange(B, device=flow_repr.device)
        position_repr = flow_repr[batch_indices, position_idx]
        
        return position_repr


class DirectionProbeWrapper(BaseProbeWrapper):
    """
    Task 1: Direction Prediction task wrapper.
    """
    def __init__(self, backbone, d_packet=260, freeze_backbone=True):
        super().__init__(backbone, d_packet, freeze_backbone)
        
        # Learnable Mask Token (to replace the prediction position)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_packet) * 0.02)
        
        # Direction prediction head: Binary classification (OUT/IN)
        self.direction_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_packet // 2, 2)
        )
    
    def forward(self, batch):
        bytes_nlp = batch['bytes_nlp']
        seq_nlp = batch['seq_nlp']
        tcp_data = batch['tcp_data']
        pred_idx = batch['pred_idx']
        
        # Get packet representation
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        
        # Replace the target prediction position with Mask Token
        B, N, D = packet_repr.shape
        batch_indices = torch.arange(B, device=packet_repr.device)
        packet_repr[batch_indices, pred_idx] = self.mask_token.squeeze(1)
        
        # Process through Transformer
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        padding_mask = (seq_nlp[:, :, 0] == 0)
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, padding_mask)
        
        # Extract representation for the masked position
        target_repr = flow_repr[batch_indices, pred_idx]
        
        # Predict
        logits = self.direction_head(target_repr)
        
        return logits


class IATAnomalyProbeWrapper(BaseProbeWrapper):
    """
    Task 2: IAT Anomaly Detection task wrapper.
    """
    def __init__(self, backbone, d_packet=260, freeze_backbone=True):
        super().__init__(backbone, d_packet, freeze_backbone)
        
        # Anomaly detection head: Binary classification (Normal/Anomaly)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_packet // 2, 2)
        )
    
    def forward(self, batch):
        bytes_nlp = batch['bytes_nlp']
        seq_nlp = batch['seq_nlp']
        tcp_data = batch['tcp_data']
        
        # Get global representation
        global_repr, _ = self._get_flow_representation(bytes_nlp, seq_nlp, tcp_data)
        
        # Predict
        logits = self.anomaly_head(global_repr)
        
        return logits


class SizePredictionProbeWrapper(BaseProbeWrapper):
    """
    Task 3: Packet Size Prediction task wrapper.
    """
    def __init__(self, backbone, d_packet=260, num_bins=10, freeze_backbone=True):
        super().__init__(backbone, d_packet, freeze_backbone)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_packet) * 0.02)
        
        # Size prediction head: Multi-class classification
        self.size_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_packet // 2, num_bins)
        )
    
    def forward(self, batch):
        bytes_nlp = batch['bytes_nlp']
        seq_nlp = batch['seq_nlp']
        tcp_data = batch['tcp_data']
        pred_idx = batch['pred_idx']
        
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        
        B, N, D = packet_repr.shape
        batch_indices = torch.arange(B, device=packet_repr.device)
        packet_repr[batch_indices, pred_idx] = self.mask_token.squeeze(1)
        
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        padding_mask = (seq_nlp[:, :, 0] == 0)
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, padding_mask)
        
        target_repr = flow_repr[batch_indices, pred_idx]
        logits = self.size_head(target_repr)
        
        return logits


class TCPMatchingProbeWrapper(BaseProbeWrapper):
    """
    Task 4: TCP Request-Response Matching task wrapper.
    """
    def __init__(self, backbone, d_packet=260, freeze_backbone=True):
        super().__init__(backbone, d_packet, freeze_backbone)
        
        # Matching head: Process triplet representations then perform binary classification
        self.matching_head = nn.Sequential(
            nn.Linear(d_packet * 3, d_packet),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Linear(d_packet // 2, 2)
        )
    
    def _get_packet_representation(self, bytes_data, seq_data, tcp_data):
        """Get representation for a single packet"""
        # (B, L) -> (B, 1, L)
        bytes_nlp = bytes_data.unsqueeze(1)
        seq_nlp = seq_data.unsqueeze(1)
        tcp_nlp = tcp_data.unsqueeze(1)
        
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        return packet_repr.squeeze(1)
    
    def forward(self, batch):
        # Get request packet representation
        req_repr = self._get_packet_representation(
            batch['request_bytes'],
            batch['request_seq'],
            batch['request_tcp']
        )
        
        # Get Candidate A representation
        cand_a_repr = self._get_packet_representation(
            batch['cand_a_bytes'],
            batch['cand_a_seq'],
            batch['cand_a_tcp']
        )
        
        # Get Candidate B representation
        cand_b_repr = self._get_packet_representation(
            batch['cand_b_bytes'],
            batch['cand_b_seq'],
            batch['cand_b_tcp']
        )
        
        # Concatenate triplet
        combined_repr = torch.cat([req_repr, cand_a_repr, cand_b_repr], dim=-1)
        logits = self.matching_head(combined_repr)
        
        return logits


class ConsistencyProbeWrapper(BaseProbeWrapper):
    """
    Task 5: Cross-Modal Consistency Check task wrapper.
    """
    def __init__(self, backbone, d_packet=260, freeze_backbone=True):
        super().__init__(backbone, d_packet, freeze_backbone)
        
        # Consistency head: Binary classification (Consistent/Inconsistent)
        self.consistency_head = nn.Sequential(
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_packet // 2, 2)
        )
    
    def forward(self, batch):
        bytes_nlp = batch['bytes_nlp']
        seq_nlp = batch['seq_nlp']
        tcp_data = batch['tcp_data']
        
        global_repr, _ = self._get_flow_representation(bytes_nlp, seq_nlp, tcp_data)
        logits = self.consistency_head(global_repr)
        
        return logits


class MaskedRecoveryProbeWrapper(BaseProbeWrapper):
    """
    Task 6: Masked Context Recovery task wrapper (Original task).
    """
    def __init__(self, backbone, d_packet=260, freeze_backbone=True):
        super().__init__(backbone, d_packet, freeze_backbone)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_packet) * 0.02)
        
        self.size_head = nn.Linear(d_packet, 1)
        self.dir_head = nn.Linear(d_packet, 3)
        self.iat_head = nn.Linear(d_packet, 1)
    
    def forward(self, batch):
        bytes_nlp = batch['bytes_nlp']
        seq_nlp = batch['seq_nlp']
        tcp_data = batch['tcp_data']
        mask_idx = batch['mask_idx']
        
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        
        B, N, D = packet_repr.shape
        batch_indices = torch.arange(B, device=packet_repr.device)
        packet_repr[batch_indices, mask_idx] = self.mask_token.squeeze(1)
        
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        padding_mask = (seq_nlp[:, :, 0] == 0)
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, padding_mask)
        
        target_repr = flow_repr[batch_indices, mask_idx]
        
        size_pred = self.size_head(target_repr).squeeze(-1)
        dir_pred = self.dir_head(target_repr)
        iat_pred = self.iat_head(target_repr).squeeze(-1)
        
        return size_pred, dir_pred, iat_pred


def get_backbone(args, device):
    """Load pretrained backbone model"""
    # 1. Dynamically load configuration
    backbone_config, _, _ = load_config_from_folder(args.pretrain_path)
    print(f"Loaded config from checkpoint: {backbone_config}")
    
    # 2. Initialize model with configuration
    model = EncryptedTrafficClassifier(
        num_classes=10, 
        **backbone_config # Unpack parameters
    ).to(device)
    
    if args.random_initialization:
        print("⚠ Using RANDOM INITIALIZATION (Control Group)")
    else:
        print(f"Loading pretrained weights from {args.pretrain_path}")
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        
        if 'backbone_state_dict' in checkpoint:
            src_state = checkpoint['backbone_state_dict']
        else:
            src_state = checkpoint
        
        state_dict = {k: v for k, v in src_state.items() if 'classifier' not in k}
        model.load_state_dict(state_dict, strict=False)
        print("✓ Pretrained weights loaded")
    
    return model