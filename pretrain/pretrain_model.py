"""
Pre-training Model Wrapper
Adds pre-training heads on top of the original classifier backbone.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('..')

from models import EncryptedTrafficClassifier
from pretrain_tasks import (
    MPMTask,
    MultiModalPredictionTask,
    ContrastiveLearningTask,
    StatisticsReconstructionTask
)
from pretrain_utils import MaskGenerator


class PretrainWrapper(nn.Module):
    """
    Pre-training Model Wrapper
    
    Contains:
    1. Original EncryptedTrafficClassifier (as the backbone network)
    2. Heads for various pre-training tasks
    
    Args:
        backbone_config: Configuration for the backbone network
        mask_prob: Probability for packet-level masking
        use_byte_mask: Whether to use byte-level masking
    """
    def __init__(self, 
                 backbone_config,
                 mask_prob=0.15,
                 use_byte_mask=False):
        super().__init__()
        
        self.mask_prob = mask_prob
        self.use_byte_mask = use_byte_mask
        self.mask_generator = MaskGenerator()
        
        # ===== Backbone Network =====
        # Note: num_classes can be any value during pre-training as the classifier is not used.
        self.backbone = EncryptedTrafficClassifier(
            num_classes=10,  # Placeholder
            **backbone_config
        )
        
        d_packet = backbone_config['d_packet']
        
        # ===== Pre-training Task Heads =====
        self.mpm_task = MPMTask(d_packet=d_packet)
        self.multimodal_task = MultiModalPredictionTask(d_packet=d_packet)
        self.contrastive_task = ContrastiveLearningTask(d_packet=d_packet)
        self.stats_task = StatisticsReconstructionTask(d_packet=d_packet)
        
        # Special token (used for masking)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_packet))
    
    def forward(self, batch_data, task='all'):
        """
        Forward pass (supports different tasks)
        
        Args:
            batch_data: Input batch data
            task: 'mpm' | 'multimodal' | 'contrastive' | 'stats' | 'all'
        
        Returns:
            losses: dict, loss values for each task
        """
        if task == 'all':
            return self._forward_all_tasks(batch_data)
        elif task == 'mpm':
            return self._forward_mpm(batch_data)
        elif task == 'multimodal':
            return self._forward_multimodal(batch_data)
        elif task == 'contrastive':
            return self._forward_contrastive(batch_data)
        elif task == 'stats':
            return self._forward_stats(batch_data)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _forward_all_tasks(self, batch_data):
        all_losses = {}
        
        # 1. Unified computation for Clean Data Backbone output (reused for multimodal and stats)
        # Extract features explicitly to avoid redundant computation in sub-functions
        clean_bytes = batch_data['bytes_nlp']
        clean_seq = batch_data['seq_nlp']
        clean_tcp = batch_data['tcp_data']
        clean_mask = (clean_seq[:, :, 0] == 0)
        
        # Backbone Forward (Clean)
        clean_packet_repr = self.backbone.byte_encoder(clean_bytes)
        clean_seq_features = self.backbone.extract_seq_features(clean_seq, clean_tcp)
        clean_flow_repr = self.backbone.packet_encoder(clean_packet_repr, clean_seq_features, clean_mask)
        
        # 2. MPM Task (Requires Masked Input, calculated independently)
        mpm_losses = self._forward_mpm(batch_data)
        all_losses.update({f'mpm_{k}': v for k, v in mpm_losses.items()})

        # 3. Multimodal Task (Reuse Clean Flow Repr)
        mm_losses = self.multimodal_task(clean_flow_repr, clean_seq, clean_tcp, clean_mask)
        all_losses.update({f'mm_{k}': v for k, v in mm_losses.items()})

        # 4. Statistical Task (Reuse Clean Flow Repr)
        # Calculate global pooling
        valid_mask = ~clean_mask
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        flow_repr_masked = clean_flow_repr * valid_mask.unsqueeze(-1).float()
        global_repr = flow_repr_masked.sum(dim=1) / valid_counts.float()
        
        # Compute target and loss
        from pretrain_utils import compute_flow_statistics
        target_stats = compute_flow_statistics(clean_seq)
        stats_losses = self.stats_task(global_repr, target_stats)
        all_losses.update({f'stats_{k}': v for k, v in stats_losses.items()})

        # 5. Contrastive Learning (Requires Anchor/Positive construction, usually recalculates backbone)
        cl_loss = self._forward_contrastive(batch_data)
        all_losses['contrastive'] = cl_loss

        return all_losses

    def _forward_mpm(self, batch_data):
        """Masked Packet Modeling"""
        bytes_nlp = batch_data['bytes_nlp']
        seq_nlp = batch_data['seq_nlp']
        tcp_data = batch_data['tcp_data']
        
        batch_size, n_packets = seq_nlp.shape[:2]
        device = bytes_nlp.device
        
        # ===== 1. Generate packet-level mask =====
        packet_mask = self.mask_generator.generate_packet_mask(
            batch_size, n_packets, 
            mask_prob=self.mask_prob,
            device=device
        )
        
        # ===== 2. Obtain original representations for all packets =====
        all_packet_repr = self.backbone.byte_encoder(bytes_nlp)
        original_packet_repr = all_packet_repr.detach()
        
        # ===== 3. Apply mask at Embedding level =====
        masked_packet_repr = all_packet_repr.clone()
        mask_token_expanded = self.mask_token.expand(batch_size, n_packets, -1) # Expand mask token
        # Use mask_token for masked positions, keep original for others
        masked_packet_repr = torch.where(
            packet_mask.unsqueeze(-1),
            mask_token_expanded,
            masked_packet_repr
        )
        
        # Pass through packet-level Transformer
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        mask_padding = (seq_nlp[:, :, 0] == 0)
        
        flow_repr = self.backbone.packet_encoder(
            masked_packet_repr, seq_features, mask_padding
        )
        
        # ===== 5. Compute MPM Loss =====
        losses = self.mpm_task(
            packet_repr=flow_repr, # Flow representation derived from masked packets
            original_repr=original_packet_repr, # Original packet representation (Target)
            packet_mask=packet_mask # Masked positions
        )
        
        return losses
    
    def _forward_multimodal(self, batch_data):
        """Multimodal feature prediction"""
        bytes_nlp = batch_data['bytes_nlp']
        seq_nlp = batch_data['seq_nlp']
        tcp_data = batch_data['tcp_data']
        
        # Normal forward pass
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        mask = (seq_nlp[:, :, 0] == 0)
        
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, mask)
        
        # Compute multimodal loss
        losses = self.multimodal_task(flow_repr, seq_nlp, tcp_data, mask)
        
        return losses
    
    def _forward_contrastive(self, batch_data):
        """Contrastive Learning"""
        from pretrain_utils import create_contrastive_pairs
        
        # Create positive pairs
        anchor_data, positive_data = create_contrastive_pairs(batch_data)
        
        # Forward pass to get global representation
        def get_global_repr(data):
            bytes_nlp = data['bytes_nlp']
            seq_nlp = data['seq_nlp']
            tcp_data = data['tcp_data']
            
            packet_repr = self.backbone.byte_encoder(bytes_nlp)
            seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
            mask = (seq_nlp[:, :, 0] == 0)
            
            flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, mask)
            
            # Global pooling
            valid_mask = ~mask
            valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            flow_repr_masked = flow_repr * valid_mask.unsqueeze(-1).float()
            global_repr = flow_repr_masked.sum(dim=1) / valid_counts.float()
            
            return global_repr
        
        anchor_repr = get_global_repr(anchor_data)
        positive_repr = get_global_repr(positive_data)
        
        # Compute contrastive loss
        loss = self.contrastive_task(anchor_repr, positive_repr)
        
        return loss
    
    def _forward_stats(self, batch_data):
        """Statistical feature reconstruction"""
        from pretrain_utils import compute_flow_statistics
        
        bytes_nlp = batch_data['bytes_nlp']
        seq_nlp = batch_data['seq_nlp']
        tcp_data = batch_data['tcp_data']
        
        # Compute target statistics
        target_stats = compute_flow_statistics(seq_nlp)
        
        # Forward pass to get global representation
        packet_repr = self.backbone.byte_encoder(bytes_nlp)
        seq_features = self.backbone.extract_seq_features(seq_nlp, tcp_data)
        mask = (seq_nlp[:, :, 0] == 0)
        
        flow_repr = self.backbone.packet_encoder(packet_repr, seq_features, mask)
        
        # Global pooling
        valid_mask = ~mask
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        flow_repr_masked = flow_repr * valid_mask.unsqueeze(-1).float()
        global_repr = flow_repr_masked.sum(dim=1) / valid_counts.float()
        
        # Compute statistical reconstruction loss
        losses = self.stats_task(global_repr, target_stats)
        
        return losses
    
    def get_backbone(self):
        """Retrieve the backbone network (for saving pre-trained weights)"""
        return self.backbone