"""
Complete Encrypted Traffic Classifier
Integrates byte-level encoder and packet-level Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .byte_encoder import PacketBytesEncoder
from .packet_attention import PacketTransformerEncoder
from .utils import create_padding_mask


class EncryptedTrafficClassifier(nn.Module):
    """
    Encrypted Traffic Classifier (Complete Model)
    
    Architecture:
    1. Byte-level Transformer: Encodes the first 300 bytes of each packet.
    2. Packet-level Multimodal Transformer: Leverages temporal, size, direction, and other multimodal info.
    3. Classifier: Global pooling + MLP.
    
    Args:
        num_classes: Number of target classes.
        d_byte: Byte embedding dimension.
        d_packet: Packet representation dimension.
        byte_layers: Number of byte-level Transformer layers.
        packet_layers: Number of packet-level Transformer layers.
        num_heads: Number of packet-level Transformer heads (fixed to 5).
        d_ff: FFN dimension.
        max_bytes: Bytes per packet.
        max_packets: Max packets per flow.
        dropout: Dropout probability.
        use_stats: Whether to fuse statistical features.
        stats_dim: Statistical feature dimension.
    """
    def __init__(self,
                 num_classes,
                 d_byte=128,
                 d_packet=260,
                 byte_layers=3,
                 packet_layers=4,
                 num_heads=5,
                 d_ff=1024,
                 max_bytes=300,
                 max_packets=100,
                 dropout=0.1,
                 use_stats=False,
                 stats_dim=36,
                 use_adaptive_gating=True,
                 disabled_head_indices=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_packet = d_packet
        self.use_stats = use_stats
        
        # ========== Layer 1: Byte-level Encoder ==========
        self.byte_encoder = PacketBytesEncoder(
            d_byte=d_byte,
            d_packet=d_packet,
            num_layers=byte_layers,
            num_heads=8,
            d_ff=512,
            max_bytes=max_bytes,
            dropout=dropout
        )
        
        # ========== Layer 2: Packet-level Transformer Encoder ==========
        self.packet_encoder = PacketTransformerEncoder(
            d_model=d_packet,
            num_layers=packet_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_packets=max_packets,
            dropout=dropout,
            use_adaptive_gating=use_adaptive_gating,
            disabled_head_indices=disabled_head_indices
        )
        
        # ========== Statistical Feature Fusion (Optional) ==========
        if use_stats:
            self.stats_proj = nn.Sequential(
                nn.Linear(stats_dim, d_packet),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = d_packet * 2
        else:
            classifier_input_dim = d_packet
        
        # ========== Classifier ==========
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_packet),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_packet, d_packet // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_packet // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def extract_seq_features(self, seq_data, tcp_data):
        """
        Extract sequence features from seq_nlp_view.
        
        Args:
            seq_data: (batch_size, n_packets, 3) [Length, Direction, IAT].
            tcp_data: (batch_size, n_packets, 3) [Seq, Ack, Len].
        
        Returns:
            dict: {
                'timestamps': (B, N) Cumulative timestamps,
                'sizes': (B, N) Packet sizes,
                'directions': (B, N) Directions,
                'tcp_data': (B, N, 3) TCP features
            }
        """
        batch_size, n_packets, _ = seq_data.shape
        
        # Extract features
        sizes = seq_data[:, :, 0]  # (B, N)
        directions = seq_data[:, :, 1]  # (B, N)
        iats = seq_data[:, :, 2]  # (B, N)
        
        # Compute cumulative timestamps
        timestamps = torch.cumsum(iats, dim=1)  # (B, N)
        
        return {
            'timestamps': timestamps,
            'sizes': sizes,
            'directions': directions,
            'tcp_data': tcp_data 
        }
    
    def forward(self, batch_data):
        """
        Forward Pass
        
        Args:
            batch_data: dict containing:
                - 'bytes_nlp': (B, n_packets, n_bytes) Byte data.
                - 'seq_nlp': (B, n_packets, 3) Sequence features [Length, Direction, IAT].
                - 'stats': (B, n_stats) Statistical features (optional).
        
        Returns:
            logits: (batch_size, num_classes) Classification logits.
        """
        bytes_nlp = batch_data['bytes_nlp']
        seq_nlp = batch_data['seq_nlp']
        tcp_data = batch_data['tcp_data']
        
        # ========== Layer 1: Byte-level Encoding ==========
        packet_repr = self.byte_encoder(bytes_nlp)  # (B, N, d_packet)
        
        # ========== Feature Extraction ==========
        seq_features = self.extract_seq_features(seq_nlp, tcp_data)
        
        # ========== Create Padding Mask ==========
        # Identify padding based on packet size (size 0 indicates padding)
        mask = (seq_nlp[:, :, 0] == 0)  # (B, N), True means padding
        
        # ========== Layer 2: Packet-level Transformer Encoding ==========
        flow_repr = self.packet_encoder(packet_repr, seq_features, mask)  # (B, N, d_packet)
        
        # ========== Global Pooling ==========
        # Create mask for valid packets for pooling
        valid_mask = ~mask  # (B, N), True means valid packet
        
        # Calculate valid packet counts per sample
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        
        # Masked Average Pooling
        flow_repr_masked = flow_repr * valid_mask.unsqueeze(-1).float()  # (B, N, d_packet)
        global_repr = flow_repr_masked.sum(dim=1) / valid_counts.float()  # (B, d_packet)
        
        # ========== Fuse Statistical Features (Optional) ==========
        if self.use_stats and 'stats' in batch_data:
            stats = batch_data['stats']  # (B, n_stats)
            stats_repr = self.stats_proj(stats)  # (B, d_packet)
            global_repr = torch.cat([global_repr, stats_repr], dim=-1)  # (B, d_packet*2)
        
        # ========== Classification ==========
        logits = self.classifier(global_repr)  # (B, num_classes)
        
        return logits
    
    def get_attention_weights(self, batch_data, layer_idx=0):
        """
        Retrieve attention weights for visualization.
        
        Args:
            batch_data: Input data.
            layer_idx: Index of the layer to extract weights from.
        
        Returns:
            attention_weights: Attention weights.
        """
        # Attention weights would require PacketTransformerLayer to return them; not used in current pipeline.
        pass