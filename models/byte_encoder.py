"""
Byte-level Transformer Encoder
Encodes the first N bytes of each packet.
"""

import torch
import torch.nn as nn
import math
from .utils import PositionalEncoding


class ByteEmbedding(nn.Module):
    """
    Byte Embedding Layer
    
    Args:
        d_model: Embedding dimension.
        vocab_size: Byte vocabulary size (default 256, corresponding to 0-255).
        padding_idx: Index for padding.
    """
    def __init__(self, d_model=128, vocab_size=256, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_bytes), Byte sequence, dtype=long.
        Returns:
            (batch_size, n_bytes, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class ByteLevelTransformer(nn.Module):
    """
    Byte-level Transformer Encoder
    
    Encodes the byte sequence of a single packet.
    
    Args:
        d_byte: Byte embedding dimension.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        d_ff: FFN hidden layer dimension.
        max_bytes: Maximum byte length.
        dropout: Dropout probability.
    """
    def __init__(self, 
                 d_byte=128, 
                 num_layers=3, 
                 num_heads=8, 
                 d_ff=512, 
                 max_bytes=300, 
                 dropout=0.1):
        super().__init__()
        
        self.d_byte = d_byte
        self.max_bytes = max_bytes
        
        # Byte embedding
        self.byte_embedding = ByteEmbedding(d_model=d_byte, vocab_size=256, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_byte, max_len=max_bytes, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_byte,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling method selection
        self.use_cls = True
        if self.use_cls:
            # Use CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_byte))
            # Positional encoding needs one extra position for CLS
            self.pos_encoder = PositionalEncoding(d_byte, max_len=max_bytes+1, dropout=dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_byte, max_len=max_bytes, dropout=dropout)
        
    def forward(self, byte_seq, return_all=False):
        """
        Args:
            byte_seq: (batch_size, n_bytes), dtype=long, byte sequence.
            return_all: Whether to return outputs for all positions.
        
        Returns:
            If use_cls=True and return_all=False:
                (batch_size, d_byte), Packet representation vector.
            Otherwise:
                (batch_size, n_bytes, d_byte)
        """
        batch_size = byte_seq.size(0)
        
        # Byte embedding
        x = self.byte_embedding(byte_seq)  # (B, N, d_byte)
        
        # If using CLS token
        if self.use_cls:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_byte)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d_byte)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask (Optional, can be omitted if all packets are fixed length)
        # src_key_padding_mask = (byte_seq == 0)  # (B, N)
        
        # Transformer encoding
        x = self.transformer(x)  # (B, N+1, d_byte) or (B, N, d_byte)
        
        if self.use_cls and not return_all:
            # Return output of CLS token
            return x[:, 0, :]  # (B, d_byte)
        elif return_all:
            return x
        else:
            # Global average pooling
            return x.mean(dim=1)  # (B, d_byte)


class PacketBytesEncoder(nn.Module):
    """
    Batch encoding for byte sequences of multiple packets.
    
    Args:
        d_byte: Byte embedding dimension.
        d_packet: Output packet representation dimension.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        d_ff: FFN dimension.
        max_bytes: Maximum number of bytes per packet.
        dropout: Dropout probability.
    """
    def __init__(self, 
                 d_byte=128, 
                 d_packet=260, 
                 num_layers=3, 
                 num_heads=8, 
                 d_ff=512, 
                 max_bytes=300, 
                 dropout=0.1):
        super().__init__()
        
        self.d_packet = d_packet
        
        # Byte-level encoder
        self.byte_transformer = ByteLevelTransformer(
            d_byte=d_byte,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_bytes=max_bytes,
            dropout=dropout
        )
        
        # Project to packet representation dimension
        self.projection = nn.Linear(d_byte, d_packet)
        
    def forward(self, packet_bytes):
        """
        Args:
            packet_bytes: (batch_size, n_packets, n_bytes), dtype=long.
        
        Returns:
            (batch_size, n_packets, d_packet), Representation vector for each packet.
        """
        batch_size, n_packets, n_bytes = packet_bytes.shape
        
        # Reshape to (batch_size * n_packets, n_bytes)
        byte_seq = packet_bytes.view(-1, n_bytes)
        
        # Byte-level encoding
        packet_repr = self.byte_transformer(byte_seq)  # (B*N, d_byte)
        
        # Projection
        packet_repr = self.projection(packet_repr)  # (B*N, d_packet)
        
        # Reshape back to (batch_size, n_packets, d_packet)
        packet_repr = packet_repr.view(batch_size, n_packets, self.d_packet)
        
        return packet_repr