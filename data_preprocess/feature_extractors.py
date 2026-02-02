import numpy as np
from scapy.all import IP, IPv6, TCP, UDP
from stats_utils import _compute_statistics, _compute_direction_statistics, _compute_flow_statistics, _compute_bidirectional_features, _compute_burst_features

def _mask_ip_and_port(pkt_bytes: bytearray) -> bytearray:
    """
    Mask IP addresses (IPv4/IPv6) and port numbers (TCP/UDP).
    
    Args:
        pkt_bytes: Bytearray of the IP packet (mutable).
    
    Returns:
        bytearray: Masked bytearray.
    """
    if len(pkt_bytes) < 1:
        return pkt_bytes

    # Get IP version (top 4 bits of the first byte)
    version = pkt_bytes[0] >> 4
    
    payload_start = 0
    protocol = 0

    # ==========================
    # IPv4 Processing
    # ==========================
    if version == 4:
        if len(pkt_bytes) < 20: # Minimum IPv4 header length
            return pkt_bytes
            
        # Get Header Length (IHL * 4)
        ihl = pkt_bytes[0] & 0x0F
        payload_start = ihl * 4
        
        # Get Protocol Type (Byte 9)
        protocol = pkt_bytes[9]
        
        # Mask Source IP (Bytes 12-15)
        pkt_bytes[12:16] = b'\x00\x00\x00\x00'
        # Mask Destination IP (Bytes 16-19)
        pkt_bytes[16:20] = b'\x00\x00\x00\x00'

    # ==========================
    # IPv6 Processing
    # ==========================
    elif version == 6:
        if len(pkt_bytes) < 40: # Fixed IPv6 header length
            return pkt_bytes
            
        # Fixed length is 40 bytes 
        # (Note: IPv6 extension headers are not handled here)
        payload_start = 40 
        
        # Get Next Header type (Byte 6, equivalent to IPv4 Protocol)
        protocol = pkt_bytes[6]
        
        # Mask Source IP (Bytes 8-23, 16 bytes)
        pkt_bytes[8:24] = b'\x00' * 16
        # Mask Destination IP (Bytes 24-39, 16 bytes)
        pkt_bytes[24:40] = b'\x00' * 16
        
    else:
        # Non-IP packets or unrecognized versions are left untouched
        return pkt_bytes

    # ==========================
    # Port and Checksum Processing (TCP/UDP)
    # ==========================
    # TCP (6) or UDP (17)
    if protocol in [6, 17] and len(pkt_bytes) >= payload_start + 4:
        # Mask Source Port (Payload offset 0-1)
        pkt_bytes[payload_start : payload_start + 2] = b'\x00\x00'
        # Mask Destination Port (Payload offset 2-3)
        pkt_bytes[payload_start + 2 : payload_start + 4] = b'\x00\x00'
        
        # Zero out checksums to prevent potential data leakage
        if protocol == 17 and len(pkt_bytes) >= payload_start + 8:  # UDP
            # UDP checksum at offset 6-7
            pkt_bytes[payload_start + 6 : payload_start + 8] = b'\x00\x00'
            
        elif protocol == 6 and len(pkt_bytes) >= payload_start + 18:  # TCP
            # TCP checksum at offset 16-17
            pkt_bytes[payload_start + 16 : payload_start + 18] = b'\x00\x00'
    
    return pkt_bytes

def extract_bytes_nlp_view(packets, n_packets=100, max_bytes_per_packet=768, mask_ip_port=True):
    """
    Byte Modality - NLP View

    Extracts the first max_bytes_per_packet from the first n_packets.
    Extraction begins from the IP layer.

    Args:
        packets: List of packets parsed by rdpcap.
        n_packets: Number of packets to retain (default 100).
        max_bytes_per_packet: Max bytes per packet (default 768).

    Returns:
        np.ndarray: shape (n_packets, max_bytes_per_packet), dtype=uint8
    """
    all_packet_bytes = []
    # Ensure only IP packets are processed
    ip_packets = [p for p in packets if IP in p]

    # Iterate through the first n_packets
    for p in ip_packets[:n_packets]:
        # Extract bytes starting from the IP layer
        try:
            pkt_bytes = bytearray(bytes(p[IP]))
            
            if mask_ip_port and len(pkt_bytes) >= 20:
                # Mask IP addresses and ports
                pkt_bytes = _mask_ip_and_port(pkt_bytes)
            
            pkt_bytes = bytes(pkt_bytes)
        except Exception:
            pkt_bytes = b""
        
        # Truncate or pad to max_bytes_per_packet
        if len(pkt_bytes) >= max_bytes_per_packet:
            pkt_arr = np.frombuffer(pkt_bytes[:max_bytes_per_packet], dtype=np.uint8)
        else:
            padding = max_bytes_per_packet - len(pkt_bytes)
            pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
            pkt_arr = np.pad(pkt_arr, (0, padding), "constant", constant_values=0)
        all_packet_bytes.append(pkt_arr)

    # Pad the packet count
    while len(all_packet_bytes) < n_packets:
        all_packet_bytes.append(np.zeros(max_bytes_per_packet, dtype=np.uint8))
    
    return np.stack(all_packet_bytes) # (n_packets, max_bytes_per_packet) 


# ============================================================================
# Sequence Modality
# ============================================================================

def extract_sequence_nlp_view(packets, n_packets=100):
    """
    Sequence Modality - NLP View

    Extracts packet sequence features: (length, direction, IAT) and TCP-related info.

    Args:
        packets: List of packets parsed by rdpcap.
        n_packets: Number of packets used (default 100).

    Returns:
        tuple: (seq_data, tcp_data)
            - seq_data: np.ndarray, shape (n_packets, 3), dtype=float32
                        Each row is [length, direction, IAT]
            - tcp_data: np.ndarray, shape (n_packets, 3), dtype=float32
                        Each row is [TCP Seq, TCP Ack, Payload Len]
    """
    # Filter IP packets
    ip_packets = [p for p in packets if IP in p or IPv6 in p]

    if not ip_packets:
        # Return zero-padded arrays if no IP packets exist
        seq_zeros = np.zeros((n_packets, 3), dtype=np.float32)
        tcp_zeros = np.zeros((n_packets, 3), dtype=np.float32)
        return seq_zeros, tcp_zeros

    # Assume the source IP of the first packet is the client
    if IP in ip_packets[0]:
        client_ip = ip_packets[0][IP].src
    else:
        client_ip = ip_packets[0][IPv6].src

    seq_data = []
    tcp_seqs = []
    tcp_acks = []
    payload_lens = []
    last_time = 0

    for packet in ip_packets[:n_packets]:
        # Get packet length and source IP
        if IP in packet:
            pkt_len = packet[IP].len
            src_ip = packet[IP].src
        elif IPv6 in packet:
            pkt_len = len(packet)  # Use total length for IPv6
            src_ip = packet[IPv6].src
        else:
            continue

        # Determine direction: 1 for outbound (client), -1 for inbound (server)
        direction = 1 if src_ip == client_ip else -1

        # Calculate Inter-Arrival Time (IAT)
        time = float(packet.time)
        iat = time - last_time if last_time != 0 else 0
        last_time = time

        seq_data.append([pkt_len, direction, iat])
        
        # ===== Extract TCP Information =====
        if TCP in packet:
            tcp_seqs.append(packet[TCP].seq)
            tcp_acks.append(packet[TCP].ack)
            # Ensure payload exists
            payload = packet[TCP].payload
            payload_len = len(payload) if payload else 0
            payload_lens.append(payload_len)
        else:
            # Pad with 0 for non-TCP packets
            tcp_seqs.append(0)
            tcp_acks.append(0)
            payload_lens.append(0)

    # Convert to numpy arrays
    seq_array = np.array(seq_data, dtype=np.float32)
    tcp_array = np.array([tcp_seqs, tcp_acks, payload_lens], dtype=np.float32).T

    # Pad to n_packets
    if len(seq_array) < n_packets:
        padding_len = n_packets - len(seq_array)
        seq_array = np.vstack([seq_array, np.zeros((padding_len, 3), dtype=np.float32)])
        tcp_array = np.vstack([tcp_array, np.zeros((padding_len, 3), dtype=np.float32)])

    return seq_array, tcp_array


# ============================================================================
# Statistical Modality
# ============================================================================

def extract_stats_view(packets, n_stats=None):
    """
    Statistical Modality - Enhanced Version

    Extracts flow-level multi-dimensional statistical features, including:
    - Packet Length Stats (7)
    - IAT Stats (7)
    - Direction Stats (6)
    - Flow-level Macro Features (4)
    - Bidirectional Traffic Features (8)
    - Burstiness Features (4)

    Total: 36 features
    """
    DEFAULT_N_STATS = 36
    n_stats = n_stats or DEFAULT_N_STATS

    if not packets or len(packets) == 0:
        return np.zeros(n_stats, dtype=np.float32)

    try:
        # ========== Data Preparation ==========
        times = np.array([float(p.time) for p in packets])
        lengths = []
        directions = []  # 1 for outbound, -1 for inbound

        local_ip = None
        if IP in packets[0]:
            local_ip = packets[0][IP].src
        elif IPv6 in packets[0]:
            local_ip = packets[0][IPv6].src


        for p in packets:
            if IP in p:
                lengths.append(p[IP].len)
                if local_ip:
                    directions.append(1 if p[IP].src == local_ip else -1)
                else:
                    directions.append(1) # default
            elif IPv6 in p:
                lengths.append(len(p))
                if local_ip:
                    directions.append(1 if p[IPv6].src == local_ip else -1)
                else:
                    directions.append(1)

        if not lengths:
            return np.zeros(n_stats, dtype=np.float32)

        lengths = np.array(lengths, dtype=np.float32)
        directions = np.array(directions, dtype=np.int8)

        # Calculate IAT
        iats = np.diff(times)
        if len(iats) == 0:
            iats = np.array([0.0])

        # ========== Feature Calculation ==========
        features = []

        # 1. Packet Length Stats (7)
        features.extend(_compute_statistics(lengths))

        # 2. IAT Stats (7)
        features.extend(_compute_statistics(iats))

        # 3. Direction Stats (6)
        features.extend(_compute_direction_statistics(lengths, directions))

        # 4. Flow-level Macro Features (4)
        features.extend(_compute_flow_statistics(lengths, times))

        # 5. Bidirectional Traffic Features (8)
        features.extend(_compute_bidirectional_features(lengths, directions, times))

        # 6. Burstiness Features (4)
        features.extend(_compute_burst_features(lengths, times, directions))

        # Convert and truncate/pad
        features = np.array(features, dtype=np.float32)
        if len(features) < n_stats:
            features = np.pad(features, (0, n_stats - len(features)), "constant")
        else:
            features = features[:n_stats]
        return features

    except Exception as e:
        print(f"Error extracting stats: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(n_stats, dtype=np.float32)

# ============================================================================
# Graph Construction Module
# ============================================================================

def build_multi_relation_graph(seq_data, packets, n_packets=100, 
                               size_similar_threshold=0.85, 
                               window_size=20):
    """
    Constructs a multi-relational graph.
    
    Edge Type Definitions:
    - 0: R_temporal (Temporal Edge) - connects packets consecutive in time
    - 1: R_response (Response Edge) - connects request/response based on TCP Seq/Ack
    - 2: R_direction (Direction Edge) - connects consecutive packets in the same direction
    - 3: R_size_similar (Size Similarity Edge) - connects packets with similar sizes
    
    Args:
        seq_data: np.ndarray, [length, direction, IAT]
        packets: list, raw packets for TCP info extraction
        n_packets: int, total nodes (default 100)
        size_similar_threshold: float, threshold for size similarity
        window_size: int, sliding window for size similarity edges
    
    Returns:
        edge_index: np.ndarray, shape (2, num_edges), dtype=int32
        edge_type: np.ndarray, shape (num_edges,), dtype=int8
    """
    edges = []
    edge_types = []
    
    # Identify valid packets (non-padded, length > 0)
    valid_mask = seq_data[:, 0] > 0
    n_valid = int(np.sum(valid_mask))
    
    if n_valid < 2:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.int8)
    
    # ========== 1. R_temporal: Temporal Edges (adjacent packets) ==========
    for i in range(n_valid - 1):
        edges.append([i, i + 1])
        edge_types.append(0)
    
    # ========== 2. R_direction: Direction Edges (same direction, adjacent) ==========
    for i in range(n_valid - 1):
        dir_i = seq_data[i, 1]
        dir_j = seq_data[i + 1, 1]
        if dir_i == dir_j and dir_i != 0:
            edges.append([i, i + 1])
            edge_types.append(2)

    # ========== 3. R_response: Response Edges (TCP Seq/Ack based) ==========
    try:
        tcp_info = []  # [(packet_idx, seq, ack)]
        for i in range(min(len(packets), n_valid)):
            p = packets[i]
            if TCP in p:
                tcp_info.append({
                    'idx': i,
                    'seq': p[TCP].seq,
                    'ack': p[TCP].ack,
                    'payload_len': len(p[TCP].payload) if p[TCP].payload else 0
                })
        
        for i, info_i in enumerate(tcp_info):
            seq_i = info_i['seq']
            idx_i = info_i['idx']
            payload_i = info_i['payload_len']
            expected_ack = seq_i + max(1, payload_i)
            
            for j in range(i + 1, len(tcp_info)):
                info_j = tcp_info[j]
                idx_j = info_j['idx']
                ack_j = info_j['ack']
                
                if ack_j > 0 and abs(ack_j - expected_ack) < 1500:
                    edges.append([idx_i, idx_j])
                    edge_types.append(1)
                    break 
    except Exception:
        pass
    
    # ========== 4. R_size_similar: Size Similarity Edges (within window) ==========
    for i in range(n_valid):
        len_i = seq_data[i, 0]
        if len_i <= 0: continue
        
        for j in range(i + 1, min(i + window_size, n_valid)):
            len_j = seq_data[j, 0]
            if len_j <= 0: continue
            
            similarity = min(len_i, len_j) / max(len_i, len_j)
            if similarity >= size_similar_threshold:
                # Add bidirectional edges
                edges.append([i, j]); edge_types.append(3)
                edges.append([j, i]); edge_types.append(3)
    
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.int8)
    
    edge_index = np.array(edges, dtype=np.int32).T
    edge_type = np.array(edge_types, dtype=np.int8)
    
    return edge_index, edge_type