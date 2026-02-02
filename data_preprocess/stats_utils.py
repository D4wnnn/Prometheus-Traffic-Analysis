import numpy as np

def _compute_statistics(values):
    """
    Computes 7 statistical features: Mean, Std, Min, Max, Q1, Q2 (Median), Q3.
    
    Args:
        values: 1D numpy array.
    
    Returns:
        list: Contains 7 statistical values.
    """
    if len(values) == 0:
        return [0.0] * 7
    
    return [
        float(np.mean(values)),           # Mean
        float(np.std(values)),            # Standard Deviation
        float(np.min(values)),            # Minimum
        float(np.max(values)),            # Maximum
        float(np.percentile(values, 25)), # 1st Quartile (Q1)
        float(np.percentile(values, 50)), # Median (Q2)
        float(np.percentile(values, 75)), # 3rd Quartile (Q3)
    ]


def _compute_direction_statistics(lengths, directions):
    """
    Computes 6 direction-related statistical features.
    
    Args:
        lengths: Array of packet lengths.
        directions: Array of directions (1 for outbound, -1 for inbound).
    
    Returns:
        list: Contains 6 directional statistical values.
    """
    if len(directions) == 0:
        return [0.0] * 6
    
    # Separate inbound and outbound
    outbound_mask = directions == 1
    inbound_mask = directions == -1
    
    n_outbound = np.sum(outbound_mask)
    n_inbound = np.sum(inbound_mask)
    total_packets = len(directions)
    
    outbound_bytes = np.sum(lengths[outbound_mask]) if n_outbound > 0 else 0
    inbound_bytes = np.sum(lengths[inbound_mask]) if n_inbound > 0 else 0
    total_bytes = np.sum(lengths)
    
    # Avoid division by zero
    total_packets = max(total_packets, 1)
    total_bytes = max(total_bytes, 1)
    
    # Calculate ratios
    outbound_packet_ratio = n_outbound / total_packets
    inbound_packet_ratio = n_inbound / total_packets
    outbound_byte_ratio = outbound_bytes / total_bytes
    inbound_byte_ratio = inbound_bytes / total_bytes
    
    # Calculate differences (log-transformed)
    packet_diff = n_outbound - n_inbound
    byte_diff = outbound_bytes - inbound_bytes
    
    packet_diff_transformed = np.sign(packet_diff) * np.log1p(np.abs(packet_diff))
    byte_diff_transformed = np.sign(byte_diff) * np.log1p(np.abs(byte_diff))
    
    return [
        float(outbound_packet_ratio),    # Outbound packet ratio
        float(inbound_packet_ratio),     # Inbound packet ratio
        float(outbound_byte_ratio),      # Outbound byte ratio
        float(inbound_byte_ratio),       # Inbound byte ratio
        float(packet_diff_transformed),  # Packet diff (log-transformed)
        float(byte_diff_transformed),    # Byte diff (log-transformed)
    ]


def _compute_flow_statistics(lengths, times):
    """
    Computes 4 flow-level macro features.
    
    Args:
        lengths: Array of packet lengths.
        times: Array of timestamps.
    
    Returns:
        list: Contains 4 flow statistics.
    """
    total_packets = len(lengths)
    total_bytes = float(np.sum(lengths))
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    
    duration = max(duration, 1e-6)
    
    throughput = total_bytes / duration     # Throughput (Bytes/sec)
    packet_rate = total_packets / duration  # Packet rate (Packets/sec)
    
    return [
        float(total_packets),  # Total packets
        total_bytes,           # Total bytes
        duration,              # Flow duration
        throughput,            # Throughput
    ]


def _compute_bidirectional_features(lengths, directions, times):
    """
    Computes 8 bidirectional traffic features.
    
    Args:
        lengths: Array of packet lengths.
        directions: Array of directions.
        times: Array of timestamps.
    
    Returns:
        list: Contains 8 bidirectional features.
    """
    if len(directions) == 0:
        return [0.0] * 8
    
    outbound_mask = directions == 1
    inbound_mask = directions == -1
    
    # Outbound stats
    outbound_lengths = lengths[outbound_mask]
    outbound_mean = float(np.mean(outbound_lengths)) if len(outbound_lengths) > 0 else 0.0
    outbound_std = float(np.std(outbound_lengths)) if len(outbound_lengths) > 0 else 0.0
    outbound_max = float(np.max(outbound_lengths)) if len(outbound_lengths) > 0 else 0.0
    
    # Inbound stats
    inbound_lengths = lengths[inbound_mask]
    inbound_mean = float(np.mean(inbound_lengths)) if len(inbound_lengths) > 0 else 0.0
    inbound_std = float(np.std(inbound_lengths)) if len(inbound_lengths) > 0 else 0.0
    inbound_max = float(np.max(inbound_lengths)) if len(inbound_lengths) > 0 else 0.0
    
    # Bidirectional IAT
    if len(times) > 1:
        outbound_times = times[:-1][outbound_mask[:-1]]
        outbound_iats = np.diff(outbound_times) if len(outbound_times) > 1 else np.array([0.0])
        outbound_iat_mean = float(np.mean(outbound_iats)) if len(outbound_iats) > 0 else 0.0
        
        inbound_times = times[:-1][inbound_mask[:-1]]
        inbound_iats = np.diff(inbound_times) if len(inbound_times) > 1 else np.array([0.0])
        inbound_iat_mean = float(np.mean(inbound_iats)) if len(inbound_iats) > 0 else 0.0
    else:
        outbound_iat_mean = 0.0
        inbound_iat_mean = 0.0
    
    return [
        outbound_mean,      # Mean outbound packet length
        outbound_std,       # Std outbound packet length
        outbound_max,       # Max outbound packet length
        outbound_iat_mean,  # Mean outbound IAT
        inbound_mean,       # Mean inbound packet length
        inbound_std,        # Std inbound packet length
        inbound_max,        # Max inbound packet length
        inbound_iat_mean,   # Mean inbound IAT
    ]


def _compute_burst_features(lengths, times, directions):
    """
    Computes 4 burstiness features.
    
    Burst Definition: Inter-arrival time < threshold (e.g., 1ms) 
    and same packet direction.
    
    Args:
        lengths: Array of packet lengths.
        times: Array of timestamps.
        directions: Array of directions.
    
    Returns:
        list: Contains 4 burstiness features.
    """
    if len(times) < 2:
        return [0.0] * 4
    
    BURST_THRESHOLD = 0.001  # 1ms
    
    iats = np.diff(times)
    
    # Identify bursts: IAT < threshold AND same direction
    burst_mask = (iats < BURST_THRESHOLD) & (directions[:-1] == directions[1:])
    
    if not np.any(burst_mask):
        return [0.0] * 4
    
    burst_indices = np.where(burst_mask)[0]
    
    # Group consecutive burst indices
    bursts = []
    current_burst = [burst_indices[0]]
    
    for i in range(1, len(burst_indices)):
        if burst_indices[i] == burst_indices[i-1] + 1:
            current_burst.append(burst_indices[i])
        else:
            bursts.append(current_burst)
            current_burst = [burst_indices[i]]
    bursts.append(current_burst)
    
    # Calculate burst features
    burst_sizes = [len(b) + 1 for b in bursts]  
    burst_bytes = [np.sum(lengths[b[0]:b[-1]+2]) for b in bursts]  
    
    max_burst_size = float(np.max(burst_sizes))
    mean_burst_size = float(np.mean(burst_sizes))
    max_burst_bytes = float(np.max(burst_bytes))
    burst_count = float(len(bursts))
    
    return [
        max_burst_size,    # Max packets in a burst
        mean_burst_size,   # Mean packets in a burst
        max_burst_bytes,   # Max bytes in a burst
        burst_count,       # Total burst count
    ]