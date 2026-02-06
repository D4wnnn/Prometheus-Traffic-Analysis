import os
import shutil
from pathlib import Path
from datetime import datetime
import logging
import struct
import random

logger = logging.getLogger(__name__)

class FlowInfo:
    """Class to store flow information"""
    def __init__(self, filepath, start_time, packet_count=0, size=0):
        self.filepath = Path(filepath)
        self.start_time = start_time
        self.packet_count = packet_count
        self.size = size
        self.filename = self.filepath.name
    
    def __repr__(self):
        return f"FlowInfo({self.filename}, {self.start_time})"
    
    def __lt__(self, other):
        """Used for sorting flows chronologically"""
        return self.start_time < other.start_time

def get_pcap_timestamp_direct(pcap_path):
    """Directly read the timestamp of the first packet from PCAP (no pyshark dependency)"""
    try:
        with open(pcap_path, 'rb') as f:
            # Read PCAP file header (24 bytes)
            pcap_header = f.read(24)
            if len(pcap_header) < 24:
                return None
            
            # Check magic number to determine endianness
            magic = struct.unpack('I', pcap_header[:4])[0]
            if magic == 0xa1b2c3d4:
                # Standard endianness
                endian = '<'
            elif magic == 0xd4c3b2a1:
                # Reverse endianness
                endian = '>'
            else:
                logger.warning(f"Invalid PCAP file: {pcap_path}")
                return None
            
            # Read first packet header (16 bytes)
            packet_header = f.read(16)
            if len(packet_header) < 16:
                return None
            
            # Parse timestamp
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', packet_header)
            
            # Convert to datetime object
            timestamp = ts_sec + ts_usec / 1000000.0
            return datetime.fromtimestamp(timestamp)
            
    except Exception as e:
        logger.error(f"Failed to read PCAP timestamp {pcap_path}: {e}")
        return None

def get_flow_start_time_pyshark(pcap_path):
    """Backup method: Use pyshark to get the timestamp of the first packet"""
    try:
        import pyshark
        
        # Ensure absolute path
        pcap_path = Path(pcap_path).resolve()
        
        # Check file existence and size
        if not pcap_path.exists():
            logger.error(f"File does not exist: {pcap_path}")
            return None
            
        file_size = pcap_path.stat().st_size
        if file_size == 0:
            logger.warning(f"Empty file: {pcap_path}")
            return None
        
        logger.debug(f"Attempting to read file: {pcap_path} (Size: {file_size} bytes)")
        
        # Use pyshark to read capture
        cap = pyshark.FileCapture(
            str(pcap_path),
            keep_packets=False,
            use_json=True,
            include_raw=False
        )
        
        # Read the first packet
        first_packet = None
        packet_count = 0
        for packet in cap:
            packet_count += 1
            if packet_count == 1:
                first_packet = packet
                break
        
        cap.close()
        
        if first_packet:
            # Try different timestamp attributes
            timestamp = None
            
            # Method 1: sniff_timestamp
            if hasattr(first_packet, 'sniff_timestamp'):
                timestamp = float(first_packet.sniff_timestamp)
            # Method 2: frame_info
            elif hasattr(first_packet, 'frame_info'):
                if hasattr(first_packet.frame_info, 'time_epoch'):
                    timestamp = float(first_packet.frame_info.time_epoch)
            # Method 3: sniff_time
            elif hasattr(first_packet, 'sniff_time'):
                timestamp = first_packet.sniff_time.timestamp()
            
            if timestamp:
                return datetime.fromtimestamp(timestamp)
            else:
                logger.warning(f"Unable to extract timestamp from packet: {pcap_path}")
                return None
        else:
            logger.warning(f"No packets found in file: {pcap_path}")
            return None
            
    except Exception as e:
        logger.error(f"Pyshark read failed for {pcap_path}: {e}")
        return None

def get_flow_start_time(pcap_path):
    """Comprehensive method to get the start timestamp of a PCAP file"""
    # Try direct read first (faster and more reliable)
    timestamp = get_pcap_timestamp_direct(pcap_path)
    
    # If failed, fallback to pyshark
    if timestamp is None:
        logger.debug(f"Direct read failed, falling back to pyshark: {pcap_path}")
        timestamp = get_flow_start_time_pyshark(pcap_path)
    
    return timestamp

def get_pcap_packet_count_direct(pcap_path):
    """Directly read packet count from PCAP file (no pyshark dependency)"""
    try:
        with open(pcap_path, 'rb') as f:
            pcap_header = f.read(24)
            if len(pcap_header) < 24:
                return 0
            
            magic = struct.unpack('I', pcap_header[:4])[0]
            if magic == 0xa1b2c3d4:
                endian = '<'
            elif magic == 0xd4c3b2a1:
                endian = '>'
            else:
                logger.warning(f"Invalid PCAP file: {pcap_path}")
                return 0
            
            packet_count = 0
            # Read packets one by one
            while True:
                packet_header = f.read(16)
                if len(packet_header) < 16:
                    break
                
                # Parse packet header to get length
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', packet_header)
                
                # Skip packet data
                f.seek(incl_len, 1)  # Skip incl_len bytes from current position
                
                packet_count += 1
            
            return packet_count
            
    except Exception as e:
        logger.error(f"Failed to read packet count for {pcap_path}: {e}")
        return 0

def get_flow_info(pcap_path):
    """Retrieve detailed information about a flow"""
    pcap_path = Path(pcap_path)
    
    if not pcap_path.exists():
        logger.error(f"File not found: {pcap_path}")
        return None
    
    size = pcap_path.stat().st_size
    if size == 0:
        logger.warning(f"Skipping empty file: {pcap_path}")
        return None
    
    start_time = get_flow_start_time(pcap_path)
    if start_time is None:
        # Fallback to file modification time if timestamp cannot be extracted
        logger.warning(f"Could not extract timestamp, using file mtime as fallback: {pcap_path}")
        start_time = datetime.fromtimestamp(pcap_path.stat().st_mtime)
    
    packet_count = get_pcap_packet_count_direct(pcap_path)
    return FlowInfo(pcap_path, start_time, packet_count, size)

def collect_flows(flow_dir):
    """Collect all flow files in a directory and extract information"""
    flow_dir = Path(flow_dir)
    flows = []
    
    # Identify all PCAP files
    pcap_patterns = ['*.pcap', '*.cap', '*.pcapng']
    pcap_files = []
    for pattern in pcap_patterns:
        pcap_files.extend(flow_dir.glob(pattern))
    
    if not pcap_files:
        logger.warning(f"No PCAP files found in directory {flow_dir}")
        return flows
    
    logger.info(f"Collecting information for {len(pcap_files)} flows...")
    
    valid_count = 0
    skip_count = 0
    error_count = 0
    
    for i, pcap_file in enumerate(pcap_files, 1):
        if i % 10 == 0:
            logger.debug(f"Progress: {i}/{len(pcap_files)}")
        
        try:
            flow_info = get_flow_info(pcap_file)
            if flow_info:
                # Filter flows based on size/packet constraints from config
                from config import MIN_FLOW_PACKETS, MAX_FLOW_SIZE
                if flow_info.size > MAX_FLOW_SIZE:
                    logger.debug(f"Skipping oversized flow: {flow_info.filename} ({flow_info.size} bytes)")
                    skip_count += 1
                    continue
                if flow_info.packet_count < MIN_FLOW_PACKETS:
                    logger.info(f"Skipping undersized flow: {flow_info.filename} ({flow_info.size} bytes)")
                    skip_count += 1
                    continue
                    
                flows.append(flow_info)
                valid_count += 1
            else:
                error_count += 1
                logger.warning(f"Could not process flow file: {pcap_file}")
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing flow file {pcap_file}: {e}")
    
    logger.info(f"Collection complete: Valid={valid_count}, Skipped={skip_count}, Errors={error_count}")
    return flows

def split_flows_by_time(flows, train_ratio=0.7, val_ratio=0.15):
    """
    Split flows into training, validation, and test sets chronologically.
    
    Args:
        flows: List of FlowInfo objects
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        train_flows, val_flows, test_flows: Three split lists
    """
    if not flows:
        return [], [], []
    
    # Sort by start time
    sorted_flows = sorted(flows, key=lambda x: x.start_time)
    total_count = len(sorted_flows)
    
    # Calculate split indices
    train_end_idx = int(total_count * train_ratio)
    val_end_idx = int(total_count * (train_ratio + val_ratio))
    
    # Ensure at least 1 sample per set if total count >= 3
    if total_count >= 3:
        if train_end_idx == 0:
            train_end_idx = 1
        if val_end_idx <= train_end_idx:
            val_end_idx = train_end_idx + 1
        if val_end_idx >= total_count:
            val_end_idx = total_count - 1
    
    train_flows = sorted_flows[:train_end_idx]
    val_flows = sorted_flows[train_end_idx:val_end_idx]
    test_flows = sorted_flows[val_end_idx:]
    
    logger.info(f"Time split: Train {len(train_flows)}, Val {len(val_flows)}, Test {len(test_flows)}")
    
    if train_flows:
        logger.info(f"Train time range: {train_flows[0].start_time} - {train_flows[-1].start_time}")
    if val_flows:
        logger.info(f"Val time range: {val_flows[0].start_time} - {val_flows[-1].start_time}")
    if test_flows:
        logger.info(f"Test time range: {test_flows[0].start_time} - {test_flows[-1].start_time}")
    
    return train_flows, val_flows, test_flows

def split_flows_randomly(flows, train_ratio=0.7, val_ratio=0.15, seed=None):
    """Split flows into sets randomly"""
    if not flows:
        return [], [], []
    
    flows_copy = flows.copy()
    if seed is not None:
        random.seed(seed)
    
    random.shuffle(flows_copy)
    total_count = len(flows_copy)
    
    train_end_idx = int(total_count * train_ratio)
    val_end_idx = int(total_count * (train_ratio + val_ratio))
    
    # --- Intelligent boundary checking ---
    if total_count > 0:
        if train_ratio > 0 and train_end_idx == 0:
            train_end_idx = 1
            
        if val_ratio > 0 and val_end_idx <= train_end_idx:
            val_end_idx = min(train_end_idx + 1, total_count)
            
        # Ensure test set remains if intended (sum of ratios < 1.0)
        if (train_ratio + val_ratio) < 0.999 and val_end_idx >= total_count:
            val_end_idx = total_count - 1
        
        elif val_end_idx > total_count:
            val_end_idx = total_count

    train_flows = flows_copy[:train_end_idx]
    val_flows = flows_copy[train_end_idx:val_end_idx]
    test_flows = flows_copy[val_end_idx:]
    
    logger.info(f"Random split: Train {len(train_flows)}, Val {len(val_flows)}, Test {len(test_flows)}")
    return train_flows, val_flows, test_flows

def save_flows_to_directory(flows, target_dir):
    """Copy flow files to target directory"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for flow_info in flows:
        try:
            source_path = flow_info.filepath.resolve()
            target_path = target_dir / flow_info.filename
            
            if target_path.exists():
                logger.debug(f"Target file exists, overwriting: {target_path}")
            
            shutil.copy2(source_path, target_path)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to copy file {flow_info.filename}: {e}")
    
    logger.info(f"Saved {success_count}/{len(flows)} flows to {target_dir}")
    return success_count

def save_flows_to_directory_with_prefix(flows, target_dir, prefix=""):
    """Copy flow files to target directory with a filename prefix"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, flow_info in enumerate(flows, 1):
        try:
            source_path = flow_info.filepath.resolve()
            
            # Construct new filename: prefix_originalname
            if prefix:
                original_name = flow_info.filename
                suffix = ''.join(source_path.suffixes) # Handle multiple extensions like .pcap.gz
                name_without_suffix = original_name[:-len(suffix)] if suffix else original_name
                new_filename = f"{prefix}_{name_without_suffix}{suffix}"
            else:
                new_filename = flow_info.filename
            
            target_path = target_dir / new_filename
            
            # Handle filename collisions by adding a counter
            if target_path.exists():
                base_name = target_path.stem
                extension = ''.join(target_path.suffixes)
                counter = 1
                while target_path.exists():
                    new_name = f"{base_name}_{counter}{extension}"
                    target_path = target_dir / new_name
                    counter += 1
                logger.debug(f"Collision detected, using new name: {target_path.name}")
            
            shutil.copy2(source_path, target_path)
            success_count += 1
            
            if i % 100 == 0:
                logger.debug(f"Saved {i}/{len(flows)} flow files")
                
        except Exception as e:
            logger.error(f"Failed to copy file {flow_info.filename}: {e}")
    
    logger.info(f"Saved {success_count}/{len(flows)} flows to {target_dir}")
    return success_count