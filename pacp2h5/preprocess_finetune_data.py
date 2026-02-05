import h5py
import numpy as np
import os
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from scapy.all import rdpcap, IP, TCP, UDP
import re

from feature_extractors import extract_bytes_nlp_view, extract_sequence_nlp_view, extract_stats_view
import debugpy

# --- Configuration Parameters ---
# Byte Modality (NLP view)
N_PACKETS_BYTES_NLP = 10
N_BYTES_PER_PACKET_NLP = 300

# Sequence Modality (NLP view)
N_PACKETS_SEQ = 10

# Statistical Modality
N_STATS = 36


# --- HDF5 Dataset Names ---
DS_BYTES_NLP = "bytes_nlp_view"
DS_SEQ_NLP = "seq_nlp_view"
DS_TCP_DATA = "tcp_data_view" 
DS_STATS = "stats_view"
DS_LABELS = "labels"
DS_NUM_NODES = "num_nodes"


# ============================================================================
# CSTNET Dataset Processing Functions
# ============================================================================
def parse_ipv4_frames(hex_string: str, mask_ip_port: bool = True):
    """
    Extract IPv4 packets from a continuous hex string (stripping Ethernet headers).
    
    Args:
        hex_string: Hexadecimal string of the traffic.
        mask_ip_port: If True, masks IP addresses and port numbers.
    
    Returns: 
        List of dictionaries: [{'ip_header': bytes, 'payload': bytes}, ...]
    """
    hex_string = hex_string.replace(" ", "").replace("\n", "").lower()
    try:
        data = bytes.fromhex(hex_string)
    except ValueError:
        return []
    
    frames = []
    pattern = re.compile(b"\x08\x00\x00\x45|\x08\x00\x45")
    matches = [m.start() for m in re.finditer(pattern, data)]
    
    for i, start in enumerate(matches):
        end = matches[i + 1] if i + 1 < len(matches) else len(data)
        frame_data = data[start:end]
        idx = frame_data.find(b"\x45")
        if idx == -1:
            continue
        
        ipv4_bytes = frame_data[idx:]
        if len(ipv4_bytes) < 20:
            continue
        
        ihl = ipv4_bytes[0] & 0x0F
        header_len = ihl * 4
        total_len = int.from_bytes(ipv4_bytes[2:4], "big", signed=False)
        
        if total_len == 0 or total_len > len(ipv4_bytes):
            total_len = len(ipv4_bytes)
        
        ip_header = bytearray(ipv4_bytes[:header_len])
        payload = bytearray(ipv4_bytes[header_len:total_len])
        
        if mask_ip_port:
            # Mask Source IP (Bytes 12-15)
            ip_header[12:16] = b'\x00\x00\x00\x00'
            # Mask Destination IP (Bytes 16-19)
            ip_header[16:20] = b'\x00\x00\x00\x00'
            
            # Mask Port Numbers (TCP=6, UDP=17)
            protocol = ip_header[9]
            if protocol in [6, 17] and len(payload) >= 4:
                # Mask Source Port (Bytes 0-1)
                payload[0:2] = b'\x00\x00'
                # Mask Destination Port (Bytes 2-3)
                payload[2:4] = b'\x00\x00'

        
        frames.append({"ip_header": bytes(ip_header), "payload": bytes(payload)})
    
    return frames


def extract_cstnet_bytes_nlp_view(datagram_str, n_packets=100, max_bytes_per_packet=500):
    frames = parse_ipv4_frames(datagram_str)
    all_packet_bytes = []
    for frame in frames[:n_packets]:
        pkt_bytes = frame["ip_header"] + frame["payload"]
        if len(pkt_bytes) >= max_bytes_per_packet:
            pkt_arr = np.frombuffer(pkt_bytes[:max_bytes_per_packet], dtype=np.uint8)
        else:
            padding = max_bytes_per_packet - len(pkt_bytes)
            pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
            pkt_arr = np.pad(pkt_arr, (0, padding), "constant", constant_values=0)
        all_packet_bytes.append(pkt_arr)
    while len(all_packet_bytes) < n_packets:
        all_packet_bytes.append(np.zeros(max_bytes_per_packet, dtype=np.uint8))
    return np.stack(all_packet_bytes)


def extract_cstnet_sequence_nlp_view(lengths, directions, times, n_packets=100):
    valid_mask = lengths != 0
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.zeros((n_packets, 3), dtype=np.float32), None
    valid_count = min(len(valid_indices), n_packets)
    valid_indices = valid_indices[:valid_count]
    valid_lengths = lengths[valid_indices]
    valid_directions = directions[valid_indices]
    valid_times = times[valid_indices]
    iats = np.zeros(valid_count, dtype=np.float32)
    if valid_count > 1:
        iats[1:] = np.diff(valid_times)
    seq_data = np.column_stack([np.abs(valid_lengths), valid_directions, iats]).astype(np.float32)
    if len(seq_data) < n_packets:
        padding = np.zeros((n_packets - len(seq_data), 3), dtype=np.float32)
        seq_data = np.vstack([seq_data, padding])
    return seq_data, None


def extract_cstnet_stats_view(lengths, directions, times, n_stats=36):
    valid_mask = lengths != 0
    if not np.any(valid_mask):
        return np.zeros(n_stats, dtype=np.float32)
    try:
        valid_lengths = np.abs(lengths[valid_mask]).astype(np.float32)[:500]
        valid_directions = directions[valid_mask].astype(np.int8)[:500]
        valid_times = times[valid_mask][:500]
        iats = np.diff(valid_times) if len(valid_times) > 1 else np.array([0.0])
        from stats_utils import _compute_statistics, _compute_direction_statistics, _compute_flow_statistics, _compute_bidirectional_features, _compute_burst_features

        features = []
        features.extend(_compute_statistics(valid_lengths))
        features.extend(_compute_statistics(iats))
        features.extend(_compute_direction_statistics(valid_lengths, valid_directions))
        features.extend(_compute_flow_statistics(valid_lengths, valid_times))
        features.extend(_compute_bidirectional_features(valid_lengths, valid_directions, valid_times))
        features.extend(_compute_burst_features(valid_lengths, valid_times, valid_directions))
        features = np.array(features, dtype=np.float32)
        if len(features) < n_stats:
            features = np.pad(features, (0, n_stats - len(features)), "constant")
        else:
            features = features[:n_stats]
        return features
    except Exception:
        return np.zeros(n_stats, dtype=np.float32)


def process_cstnet_datagram(args):
    """
    Process a single CSTNET sample and extract all modality features.
    """
    datagram_str, directions, lengths, times, label_idx = args
    try:
        bytes_nlp = extract_cstnet_bytes_nlp_view(datagram_str, n_packets=N_PACKETS_BYTES_NLP, max_bytes_per_packet=N_BYTES_PER_PACKET_NLP)
        seq_nlp, _ = extract_cstnet_sequence_nlp_view(lengths, directions, times, n_packets=N_PACKETS_SEQ)
        stats = extract_cstnet_stats_view(lengths, directions, times, n_stats=N_STATS)
        tcp_data = np.zeros((N_PACKETS_SEQ, 3), dtype=np.float32)

        valid_mask = seq_nlp[:, 0] > 0
        num_nodes = int(np.sum(valid_mask))

        return (bytes_nlp, seq_nlp, tcp_data, stats, num_nodes, label_idx)

    except Exception as e:
        print(f"Error processing CSTNET datagram: {e}")
        return None


# ============================================================================
# PCAP File Processing Functions
# ============================================================================
def process_pcap_with_label(args):
    """
    Process a single PCAP file, extract all modalities/views, and labels.
    """
    pcap_path, label_idx = args
    try:
        packets = rdpcap(str(pcap_path), count=500)
        ip_packets = [p for p in packets]
        if not ip_packets:
            return None

        # Extract features for each modality
        bytes_nlp = extract_bytes_nlp_view(ip_packets, n_packets=N_PACKETS_BYTES_NLP, max_bytes_per_packet=N_BYTES_PER_PACKET_NLP)
        seq_nlp, tcp_data = extract_sequence_nlp_view(ip_packets, n_packets=N_PACKETS_SEQ)
        stats = extract_stats_view(ip_packets, n_stats=N_STATS)
        valid_mask = seq_nlp[:, 0] > 0
        num_nodes = int(np.sum(valid_mask))

        return (bytes_nlp, seq_nlp, tcp_data, stats, num_nodes, label_idx)

    except Exception as e:
        print(f"Error processing {pcap_path.name}: {e}")
        return None


# ============================================================================
# Label Mapping and File Collection
# ============================================================================
def build_global_label_mapping(root_dir):
    all_labels = set()
    for split in ["train", "test"]:
        split_dir = root_dir / split
        if split_dir.exists():
            label_dirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
            all_labels.update(label_dirs)
    sorted_labels = sorted(all_labels)
    label_to_idx = {name: idx for idx, name in enumerate(sorted_labels)}
    return label_to_idx


def collect_pcap_files(root_dir, split, label_to_idx):
    split_dir = root_dir / split
    if not split_dir.exists():
        raise ValueError(f"Split directory {split_dir} does not exist!")
    label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        raise ValueError(f"No label directories found in {split_dir}")
    pcap_files = []
    labels = []
    for label_dir in label_dirs:
        label_name = label_dir.name
        if label_name not in label_to_idx:
            print(f"Warning: Label '{label_name}' not in global mapping, skipping")
            continue
        label_idx = label_to_idx[label_name]
        pcap_list = list(label_dir.rglob("*.pcap"))
        print(f"  - {label_name} ({label_idx}): {len(pcap_list)} files")
        pcap_files.extend(pcap_list)
        labels.extend([label_idx] * len(pcap_list))
    return pcap_files, labels


# ============================================================================
# HDF5 Writing
# ============================================================================
def write_batch_to_h5(h5f, results, start_index):
    """
    Write a batch of data to HDF5.
    """
    batch_len = len(results)
    end_index = start_index + batch_len

    # Resize all datasets at once
    h5f[DS_BYTES_NLP].resize(end_index, axis=0)
    h5f[DS_SEQ_NLP].resize(end_index, axis=0)
    h5f[DS_TCP_DATA].resize(end_index, axis=0)
    h5f[DS_STATS].resize(end_index, axis=0)
    h5f[DS_NUM_NODES].resize(end_index, axis=0)
    h5f[DS_LABELS].resize(end_index, axis=0)

    try:
        # Stack batch data
        # results tuple: (bytes_nlp, seq_nlp, tcp_data, stats, num_nodes, label_idx)
        bytes_nlp_batch = np.stack([r[0] for r in results])
        seq_nlp_batch = np.stack([r[1] for r in results])
        tcp_data_batch = np.stack([r[2] for r in results])
        stats_batch = np.stack([r[3] for r in results])
        num_nodes_batch = np.array([r[4] for r in results], dtype=np.int32)
        labels_batch = np.array([r[5] for r in results], dtype=np.int32)

        # Bulk write
        h5f[DS_BYTES_NLP][start_index:end_index] = bytes_nlp_batch
        h5f[DS_SEQ_NLP][start_index:end_index] = seq_nlp_batch
        h5f[DS_TCP_DATA][start_index:end_index] = tcp_data_batch
        h5f[DS_STATS][start_index:end_index] = stats_batch
        h5f[DS_NUM_NODES][start_index:end_index] = num_nodes_batch
        h5f[DS_LABELS][start_index:end_index] = labels_batch

        return batch_len

    except ValueError as e:
        print(f"Warning: Error writing batch, falling back to sequential write: {e}")
        # Fallback mechanism: write entry by entry
        count = 0
        for i, result in enumerate(results):
            idx = start_index + i
            try:
                h5f[DS_BYTES_NLP][idx] = result[0]
                h5f[DS_SEQ_NLP][idx] = result[1]
                h5f[DS_TCP_DATA][idx] = result[2]
                h5f[DS_STATS][idx] = result[3]
                h5f[DS_NUM_NODES][idx] = result[4]
                h5f[DS_LABELS][idx] = result[5]
                count += 1
            except Exception as write_e:
                print(f"Error: Unable to write index {idx}: {write_e}")
        return count


# ============================================================================
# Main Processing Logic
# ============================================================================
def create_h5_datasets(h5f):
    """Create all datasets within the HDF5 file."""
    h5f.create_dataset(DS_BYTES_NLP, shape=(0, N_PACKETS_BYTES_NLP, N_BYTES_PER_PACKET_NLP), maxshape=(None, N_PACKETS_BYTES_NLP, N_BYTES_PER_PACKET_NLP), dtype=np.uint8, chunks=True, compression="gzip")
    h5f.create_dataset(DS_SEQ_NLP, shape=(0, N_PACKETS_SEQ, 3), maxshape=(None, N_PACKETS_SEQ, 3), dtype=np.float32, chunks=True, compression="gzip")
    h5f.create_dataset(DS_TCP_DATA, shape=(0, N_PACKETS_SEQ, 3), maxshape=(None, N_PACKETS_SEQ, 3), dtype=np.float32, chunks=True, compression="gzip")
    h5f.create_dataset(DS_STATS, shape=(0, N_STATS), maxshape=(None, N_STATS), dtype=np.float32, chunks=True, compression="gzip")
    h5f.create_dataset(DS_NUM_NODES, shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
    h5f.create_dataset(DS_LABELS, shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True, compression="gzip")


def process_cstnet_split(dataset_path, split, output_dir, num_workers=8, batch_size=5000):
    """Process a single split of the CSTNET dataset."""
    print(f"\n{'='*60}\nStarting Processing CSTNET {split.upper()} Dataset\n{'='*60}")
    datagram_file = dataset_path / f"x_datagram_{split}.npy"
    direction_file = dataset_path / f"x_direction_{split}.npy"
    len_file = dataset_path / f"x_len_{split}.npy"
    time_file = dataset_path / f"x_time_{split}.npy"
    label_file = dataset_path / f"y_{split}.npy"
    required_files = [datagram_file, direction_file, len_file, time_file, label_file]
    if any(not f.exists() for f in required_files):
        print(f"Skipping {split}: Missing files")
        return None

    print("Loading data files...")
    x_datagram = np.load(datagram_file, allow_pickle=True)
    x_direction = np.load(direction_file)
    x_len = np.load(len_file)
    x_time = np.load(time_file)
    y_labels = np.load(label_file)
    print(f"Loaded {len(x_datagram)} samples")

    unique_labels = np.unique(y_labels)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    process_args = [(d, dr, l, t, label_to_idx[y]) for d, dr, l, t, y in zip(x_datagram, x_direction, x_len, x_time, y_labels)]
    if split == 'valid':
        split = 'val'
    output_h5_file = output_dir / f"{split}_data.h5"
    if output_h5_file.exists():
        os.remove(output_h5_file)

    with h5py.File(output_h5_file, "w") as h5f:
        create_h5_datasets(h5f)  
        current_size = 0
        batch_results = []
        print(f"Starting processing with {num_workers} worker processes...")
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(process_args), desc=f"Processing CSTNET {split}") as pbar:
                for result in pool.imap_unordered(process_cstnet_datagram, process_args):
                    pbar.update(1)
                    if result:
                        batch_results.append(result)
                    if len(batch_results) >= batch_size:
                        written_count = write_batch_to_h5(h5f, batch_results, current_size)
                        current_size += written_count
                        batch_results = []
        if batch_results:
            written_count = write_batch_to_h5(h5f, batch_results, current_size)
            current_size += written_count

    print(f"\nCSTNET {split.upper()} processing complete!")
    print(f"Successfully processed and saved {current_size} / {len(x_datagram)} flows")
    print(f"Data saved to: {output_h5_file}")
    return label_to_idx


def process_split(root_dir, split, output_dir, label_to_idx, num_workers=8, batch_size=5000):
    """Process a single split (train or test) of the PCAP dataset."""
    print(f"\n{'='*60}\nStarting Processing {split.upper()} Dataset\n{'='*60}")
    print(f"Collecting files for {split} split...")
    pcap_files, labels = collect_pcap_files(root_dir, split, label_to_idx)
    print(f"\nTotal files found: {len(pcap_files)}")
    if not pcap_files:
        return

    output_h5_file = output_dir / f"{split}_data.h5"
    if output_h5_file.exists():
        os.remove(output_h5_file)

    with h5py.File(output_h5_file, "w") as h5f:
        create_h5_datasets(h5f)  
        process_args = list(zip(pcap_files, labels))
        current_size = 0
        batch_results = []
        print(f"Starting processing with {num_workers} worker processes...")
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(process_args), desc=f"Processing {split}") as pbar:
                for result in pool.imap_unordered(process_pcap_with_label, process_args):
                    pbar.update(1)
                    if result:
                        batch_results.append(result)
                    if len(batch_results) >= batch_size:
                        written_count = write_batch_to_h5(h5f, batch_results, current_size)
                        current_size += written_count
                        batch_results = []
        if batch_results:
            written_count = write_batch_to_h5(h5f, batch_results, current_size)
            current_size += written_count

    print(f"\n{split.upper()} processing complete!")
    print(f"Successfully processed and saved {current_size} / {len(pcap_files)} flows")
    print(f"Data saved to: {output_h5_file}")


def detect_dataset_type(dataset_root):
    cstnet_files = ["x_datagram_train.npy", "y_train.npy"]
    if all((dataset_root / f).exists() for f in cstnet_files):
        return "cstnet"
    if (dataset_root / "train").exists() or (dataset_root / "test").exists():
        return "pcap"
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess fine-tuning datasets")
    parser.add_argument("--input", "-i", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--batch-size", "-b", type=int, default=5000, help="Batch size for writing")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.input)
    output_dir = Path(args.output)
    num_workers = args.workers
    batch_size = args.batch_size
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root directory does not exist: {dataset_root}")
    
    dataset_type = detect_dataset_type(dataset_root)
    if dataset_type is None:
        raise ValueError("Unrecognized dataset type!")
    print(f"Detected dataset type: {dataset_type.upper()}")
    
    if dataset_type == "cstnet":
        train_label_map = process_cstnet_split(dataset_root, "train", output_dir, num_workers, batch_size) if (dataset_root / "x_datagram_train.npy").exists() else None
        val_label_map = process_cstnet_split(dataset_root, "valid", output_dir, num_workers, batch_size) if (dataset_root / "x_datagram_valid.npy").exists() else None
        test_label_map = process_cstnet_split(dataset_root, "test", output_dir, num_workers, batch_size) if (dataset_root / "x_datagram_test.npy").exists() else None
        label_to_idx = {**train_label_map, **test_label_map, **val_label_map} if train_label_map and test_label_map and val_label_map else (train_label_map or test_label_map or val_label_map or {})
    else:
        print("Building global label mapping...")
        label_to_idx = build_global_label_mapping(dataset_root)
        print(f"\nNumber of label classes: {len(label_to_idx)}")
        for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
            print(f"  {idx}: {label}")
        if (dataset_root / "train").exists():
            process_split(dataset_root, "train", output_dir, label_to_idx, num_workers, batch_size)
        if (dataset_root / "test").exists():
            process_split(dataset_root, "test", output_dir, label_to_idx, num_workers, batch_size)
        if (dataset_root / "val").exists():
            process_split(dataset_root, "val", output_dir, label_to_idx, num_workers, batch_size)
    
    label_map_file = output_dir / "label_mapping.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_to_idx.items()}, f, indent=2, ensure_ascii=False)
    print(f"\nLabel mapping saved to: {label_map_file}")
    
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    idx_map_file = output_dir / "idx_to_label.json"
    with open(idx_map_file, "w", encoding="utf-8") as f:
        json.dump({str(k): str(v) for k, v in idx_to_label.items()}, f, indent=2, ensure_ascii=False)
    print(f"Index mapping saved to: {idx_map_file}")
    print("\n" + "=" * 60 + "\nAll datasets processed successfully!\n" + "=" * 60)

if __name__ == "__main__":
    main()