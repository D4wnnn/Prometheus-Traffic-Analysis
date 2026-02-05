import h5py
import numpy as np
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scapy.all import rdpcap, IP, TCP, UDP

from feature_extractors import (
    extract_bytes_nlp_view, 
    extract_sequence_nlp_view, 
    extract_stats_view,
)

# --- Configuration Parameters ---
# Byte Modality (NLP view)
N_PACKETS_BYTES_NLP = 10
N_BYTES_PER_PACKET_NLP = 500

# Sequence Modality (NLP view)
N_PACKETS_SEQ = 10

# Statistical Modality
N_STATS = 36

# --- HDF5 Dataset Names ---
DS_BYTES_NLP = "bytes_nlp_view"
DS_SEQ_NLP = "seq_nlp_view"
DS_TCP_DATA = "tcp_data_view"
DS_STATS = "stats_view"
DS_NUM_NODES = "num_nodes" # Count of valid packets

def process_pcap(pcap_path):
    """
    Process a single PCAP file and extract raw features only.
    """
    try:
        packets = rdpcap(str(pcap_path), count=N_PACKETS_BYTES_NLP)
        ip_packets = [p for p in packets if IP in p and (TCP in p or UDP in p)]
        if not ip_packets:
            return None

        # ========== 1. Raw Feature Extraction ==========
        bytes_nlp = extract_bytes_nlp_view(
            ip_packets, 
            n_packets=N_PACKETS_BYTES_NLP, 
            max_bytes_per_packet=N_BYTES_PER_PACKET_NLP,
            mask_ip_port=False
        )
        
        seq_nlp, tcp_data = extract_sequence_nlp_view(ip_packets, n_packets=N_PACKETS_SEQ)
        stats = extract_stats_view(ip_packets, n_stats=N_STATS)
        
        # Calculate number of valid nodes
        valid_mask = seq_nlp[:, 0] > 0
        num_nodes = int(np.sum(valid_mask))
        
        return (bytes_nlp, seq_nlp, tcp_data, stats, num_nodes)

    except Exception as e:
        print(f"Error processing {pcap_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # --- Configuration ---
    dataset_dirs = [
        Path("/path/to/CIC_IOT_Dataset2022-processed"),
        Path("/path/to/MAWI-processed"),
    ]
    output_h5_file = Path("./t48-pretrain_data_all-big.h5")

    if output_h5_file.exists():
        os.remove(output_h5_file)
        print(f"Deleted old file: {output_h5_file}")

    print("Locating all .pcap files...")
    pcap_files = []
    for d in dataset_dirs:
        if not d.exists():
            print(f"Warning: Directory {d} does not exist, skipping.")
            continue
        pcap_files.extend(list(d.rglob("*.pcap")))

    print(f"Total .pcap files found: {len(pcap_files)}")
    if not pcap_files:
        print("No files found. Exiting.")
        return

    # ========== Initialize HDF5 File ==========
    print("\nCreating HDF5 file...")
    with h5py.File(output_h5_file, "w") as h5f:
        h5f.create_dataset(DS_BYTES_NLP, shape=(0, N_PACKETS_BYTES_NLP, N_BYTES_PER_PACKET_NLP), 
                           maxshape=(None, N_PACKETS_BYTES_NLP, N_BYTES_PER_PACKET_NLP), 
                           dtype=np.uint8, chunks=True, compression="gzip")
        h5f.create_dataset(DS_SEQ_NLP, shape=(0, N_PACKETS_SEQ, 3), 
                           maxshape=(None, N_PACKETS_SEQ, 3), 
                           dtype=np.float32, chunks=True, compression="gzip")
        h5f.create_dataset(DS_TCP_DATA, shape=(0, N_PACKETS_SEQ, 3), 
                           maxshape=(None, N_PACKETS_SEQ, 3), 
                           dtype=np.float32, chunks=True, compression="gzip")
        h5f.create_dataset(DS_STATS, shape=(0, N_STATS), 
                           maxshape=(None, N_STATS), 
                           dtype=np.float32, chunks=True, compression="gzip")
        h5f.create_dataset(DS_NUM_NODES, shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
        
        current_size = 0
        num_workers = min(cpu_count() - 2, 64)
        print(f"Starting processing with {num_workers} worker processes...")
        
        BATCH_SIZE = 200
        batch_results = []
        def write_batch_to_h5(h5f, results, start_index):
            """Writes batch data to HDF5"""
            batch_len = len(results)
            end_index = start_index + batch_len

            # Resize all datasets
            for ds_name in [DS_BYTES_NLP, DS_SEQ_NLP, DS_TCP_DATA, DS_STATS, DS_NUM_NODES]:
                h5f[ds_name].resize(end_index, axis=0)

            try:
                original_data = results
                # Write raw data
                h5f[DS_BYTES_NLP][start_index:end_index] = np.stack([d[0] for d in original_data])
                h5f[DS_SEQ_NLP][start_index:end_index] = np.stack([d[1] for d in original_data])
                h5f[DS_TCP_DATA][start_index:end_index] = np.stack([d[2] for d in original_data])
                h5f[DS_STATS][start_index:end_index] = np.stack([d[3] for d in original_data])
                h5f[DS_NUM_NODES][start_index:end_index] = np.array([d[4] for d in original_data])
                
                return batch_len
                
            except Exception as e:
                print(f"Batch write failed: {e}")
                import traceback
                traceback.print_exc()
                return 0

        # ========== Multi-process Processing ==========
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(pcap_files), desc="Processing") as pbar:
                for result in pool.imap_unordered(process_pcap, pcap_files):
                    pbar.update(1)
                    if result:
                        batch_results.append(result)

                    if len(batch_results) >= BATCH_SIZE:
                        written_count = write_batch_to_h5(h5f, batch_results, current_size)
                        current_size += written_count
                        batch_results = []
        
        if batch_results:
            written_count = write_batch_to_h5(h5f, batch_results, current_size)
            current_size += written_count

    # ========== Print Statistics ==========
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {current_size} flows")
    print(f"Data saved to: {output_h5_file}")
    print(f"\nDataset size: {output_h5_file.stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()