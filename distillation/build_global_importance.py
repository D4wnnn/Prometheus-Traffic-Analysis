# File: distillation/build_global_importance.py

"""
Global Byte Importance Analysis Tool
Analyzes gradients on a subset of the training set to identify the "average" 
most important packets and byte intervals. 
Outputs a global configuration for consistent use across all samples.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from models import EncryptedTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed


class EmbeddingGradCatcher:
    """Captures the output and gradients of ByteEmbedding."""
    def __init__(self):
        self.emb = None
        
    def fwd_hook(self, module, inp, out):
        self.emb = out
        if out.requires_grad:
            out.retain_grad()
    
    def clear(self):
        self.emb = None


def compute_global_importance(
    teacher, 
    dataloader, 
    device, 
    max_packets,
    max_bytes,
    target_mode="pred",
    analyze_ratio=1.0
):
    """
    Computes the global byte importance heatmap.
    
    Args:
        teacher: Teacher model.
        dataloader: Data loader.
        device: Device (CPU/CUDA).
        max_packets: Maximum number of packets.
        max_bytes: Maximum number of bytes.
        target_mode: "pred" to use predicted class, "true" to use ground truth label.
        analyze_ratio: Ratio of data to analyze (0.0-1.0) for acceleration.
    
    Returns:
        global_importance: (max_packets, max_bytes) Global average importance.
        class_importance: dict {class_id: (max_packets, max_bytes)} Per-class importance.
        class_counts: dict {class_id: count} Sample count per class.
    """
    catcher = EmbeddingGradCatcher()
    hook = teacher.byte_encoder.byte_transformer.byte_embedding.register_forward_hook(
        catcher.fwd_hook
    )
    
    # Accumulators
    global_importance_sum = torch.zeros(max_packets, max_bytes, device=device)
    global_count = 0
    
    class_importance_sum = defaultdict(
        lambda: torch.zeros(max_packets, max_bytes, device=device)
    )
    class_counts = defaultdict(int)
    
    teacher.eval()
    # Enable gradients for embedding weights
    teacher.byte_encoder.byte_transformer.byte_embedding.embedding.weight.requires_grad_(True)
    
    total_batches = len(dataloader)
    analyze_batches = int(total_batches * analyze_ratio)
    
    pbar = tqdm(dataloader, desc="Computing global importance", total=analyze_batches)
    
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= analyze_batches:
            break
            
        batch_data = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)
        
        B = batch_data["bytes_nlp"].size(0)
        N = batch_data["bytes_nlp"].size(1)
        L = batch_data["bytes_nlp"].size(2)
        
        torch.set_grad_enabled(True)
        teacher.zero_grad(set_to_none=True)
        catcher.clear()
        
        logits = teacher(batch_data)
        
        if target_mode == "pred":
            target_idx = logits.argmax(dim=1)
        else:
            target_idx = labels
        
        # Calculate gradients of the logit sum for target classes
        score = logits.gather(1, target_idx.view(-1, 1)).sum()
        score.backward()
        
        emb = catcher.emb  # (B*N, L+1, d) may contain CLS token
        grad = catcher.emb.grad
        
        if emb is None or grad is None:
            continue
        
        # Handle CLS token: Extract valid sequence parts
        actual_L = emb.shape[1]
        if actual_L > L:
            # Assuming CLS is at the front, take the last L positions
            emb = emb[:, 1:L+1, :]
            grad = grad[:, 1:L+1, :]
        elif actual_L < L:
            # If length is insufficient, use actual length
            L = actual_L
        
        # Compute importance: ||grad * emb|| (B*N, L)
        imp = torch.norm(grad * emb, dim=-1).detach()
        imp = imp.view(B, N, -1)  # (B, N, L)
        
        # Ensure dimension matching
        actual_N = min(N, max_packets)
        actual_L = min(imp.shape[2], max_bytes)
        
        # Accumulate to global
        global_importance_sum[:actual_N, :actual_L] += imp[:, :actual_N, :actual_L].sum(dim=0)
        global_count += B
        
        # Accumulate per class
        for b in range(B):
            cls_id = target_idx[b].item()
            class_importance_sum[cls_id][:actual_N, :actual_L] += imp[b, :actual_N, :actual_L]
            class_counts[cls_id] += 1
        
        torch.set_grad_enabled(False)
        
        pbar.set_postfix({"samples": global_count})
    
    hook.remove()
    
    # Normalization
    global_importance = global_importance_sum / max(global_count, 1)
    
    class_importance = {}
    for cls_id in class_importance_sum:
        if class_counts[cls_id] > 0:
            class_importance[cls_id] = (
                class_importance_sum[cls_id] / class_counts[cls_id]
            ).cpu().numpy()
    
    return global_importance.cpu().numpy(), class_importance, dict(class_counts)


def select_top_packets(importance_map, top_k, valid_mask=None):
    """
    Selects the top k most important packets.
    
    Args:
        importance_map: (N, L) Importance map.
        top_k: Number of packets to select.
        valid_mask: (N,) Mask for valid packets; None implies all are valid.
    
    Returns:
        packet_indices: list of int, packet indices in original temporal order.
    """
    N, L = importance_map.shape
    
    # Total importance per packet
    packet_importance = importance_map.sum(axis=1)  # (N,)
    
    if valid_mask is not None:
        packet_importance = packet_importance * valid_mask
    
    # Select top_k
    top_indices = np.argsort(packet_importance)[::-1][:top_k]
    
    # Sort by original index to maintain temporal order
    top_indices = sorted(top_indices.tolist())
    
    return top_indices


def select_byte_windows(importance_1d, window_len, num_windows, strategy="greedy"):
    """
    Selects the most important byte windows.
    
    Args:
        importance_1d: (L,) Byte importance for a single packet.
        window_len: Length of each window.
        num_windows: Number of windows.
        strategy: 
            - "greedy": Greedily select non-overlapping windows.
            - "contiguous": Select a single best contiguous range.
    
    Returns:
        If strategy=="greedy": list of (start, end) windows.
        If strategy=="contiguous": (start, end) single range.
    """
    L = len(importance_1d)
    total_bytes_needed = window_len * num_windows
    
    if strategy == "contiguous":
        # Select contiguous range
        if L <= total_bytes_needed:
            return (0, L)
        
        # Slide window to find the best start position
        window_sums = np.convolve(importance_1d, np.ones(total_bytes_needed), mode='valid')
        best_start = int(np.argmax(window_sums))
        return (best_start, best_start + total_bytes_needed)
    
    elif strategy == "greedy":
        # Greedily select non-overlapping windows
        if L < window_len:
            return [(0, L)]
        
        # Calculate window scores for each start position
        window_scores = np.convolve(importance_1d, np.ones(window_len), mode='valid')
        
        selected = []
        used = np.zeros(L, dtype=bool)
        
        while len(selected) < num_windows:
            # Find current best non-overlapping window
            best_score = -1
            best_start = 0
            
            for start in range(len(window_scores)):
                end = start + window_len
                if not np.any(used[start:end]):
                    if window_scores[start] > best_score:
                        best_score = window_scores[start]
                        best_start = start
            
            if best_score < 0:
                # No more non-overlapping windows available
                break
            
            selected.append((best_start, best_start + window_len))
            used[best_start:best_start + window_len] = True
        
        # Sort by start position to maintain order
        selected.sort(key=lambda x: x[0])
        
        # Pad with (0, window_len) if insufficient windows
        while len(selected) < num_windows:
            selected.append((0, window_len))
        
        return selected
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_global_selection_config(
    global_importance,
    top_packets,
    window_len,
    windows_per_packet,
    byte_strategy="contiguous"
):
    """
    Constructs the global selection configuration.
    
    Args:
        global_importance: (N, L) Global importance map.
        top_packets: Number of selected packets.
        window_len: Length of each window.
        windows_per_packet: Number of windows per packet.
        byte_strategy: "contiguous" or "greedy".
    
    Returns:
        config: dict Global selection configuration.
    """
    N, L = global_importance.shape
    
    # 1. Select the most important packets
    packet_indices = select_top_packets(global_importance, top_packets)
    
    # 2. Byte selection
    target_bytes = window_len * windows_per_packet
    
    if byte_strategy == "contiguous":
        # Option A: All packets use the same byte range
        # Calculate average importance of selected packets
        avg_byte_importance = global_importance[packet_indices].mean(axis=0)
        byte_start, byte_end = select_byte_windows(
            avg_byte_importance, target_bytes, 1, strategy="contiguous"
        )
        
        config = {
            "packet_indices": packet_indices,
            "byte_range": {"start": int(byte_start), "end": int(byte_end)},
            "strategy": "contiguous",
            "top_packets": top_packets,
            "target_bytes": target_bytes,
            "window_len": window_len,
            "windows_per_packet": windows_per_packet
        }
    
    elif byte_strategy == "greedy":
        # Option B: Different windows per packet (simplified using average importance)
        avg_byte_importance = global_importance[packet_indices].mean(axis=0)
        windows = select_byte_windows(
            avg_byte_importance, window_len, windows_per_packet, strategy="greedy"
        )
        
        config = {
            "packet_indices": packet_indices,
            "byte_windows": [{"start": int(s), "end": int(e)} for s, e in windows],
            "strategy": "greedy",
            "top_packets": top_packets,
            "window_len": window_len,
            "windows_per_packet": windows_per_packet
        }
    
    elif byte_strategy == "per_packet":
        # Option C: Each packet independently selects best windows
        per_packet_config = []
        for pkt_idx in packet_indices:
            pkt_importance = global_importance[pkt_idx]
            windows = select_byte_windows(
                pkt_importance, window_len, windows_per_packet, strategy="greedy"
            )
            per_packet_config.append({
                "packet_idx": int(pkt_idx),
                "windows": [{"start": int(s), "end": int(e)} for s, e in windows]
            })
        
        config = {
            "packet_configs": per_packet_config,
            "strategy": "per_packet",
            "top_packets": top_packets,
            "window_len": window_len,
            "windows_per_packet": windows_per_packet
        }
    
    else:
        raise ValueError(f"Unknown byte_strategy: {byte_strategy}")
    
    return config


def visualize_importance(
    global_importance, 
    class_importance, 
    config, 
    save_path
):
    """
    Visualizes the importance heatmap and selected regions.
    """
    n_classes = len(class_importance)
    fig_rows = 2 + (n_classes + 2) // 3  # Global + Selection + Per-class
    
    fig, axes = plt.subplots(fig_rows, 3, figsize=(15, 4 * fig_rows))
    axes = axes.flatten()
    
    # 1. Global Importance
    ax = axes[0]
    im = ax.imshow(global_importance, aspect='auto', cmap='hot')
    ax.set_title('Global Importance (All Classes)')
    ax.set_xlabel('Byte Position')
    ax.set_ylabel('Packet Index')
    plt.colorbar(im, ax=ax)
    
    # 2. Visualization of Selected Regions
    ax = axes[1]
    selection_mask = np.zeros_like(global_importance)
    
    if config["strategy"] == "contiguous":
        pkt_indices = config["packet_indices"]
        byte_start = config["byte_range"]["start"]
        byte_end = config["byte_range"]["end"]
        for pkt_idx in pkt_indices:
            selection_mask[pkt_idx, byte_start:byte_end] = 1
    elif config["strategy"] == "greedy":
        pkt_indices = config["packet_indices"]
        for pkt_idx in pkt_indices:
            for win in config["byte_windows"]:
                selection_mask[pkt_idx, win["start"]:win["end"]] = 1
    elif config["strategy"] == "per_packet":
        for pkt_cfg in config["packet_configs"]:
            pkt_idx = pkt_cfg["packet_idx"]
            for win in pkt_cfg["windows"]:
                selection_mask[pkt_idx, win["start"]:win["end"]] = 1
    
    ax.imshow(selection_mask, aspect='auto', cmap='Blues')
    ax.set_title('Selected Regions')
    ax.set_xlabel('Byte Position')
    ax.set_ylabel('Packet Index')
    
    # 3. Overlay
    ax = axes[2]
    ax.imshow(global_importance, aspect='auto', cmap='hot', alpha=0.7)
    ax.imshow(selection_mask, aspect='auto', cmap='Blues', alpha=0.3)
    ax.set_title('Importance + Selection Overlay')
    ax.set_xlabel('Byte Position')
    ax.set_ylabel('Packet Index')
    
    # 4. Importance per category
    for i, (cls_id, cls_imp) in enumerate(sorted(class_importance.items())):
        ax_idx = 3 + i
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        im = ax.imshow(cls_imp, aspect='auto', cmap='hot')
        ax.set_title(f'Class {cls_id}')
        ax.set_xlabel('Byte Position')
        ax.set_ylabel('Packet Index')
        plt.colorbar(im, ax=ax)
    
    # Hide redundant subplots
    for i in range(3 + len(class_importance), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Build global importance-based byte selection")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="H5 data directory (e.g., ../1-data_preprocess/aes_128_gcm_h5)")
    parser.add_argument("--h5_name", type=str, default="train_data.h5",
                        help="H5 filename")
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="Path to the teacher model")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output path for the configuration file (.npz)")
    
    # Teacher model geometry
    parser.add_argument("--max_packets_teacher", type=int, default=10,
                        help="Number of packets seen by the Teacher")
    parser.add_argument("--max_bytes_teacher", type=int, default=300,
                        help="Bytes per packet seen by the Teacher")
    
    # Student target geometry
    parser.add_argument("--top_packets", type=int, default=5,
                        help="Number of packets selected for the Student")
    parser.add_argument("--window_len", type=int, default=50,
                        help="Byte length for each window")
    parser.add_argument("--windows_per_packet", type=int, default=3,
                        help="Windows per packet (Total bytes = window_len * windows_per_packet)")
    
    # Strategy selection
    parser.add_argument("--byte_strategy", type=str, default="contiguous",
                        choices=["contiguous", "greedy", "per_packet"],
                        help="Byte selection strategy")
    parser.add_argument("--target_mode", type=str, default="pred",
                        choices=["pred", "true"],
                        help="Gradient target: pred=predicted class, true=ground truth label")
    
    # Efficiency control
    parser.add_argument("--analyze_ratio", type=float, default=0.2,
                        help="Ratio of data to analyze (0.0-1.0) for acceleration")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to generate visualization plots")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Global Importance Analysis")
    print("=" * 60)
    print(f"Data: {args.data_dir}/{args.h5_name}")
    print(f"Teacher: {args.teacher_path}")
    print(f"Target geometry: {args.top_packets} packets × {args.window_len * args.windows_per_packet} bytes")
    print(f"Strategy: {args.byte_strategy}")
    print(f"Analyze ratio: {args.analyze_ratio * 100:.0f}%")
    print("=" * 60)
    
    # Load label mapping
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    with open(label_map_path, "r") as f:
        label_mapping = json.load(f)
        num_classes = len(label_mapping)
    print(f"Number of classes: {num_classes}")
    
    # Load dataset
    h5_path = os.path.join(args.data_dir, args.h5_name)
    print(f"Loading dataset from {h5_path}")
    
    dataset = TrafficDataset(
        h5_path,
        augmentation=False,
        max_packets=args.max_packets_teacher,
        max_bytes=args.max_bytes_teacher
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep unshuffled for reproducibility
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches to analyze: {int(len(dataloader) * args.analyze_ratio)}")
    
    # Load teacher model
    print(f"\nLoading teacher model...")
    teacher = EncryptedTrafficClassifier(
        num_classes=num_classes,
        d_byte=128,
        d_packet=260,
        byte_layers=3,
        packet_layers=4,
        num_heads=5,
        d_ff=1024,
        max_bytes=args.max_bytes_teacher,
        max_packets=100,
        dropout=0.1,
        use_stats=False,
        stats_dim=36,
        use_adaptive_gating=True
    ).to(device)
    
    ckpt = torch.load(args.teacher_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    teacher.load_state_dict(state_dict, strict=True)
    print("Teacher model loaded successfully")
    
    # Compute global importance
    print(f"\nComputing global importance...")
    global_importance, class_importance, class_counts = compute_global_importance(
        teacher=teacher,
        dataloader=dataloader,
        device=device,
        max_packets=args.max_packets_teacher,
        max_bytes=args.max_bytes_teacher,
        target_mode=args.target_mode,
        analyze_ratio=args.analyze_ratio
    )
    
    print(f"\nClass distribution in analyzed samples:")
    for cls_id, count in sorted(class_counts.items()):
        cls_name = list(label_mapping.keys())[cls_id] if cls_id < len(label_mapping) else f"Unknown-{cls_id}"
        print(f"  Class {cls_id} ({cls_name}): {count} samples")
    
    # Build selection configuration
    print(f"\nBuilding selection config with strategy: {args.byte_strategy}")
    config = build_global_selection_config(
        global_importance=global_importance,
        top_packets=args.top_packets,
        window_len=args.window_len,
        windows_per_packet=args.windows_per_packet,
        byte_strategy=args.byte_strategy
    )
    
    # Print configuration details
    print(f"\n{'=' * 60}")
    print("Selection Configuration:")
    print(f"{'=' * 60}")
    
    if config["strategy"] == "contiguous":
        print(f"  Selected packets: {config['packet_indices']}")
        print(f"  Byte range: [{config['byte_range']['start']}, {config['byte_range']['end']})")
        print(f"  Total bytes per packet: {config['byte_range']['end'] - config['byte_range']['start']}")
    elif config["strategy"] == "greedy":
        print(f"  Selected packets: {config['packet_indices']}")
        print(f"  Byte windows:")
        for i, win in enumerate(config["byte_windows"]):
            print(f"    Window {i}: [{win['start']}, {win['end']})")
    elif config["strategy"] == "per_packet":
        for pkt_cfg in config["packet_configs"]:
            print(f"  Packet {pkt_cfg['packet_idx']}:")
            for i, win in enumerate(pkt_cfg["windows"]):
                print(f"    Window {i}: [{win['start']}, {win['end']})")
    
    # Save configuration
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    # Convert class_importance to storable format
    save_dict = {
        "config": json.dumps(config),
        "global_importance": global_importance,
        "class_counts": json.dumps(class_counts),
        "args": json.dumps(vars(args))
    }
    for cls_id, cls_imp in class_importance.items():
        save_dict[f"class_importance_{cls_id}"] = cls_imp
    
    np.savez_compressed(args.out_path, **save_dict)
    print(f"\nConfig saved to: {args.out_path}")
    
    # Visualization
    if args.visualize:
        vis_path = args.out_path.replace(".npz", "_visualization.png")
        visualize_importance(global_importance, class_importance, config, vis_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()