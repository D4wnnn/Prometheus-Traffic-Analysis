"""
Light-Only Baseline Evaluation Script (with selection)
Performs inference using only the Light Model and evaluates various performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

sys.path.append('..')
from models.light_model import LightTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed


def evaluate_light_only(args):
    """Main function for Light-Only evaluation"""
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load data
    with open(os.path.join(args.data_dir, "label_mapping.json"), 'r') as f:
        num_classes = len(json.load(f))
    
    test_h5 = os.path.join(args.data_dir, "test_data.h5")
    test_dataset = TrafficDataset(
        test_h5,
        augmentation=False,
        max_packets=args.teacher_packets,  # First load teacher-sized dimensions
        max_bytes=args.teacher_bytes,
        global_selection_path=args.global_selector  # Then apply selection via selector
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    print(f"{'='*60}")
    print(f"Light-Only Baseline Evaluation")
    print(f"{'='*60}")
    print(f"Dataset: {args.data_dir}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load Light Model
    model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)
    checkpoint = torch.load(args.light_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # Strip 'module.' prefix from DDP-saved checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)

    model.eval()
    
    print("✓ Light Model loaded successfully\n")
    
    # Statistical variables
    all_preds = []
    all_labels = []
    all_confidences = []
    batch_latencies = []
    total_samples = 0
    
    print("Starting inference...\n")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Light-Only Inference'):
            batch_start = time.time()
            
            # Prepare data
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            # Inference
            logits = model(batch_data)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(1)
            
            # Record results
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_confidences.append(conf.cpu().numpy())
            
            # Record latency
            batch_end = time.time()
            batch_latencies.append(batch_end - batch_start)
            total_samples += labels.size(0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Aggregate predictions
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_confidences = np.concatenate(all_confidences)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision *= 100
    recall *= 100
    f1 *= 100
    
    throughput = total_samples / total_time
    
    # Confidence statistics
    conf_mean = np.mean(all_confidences)
    conf_std = np.std(all_confidences)
    
    # Latency statistics
    latencies_ms = np.array(batch_latencies) * 1000
    latency_p50 = np.percentile(latencies_ms, 50)
    latency_p95 = np.percentile(latencies_ms, 95)
    latency_p99 = np.percentile(latencies_ms, 99)
    latency_mean = np.mean(latencies_ms)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Light-Only Evaluation Results")
    print(f"{'='*60}")
    print(f"Total Time:         {total_time:.2f}s")
    print(f"Total Samples:      {total_samples}")
    print(f"Throughput:         {throughput:.2f} packets/s")
    print(f"{'-'*60}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"Precision:          {precision:.2f}%")
    print(f"Recall:             {recall:.2f}%")
    print(f"F1 Score:           {f1:.2f}%")
    print(f"{'-'*60}")
    print(f"Confidence:")
    print(f"  Mean:             {conf_mean:.4f}")
    print(f"  Std:              {conf_std:.4f}")
    print(f"{'-'*60}")
    print(f"Latency (per batch):")
    print(f"  Mean:             {latency_mean:.2f} ms")
    print(f"  P50:              {latency_p50:.2f} ms")
    print(f"  P95:              {latency_p95:.2f} ms")
    print(f"  P99:              {latency_p99:.2f} ms")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        'mode': 'Light-Only',
        'config': {
            'model_path': args.light_path,
            'batch_size': args.batch_size,
            'data_dir': args.data_dir,
        },
        'metrics': {
            'total_time': total_time,
            'total_samples': total_samples,
            'throughput': throughput,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'heavy_calls': 0,  # Light-Only does not call Heavy
            'heavy_call_rate': 0.0,
            'confidence_mean': float(conf_mean),
            'confidence_std': float(conf_std),
            'latency_mean': float(latency_mean),
            'latency_p50': float(latency_p50),
            'latency_p95': float(latency_p95),
            'latency_p99': float(latency_p99),
        },
        'latencies': latencies_ms.tolist(),
    }
    
    with open(args.save_result, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {args.save_result}\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--light_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_result', type=str, default='results_light_only.json')
    parser.add_argument('--teacher_packets', type=int, default=10)
    parser.add_argument('--teacher_bytes', type=int, default=300)
    parser.add_argument('--global_selector', type=str, required=True,
                        help='Path to global selector .npz file')
    args = parser.parse_args()
    evaluate_light_only(args)