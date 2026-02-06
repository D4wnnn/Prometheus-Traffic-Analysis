"""
eval_with_gating_analysis.py
Evaluation script with gating weight analysis.
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
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from collections import defaultdict

sys.path.append('..')
from models import EncryptedTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed


class GatingWeightCollector:
    """
    Gating weight collector.
    Collects AdaptiveHeadGating outputs for each layer using the PyTorch hook mechanism.
    """
    def __init__(self, model):
        self.model = model
        self.gating_weights = defaultdict(list)  # {layer_idx: [weights]}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Registers forward hooks to all AdaptiveHeadGating modules."""
        # Iterate through all layers in packet_encoder
        for layer_idx, layer in enumerate(self.model.packet_encoder.layers):
            # Access head_gating within the attention module
            if hasattr(layer.attention, 'head_gating'):
                gating_module = layer.attention.head_gating
                
                # Create a closure to capture the current layer_idx
                def make_hook(idx):
                    def hook(module, input, output):
                        # output shape: (batch_size, num_heads)
                        self.gating_weights[idx].append(output.detach().cpu())
                    return hook
                
                hook_handle = gating_module.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook_handle)
                print(f"✓ Hook registered for layer {layer_idx}")
    
    def clear(self):
        """Clears all collected weights."""
        self.gating_weights = defaultdict(list)
    
    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_all_weights(self):
        """
        Retrieves all collected weights.
        
        Returns:
            dict: {layer_idx: tensor of shape (num_samples, num_heads)}
        """
        result = {}
        for layer_idx, weight_list in self.gating_weights.items():
            if weight_list:
                # Concatenate weights from all batches
                result[layer_idx] = torch.cat(weight_list, dim=0)
        return result


def evaluate_with_gating_analysis(args):
    """Evaluation function with gating weight analysis."""
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load class labels
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    if isinstance(label_map, dict):
        class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    else:
        class_names = label_map
    
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")
    
    # Load test dataset
    test_h5 = os.path.join(args.data_dir, "test_data.h5")
    test_dataset = TrafficDataset(test_h5, augmentation=False, max_packets=10, max_bytes=300, use_stats=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"{'='*60}")
    print(f"Evaluation with Gating Weight Analysis")
    print(f"{'='*60}")
    print(f"Dataset: {args.data_dir}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"{'='*60}\n")
    
    # Initialize model
    model = EncryptedTrafficClassifier(
        num_classes=num_classes,
        d_byte=128, d_packet=260, byte_layers=3, packet_layers=4, num_heads=5,
        max_packets=100, max_bytes=300,
        use_stats=False, use_adaptive_gating=True
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.heavy_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # Remove 'module.' prefix if saved with DataParallel
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Initialize gating weight collector
    collector = GatingWeightCollector(model)
    print(f"✓ Gating weight collector initialized\n")
    
    # Inference and data collection
    all_preds = []
    all_labels = []
    
    print("Starting inference and collecting gating weights...\n")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            # Forward pass (hooks will automatically capture gating weights)
            logits = model(batch_data)
            _, preds = logits.max(1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Aggregate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Retrieve all collected gating weights
    gating_weights = collector.get_all_weights()
    
    # Cleanup: Remove hooks
    collector.remove_hooks()
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"Classification Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Save gating weight analysis data
    save_data = {
        'gating_weights': {layer_idx: weights.numpy() for layer_idx, weights in gating_weights.items()},
        'labels': all_labels,
        'predictions': all_preds,
        'class_names': class_names,
        'head_names': ['Content', 'Temporal', 'Size', 'Direction', 'TCP'],
        'num_layers': len(gating_weights),
        'num_samples': len(all_labels),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision * 100),
            'recall': float(recall * 100),
            'f1': float(f1 * 100)
        }
    }
    
    np.savez(args.save_gating_data, **save_data)
    print(f"✓ Gating weights saved to {args.save_gating_data}")
    
    # Print gating weight statistics
    print(f"\n{'='*60}")
    print(f"Gating Weights Statistics")
    print(f"{'='*60}")
    
    head_names = ['Content', 'Temporal', 'Size', 'Direction', 'TCP']
    
    for layer_idx, weights in sorted(gating_weights.items()):
        print(f"\nLayer {layer_idx}:")
        weights_np = weights.numpy()
        
        for head_idx, head_name in enumerate(head_names):
            head_weights = weights_np[:, head_idx]
            print(f"  {head_name:12s}: mean={head_weights.mean():.4f}, "
                  f"std={head_weights.std():.4f}, "
                  f"min={head_weights.min():.4f}, "
                  f"max={head_weights.max():.4f}")
    
    return save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--heavy_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_gating_data', type=str, default='gating_weights.npz',
                        help='Path to save gating weights data')
    
    args = parser.parse_args()
    evaluate_with_gating_analysis(args)