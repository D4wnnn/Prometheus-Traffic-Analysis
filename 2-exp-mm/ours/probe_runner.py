"""
Multi-modal Protocol Understanding Task Runner (Revised with Train/Val Split)
Supports all evaluation tasks with independent validation phases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import argparse
import sys
import json
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np

sys.path.append('../..')
sys.path.append('..')
from finetune.finetune import load_config_from_folder

from probe_dataset import MultiModalProbeDataset, probe_collate_fn, tcp_match_collate_fn
from probe_models import (
    get_backbone,
    DirectionProbeWrapper,
    IATAnomalyProbeWrapper,
    SizePredictionProbeWrapper,
    TCPMatchingProbeWrapper,
    ConsistencyProbeWrapper,
    MaskedRecoveryProbeWrapper
)

# Fix random seed for consistent Train/Val split
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def prepare_loaders(dataset, args, collate_fn=probe_collate_fn):
    """
    Unified data splitting and Loader preparation.
    80% Train, 20% Validation.
    """
    # 1. Few-shot sampling (if required)
    if args.few_shot_ratio < 1.0:
        subset_size = int(len(dataset) * args.few_shot_ratio)
        # Use fixed indices to ensure consistency across runs
        indices = list(range(subset_size))
        dataset = Subset(dataset, indices)
        print(f"Few-shot Mode: Using {subset_size} samples")

    # 2. Train/Val Split (8:2)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Use generator to ensure split reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"Data Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 3. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def filter_valid_samples(batch, device):
    """Filter invalid samples from the batch."""
    valid_mask = batch['valid'] == 1
    if not valid_mask.any():
        return None, 0
    
    filtered_batch = {}
    for k, v in batch.items():
        if k != 'valid':
            filtered_batch[k] = v[valid_mask].to(device)
    
    return filtered_batch, valid_mask.sum().item()


def run_direction_prediction(args, device):
    """Task 1: Direction Prediction"""
    print("\n" + "="*60)
    print("Task 1: Direction Prediction")
    print("="*60)
    
    dataset = MultiModalProbeDataset(args.data_file, task_type='direction', 
                                   max_bytes=args.data_max_bytes, max_packets=args.data_max_packets)
    
    train_loader, val_loader = prepare_loaders(dataset, args)
    
    backbone = get_backbone(args, device)
    model = DirectionProbeWrapper(backbone, d_packet=260, freeze_backbone=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    results = {'task': 'direction', 'epochs': []}
    
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            valid_mask = batch['valid'] == 1
            if not valid_mask.any(): continue
            
            filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
            labels = filtered_batch['label']
            
            optimizer.zero_grad()
            logits = model(filtered_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                valid_mask = batch['valid'] == 1
                if not valid_mask.any(): continue
                
                filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
                labels = filtered_batch['label']
                
                logits = model(filtered_batch)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': acc,
            'val_f1': f1
        }
        results['epochs'].append(epoch_result)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_result['train_loss']:.4f}, "
              f"Val Loss={epoch_result['val_loss']:.4f}, "
              f"Val Acc={acc*100:.2f}%, Val F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
    
    results['best_f1'] = best_f1
    return results


def run_iat_anomaly_detection(args, device):
    """Task 2: IAT Anomaly Detection"""
    print("\n" + "="*60)
    print("Task 2: IAT Anomaly Detection")
    print("="*60)
    
    dataset = MultiModalProbeDataset(args.data_file, task_type='iat_anomaly', 
                                   max_bytes=args.data_max_bytes, max_packets=args.data_max_packets)
    
    train_loader, val_loader = prepare_loaders(dataset, args)
    
    backbone = get_backbone(args, device)
    model = IATAnomalyProbeWrapper(backbone, d_packet=260, freeze_backbone=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    results = {'task': 'iat_anomaly', 'epochs': []}
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_mask = batch['valid'] == 1
            if not valid_mask.any(): continue
            filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
            
            optimizer.zero_grad()
            logits = model(filtered_batch)
            loss = criterion(logits, filtered_batch['label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                valid_mask = batch['valid'] == 1
                if not valid_mask.any(): continue
                filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
                
                logits = model(filtered_batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(filtered_batch['label'].cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_accuracy': acc,
            'val_f1': f1
        }
        results['epochs'].append(epoch_result)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_result['train_loss']:.4f}, "
              f"Val Acc={acc*100:.2f}%, Val F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
    
    results['best_f1'] = best_f1
    return results


def run_size_prediction(args, device):
    """Task 3: Size Sequence Prediction"""
    print("\n" + "="*60)
    print("Task 3: Size Sequence Prediction")
    print("="*60)
    
    dataset = MultiModalProbeDataset(args.data_file, task_type='size_pred', 
                                   max_bytes=args.data_max_bytes, max_packets=args.data_max_packets)
    
    train_loader, val_loader = prepare_loaders(dataset, args)
    
    backbone = get_backbone(args, device)
    # Ensure num_bins matches dataset definition (default to 10)
    num_bins = dataset.num_size_bins if hasattr(dataset, 'num_size_bins') else 10 
    model = SizePredictionProbeWrapper(backbone, d_packet=260, num_bins=num_bins, 
                                     freeze_backbone=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    results = {'task': 'size_pred', 'epochs': []}
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_mask = batch['valid'] == 1
            if not valid_mask.any(): continue
            filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
            
            optimizer.zero_grad()
            logits = model(filtered_batch)
            loss = criterion(logits, filtered_batch['label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                valid_mask = batch['valid'] == 1
                if not valid_mask.any(): continue
                filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
                
                logits = model(filtered_batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(filtered_batch['label'].cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_accuracy': acc,
            'val_f1_macro': f1
        }
        results['epochs'].append(epoch_result)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_result['train_loss']:.4f}, "
              f"Val Acc={acc*100:.2f}%, Val F1(macro)={f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
    
    results['best_accuracy'] = best_acc
    return results


def run_tcp_matching(args, device):
    """Task 4: TCP Request-Response Matching"""
    print("\n" + "="*60)
    print("Task 4: TCP Request-Response Matching")
    print("="*60)
    
    dataset = MultiModalProbeDataset(args.data_file, task_type='tcp_match', 
                                   max_bytes=args.data_max_bytes, max_packets=args.data_max_packets)
    
    # TCP task uses a specific collate_fn
    train_loader, val_loader = prepare_loaders(dataset, args, collate_fn=tcp_match_collate_fn)
    
    backbone = get_backbone(args, device)
    model = TCPMatchingProbeWrapper(backbone, d_packet=260, freeze_backbone=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    results = {'task': 'tcp_match', 'epochs': []}
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_mask = batch['valid'] == 1
            if not valid_mask.any(): continue
            filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
            
            optimizer.zero_grad()
            logits = model(filtered_batch)
            loss = criterion(logits, filtered_batch['label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                valid_mask = batch['valid'] == 1
                if not valid_mask.any(): continue
                filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
                
                logits = model(filtered_batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(filtered_batch['label'].cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_accuracy': acc,
            'val_f1': f1
        }
        results['epochs'].append(epoch_result)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_result['train_loss']:.4f}, "
              f"Val Acc={acc*100:.2f}%, Val F1={f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
    
    results['best_accuracy'] = best_acc
    return results


def run_consistency(args, device):
    """Task 5: Cross-Modal Consistency Check"""
    print("\n" + "="*60)
    print("Task 5: Cross-Modal Consistency")
    print("="*60)
    
    dataset = MultiModalProbeDataset(args.data_file, task_type='consistency', 
                                   max_bytes=args.data_max_bytes, max_packets=args.data_max_packets)
    
    train_loader, val_loader = prepare_loaders(dataset, args)
    
    backbone = get_backbone(args, device)
    model = ConsistencyProbeWrapper(backbone, d_packet=260, freeze_backbone=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    results = {'task': 'consistency', 'epochs': []}
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_mask = batch['valid'] == 1
            if not valid_mask.any(): continue
            filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
            
            optimizer.zero_grad()
            logits = model(filtered_batch)
            loss = criterion(logits, filtered_batch['label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                valid_mask = batch['valid'] == 1
                if not valid_mask.any(): continue
                filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
                
                logits = model(filtered_batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(filtered_batch['label'].cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_accuracy': acc,
            'val_f1': f1
        }
        results['epochs'].append(epoch_result)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_result['train_loss']:.4f}, "
              f"Val Acc={acc*100:.2f}%, Val F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            
    results['best_f1'] = best_f1
    return results


def run_masked_recovery(args, device):
    """Task 6: Context Recovery (Regression + Classification)"""
    print("\n" + "="*60)
    print("Task 6: Masked Context Recovery")
    print("="*60)
    
    dataset = MultiModalProbeDataset(args.data_file, task_type='masked_recovery', 
                                   max_bytes=args.data_max_bytes, max_packets=args.data_max_packets)
    
    train_loader, val_loader = prepare_loaders(dataset, args)
    
    backbone = get_backbone(args, device)
    model = MaskedRecoveryProbeWrapper(backbone, d_packet=260, freeze_backbone=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    results = {'task': 'masked_recovery', 'epochs': []}
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_metrics = {'size_mse': 0, 'iat_mse': 0, 'dir_loss': 0}
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_mask = batch['valid'] == 1
            if not valid_mask.any(): continue
            filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
            
            optimizer.zero_grad()
            size_pred, dir_pred, iat_pred = model(filtered_batch)
            
            loss_size = criterion_reg(size_pred, filtered_batch['target_size'])
            loss_dir = criterion_cls(dir_pred, filtered_batch['target_dir'])
            loss_iat = criterion_reg(iat_pred, filtered_batch['target_iat'])
            
            loss = loss_size + loss_dir + loss_iat
            loss.backward()
            optimizer.step()
            
            train_metrics['size_mse'] += loss_size.item()
            train_metrics['iat_mse'] += loss_iat.item()
            train_metrics['dir_loss'] += loss_dir.item()

        # --- Validation ---
        model.eval()
        val_metrics = {'size_mse': 0, 'iat_mse': 0, 'dir_correct': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                valid_mask = batch['valid'] == 1
                if not valid_mask.any(): continue
                filtered_batch = {k: v[valid_mask] for k, v in batch.items() if k != 'valid'}
                
                size_pred, dir_pred, iat_pred = model(filtered_batch)
                
                # Reg Metrics
                loss_size = criterion_reg(size_pred, filtered_batch['target_size'])
                loss_iat = criterion_reg(iat_pred, filtered_batch['target_iat'])
                
                val_metrics['size_mse'] += loss_size.item()
                val_metrics['iat_mse'] += loss_iat.item()
                
                # Class Metrics
                val_metrics['dir_correct'] += (dir_pred.argmax(1) == filtered_batch['target_dir']).sum().item()
                val_metrics['total'] += valid_mask.sum().item()
        
        # Aggregate
        dir_acc = val_metrics['dir_correct'] / max(val_metrics['total'], 1)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_size_mse': train_metrics['size_mse'] / len(train_loader),
            'val_size_mse': val_metrics['size_mse'] / len(val_loader),
            'val_iat_mse': val_metrics['iat_mse'] / len(val_loader),
            'val_dir_accuracy': dir_acc
        }
        results['epochs'].append(epoch_result)
        
        print(f"Epoch {epoch+1}: Size MSE={epoch_result['val_size_mse']:.4f}, "
              f"IAT MSE={epoch_result['val_iat_mse']:.4f}, "
              f"Dir Acc={dir_acc*100:.2f}%")
    
    return results


def run_all_tasks(args, device):
    """Run all evaluation tasks."""
    all_results = {
        'config': {
            'data_file': args.data_file,
            'pretrain_path': args.pretrain_path,
            'epochs': args.epochs,
            'few_shot_ratio': args.few_shot_ratio,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'tasks': {},
        'summary': {}  # Summary of best results
    }
    
    # Run tasks sequentially
    try:
        all_results['tasks']['direction'] = run_direction_prediction(args, device)
    except Exception as e:
        print(f"Error in Direction Task: {e}")

    try:
        all_results['tasks']['iat_anomaly'] = run_iat_anomaly_detection(args, device)
    except Exception as e:
        print(f"Error in IAT Anomaly Task: {e}")

    try:
        all_results['tasks']['size_pred'] = run_size_prediction(args, device)
    except Exception as e:
        print(f"Error in Size Pred Task: {e}")

    try:
        all_results['tasks']['tcp_match'] = run_tcp_matching(args, device)
    except Exception as e:
        print(f"Error in TCP Match Task: {e}")

    # ================= Summary Table =================
    print("\n" + "="*70)
    print("SUMMARY: Multi-Modal Protocol Understanding Results (Validation Set)")
    print("="*70)
    print(f"{'Task':<30} {'Best Metric':>20}")
    print("-"*70)
    
    for task_name, task_result in all_results['tasks'].items():
        if not task_result or 'epochs' not in task_result:
            metric = "Failed"
            val = None
        elif 'best_f1' in task_result:
            val = task_result['best_f1']
            metric = f"Val F1: {val:.4f}"
            all_results['summary'][task_name] = {'metric': 'f1', 'value': val}
        elif 'best_accuracy' in task_result:
            val = task_result['best_accuracy']
            metric = f"Val Acc: {val*100:.2f}%"
            all_results['summary'][task_name] = {'metric': 'accuracy', 'value': val}
        else:
            metric = "See logs"
            all_results['summary'][task_name] = {'metric': 'unknown', 'value': 0}
            
        print(f"{task_name:<30} {metric:>20}")
    
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    set_seed(42) # Set global random seed
    
    parser = argparse.ArgumentParser(description='Multi-Modal Protocol Understanding Tasks')
    parser.add_argument('--data_file', type=str, required=True, help='Path to HDF5 data file')
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to pre-trained weights, "none" for random init')
    parser.add_argument('--task', type=str, default='all', 
                        choices=['all', 'direction', 'iat_anomaly', 'size_pred', 
                                 'tcp_match', 'consistency', 'masked_recovery'],
                        help='Type of task to run')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--few_shot_ratio', type=float, default=1.0, help='Data sampling ratio')
    parser.add_argument('--save_results', type=str, default=None, help='Path to save results (JSON)')
    parser.add_argument('--random_initialization', action='store_true', help='Use random initialization')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    
    # Auto-load config
    if args.pretrain_path and args.pretrain_path.lower() != 'none':
        backbone_config, data_max_bytes, data_max_packets = load_config_from_folder(args.pretrain_path)
        args.data_max_bytes = data_max_bytes
        args.data_max_packets = data_max_packets
        # Override for specific teacher view size if necessary
        args.data_max_bytes = 300
        args.data_max_packets = 5
    else:
        # Default values for random initialization
        args.data_max_bytes = 1500
        args.data_max_packets = 64
        print("Warning: Using default max_bytes=1500, max_packets=64 for random initialization")

    # Execute tasks
    if args.task == 'all':
        results = run_all_tasks(args, device)
    elif args.task == 'direction':
        results = run_direction_prediction(args, device)
    elif args.task == 'iat_anomaly':
        results = run_iat_anomaly_detection(args, device)
    elif args.task == 'size_pred':
        results = run_size_prediction(args, device)
    elif args.task == 'tcp_match':
        results = run_tcp_matching(args, device)
    elif args.task == 'consistency':
        results = run_consistency(args, device)
    elif args.task == 'masked_recovery':
        results = run_masked_recovery(args, device)
    
    # Save results (includes summary field)
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results (including Best Metric Summary) saved to {args.save_results}")