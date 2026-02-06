import os
# Set strict determinism for CUDA primitives (must be before torch use)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42' # Set default, will be overridden by seed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim.swa_utils as swa_utils
import torch.distributed as dist
import h5py
import importlib.util
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys
import random
sys.path.append('..')
from pretrain.pretrain_utils import DataAugmentation
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models import EncryptedTrafficClassifier


def load_config_from_folder(pretrain_path_str):
    """
    Load backbone_config.py from the folder where pre-trained weights are located.
    """
    pretrain_path = Path(pretrain_path_str)
    # Assume the config file is in the same directory as the weight file
    config_path = pretrain_path.parent / "backbone_config.py"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    print(f"Loading configuration from: {config_path}")
    
    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("backbone_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Retrieve variables
    backbone_config = getattr(config_module, 'backbone_config', None)
    data_max_bytes = getattr(config_module, 'data_max_bytes', None)
    data_max_packets = getattr(config_module, 'data_max_packets', None)
    
    if backbone_config is None:
        raise ValueError("backbone_config not found in config file")
        
    return backbone_config, data_max_bytes, data_max_packets

def load_config_from_file(config_path_str):
    """
    Directly load backbone_config.py from a specified path.
    """
    config_path = Path(config_path_str)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    print(f"Loading configuration from: {config_path}")
    
    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("backbone_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Retrieve variables
    backbone_config = getattr(config_module, 'backbone_config', None)
    data_max_bytes = getattr(config_module, 'data_max_bytes', None)
    data_max_packets = getattr(config_module, 'data_max_packets', None)
    
    if backbone_config is None:
        raise ValueError("backbone_config not found in config file")
        
    return backbone_config, data_max_bytes, data_max_packets

def worker_init_fn(worker_id):
    """Set the seed for each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def set_seed(seed=42, rank=0):
    """
    Set all random seeds to ensure reproducibility.
    
    Args:
        seed: Random seed value.
        rank: Current process rank.
    """
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Force the use of deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Disable non-deterministic SDPA optimizations
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class TrafficDataset:
    """Dataset for downstream tasks (supports data truncation)."""
    def __init__(self, h5_file, use_stats=False, augmentation=False, aug_prob=0.8,
                 max_bytes=None, max_packets=None, global_selection_path=None):
        """
        Args:
            h5_file: Path to the HDF5 data file.
            use_stats: Whether to use statistical features.
            augmentation: Whether to enable data augmentation.
            aug_prob: Probability of applying data augmentation.
            max_bytes: Max bytes per packet (truncated if exceeded), None for no truncation.
            max_packets: Max packets per flow (truncated if exceeded), None for no truncation.
        """
        self.h5_file = h5_file
        self.use_stats = use_stats
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.max_bytes = max_bytes
        self.max_packets = max_packets
        self.data_aug = DataAugmentation() if augmentation else None
        
        with h5py.File(h5_file, 'r') as f:
            self.length = f['labels'].shape[0]
            self.has_tcp_data = 'tcp_data_view' in f
            
            # Retrieve original data dimensions for logging
            self.original_bytes = f['bytes_nlp_view'].shape[-1] if 'bytes_nlp_view' in f else None
            self.original_packets = f['bytes_nlp_view'].shape[1] if 'bytes_nlp_view' in f else None

        self.global_selection = None
        if global_selection_path is not None:
            data = np.load(global_selection_path, allow_pickle=True)
            self.global_selection = json.loads(str(data["config"]))
            print(f"Loaded global selection: {self.global_selection['strategy']}")
            print(f"  Packets: {self.global_selection.get('packet_indices', 'per_packet')}")
            if 'byte_range' in self.global_selection:
                br = self.global_selection['byte_range']
                print(f"  Bytes: [{br['start']}, {br['end']})")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # Read raw data
            bytes_nlp = f['bytes_nlp_view'][idx]  # (N, L)
            seq_nlp = f['seq_nlp_view'][idx]      # (N, 3)
            tcp_data = f['tcp_data_view'][idx]    # (N, 3)
            label = f['labels'][idx]
        
        # Truncate to the teacher view size
        bytes_nlp = bytes_nlp[:self.max_packets, :self.max_bytes]
        seq_nlp = seq_nlp[:self.max_packets]
        tcp_data = tcp_data[:self.max_packets]
        
        # Apply selection strategy
        if self.global_selection is not None:
            bytes_nlp, seq_nlp, tcp_data = self._apply_global_selection(
                bytes_nlp, seq_nlp, tcp_data
            )
        
        data = {
            'bytes_nlp': torch.tensor(bytes_nlp, dtype=torch.long),
            'seq_nlp': torch.tensor(seq_nlp, dtype=torch.float32),
            'tcp_data': torch.tensor(tcp_data, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }        
        # Data augmentation
        if self.augmentation and torch.rand(1).item() < self.aug_prob:
            data = self._apply_augmentation(data)
        
        return data

    def _apply_augmentation(self, data):
        """
        Apply data augmentation by randomly selecting one augmentation method.
        """
        aug_type = torch.randint(0, 4, (1,)).item()
        
        if aug_type == 0:
            # Byte Jitter
            data['bytes_nlp'] = self.data_aug.byte_jitter(
                data['bytes_nlp'].unsqueeze(0), corruption_prob=0.1
            ).squeeze(0)
        
        elif aug_type == 1:
            # Time Jitter
            data['seq_nlp'] = self.data_aug.time_jitter(
                data['seq_nlp'].unsqueeze(0), iat_noise_std=0.1
            ).squeeze(0)
        
        elif aug_type == 2:
            # Byte Truncation
            data['bytes_nlp'] = self.data_aug.byte_truncation(
                data['bytes_nlp'].unsqueeze(0), min_ratio=0.7
            ).squeeze(0)
        
        elif aug_type == 3:
            # Packet Dropout
            bytes_aug, seq_aug, tcp_aug = self.data_aug.packet_dropout(
                data['bytes_nlp'].unsqueeze(0),
                data['seq_nlp'].unsqueeze(0),
                data['tcp_data'].unsqueeze(0),
                dropout_prob=0.2
            )
            data['bytes_nlp'] = bytes_aug.squeeze(0)
            data['seq_nlp'] = seq_aug.squeeze(0)
            data['tcp_data'] = tcp_aug.squeeze(0)
        
        return data
    
    def _apply_global_selection(self, bytes_nlp, seq_nlp, tcp_data):
        """
        Apply global selection configuration.
        
        Args:
            bytes_nlp: (N, L) Raw byte data.
            seq_nlp: (N, 3) Sequence features.
            tcp_data: (N, 3) TCP features.
        
        Returns:
            Tuple of (selected_bytes, selected_seq, selected_tcp).
        """
        config = self.global_selection
        strategy = config["strategy"]
        
        if strategy == "contiguous":
            # Select specific packets
            pkt_indices = config["packet_indices"]
            selected_bytes = bytes_nlp[pkt_indices]  # (top_packets, L)
            selected_seq = seq_nlp[pkt_indices]
            selected_tcp = tcp_data[pkt_indices]
            
            # Select specified byte range
            byte_start = config["byte_range"]["start"]
            byte_end = config["byte_range"]["end"]
            selected_bytes = selected_bytes[:, byte_start:byte_end]
            
        elif strategy == "greedy":
            # Select specific packets
            pkt_indices = config["packet_indices"]
            selected_seq = seq_nlp[pkt_indices]
            selected_tcp = tcp_data[pkt_indices]
            
            # Concatenate multiple windows
            windows = config["byte_windows"]
            byte_chunks = []
            for win in windows:
                chunk = bytes_nlp[pkt_indices, win["start"]:win["end"]]
                byte_chunks.append(chunk)
            selected_bytes = np.concatenate(byte_chunks, axis=1)
            
        elif strategy == "per_packet":
            # Per-packet configuration
            packet_configs = config["packet_configs"]
            selected_bytes_list = []
            selected_seq_list = []
            selected_tcp_list = []
            
            for pkt_cfg in packet_configs:
                pkt_idx = pkt_cfg["packet_idx"]
                windows = pkt_cfg["windows"]
                
                # Concatenate windows for this specific packet
                byte_chunks = [
                    bytes_nlp[pkt_idx, win["start"]:win["end"]]
                    for win in windows
                ]
                pkt_bytes = np.concatenate(byte_chunks, axis=0)
                
                selected_bytes_list.append(pkt_bytes)
                selected_seq_list.append(seq_nlp[pkt_idx])
                selected_tcp_list.append(tcp_data[pkt_idx])
            
            selected_bytes = np.stack(selected_bytes_list)
            selected_seq = np.stack(selected_seq_list)
            selected_tcp = np.stack(selected_tcp_list)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return selected_bytes, selected_seq, selected_tcp

def collate_fn(batch):
    """Batch processing function."""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def setup(rank, world_size):
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def load_pretrained_weights(model, pretrain_path, rank):
    """
    Load pre-trained backbone weights; parameters with shape mismatch are dropped.
    """
    if rank == 0:
        print(f"Loading pretrained weights from {pretrain_path}...")
    
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    
    # Get state_dict from checkpoint
    pretrained_state_dict = checkpoint['backbone_state_dict']
    
    # Remove classifier head keys from pretrained state dict
    keys_to_remove = []
    for key in pretrained_state_dict.keys():
        if key.startswith('classifier.'):
            keys_to_remove.append(key)
    
    if keys_to_remove and rank == 0:
        print(f"Removing classifier head keys from pretrained weights: {len(keys_to_remove)} keys")
    
    for key in keys_to_remove:
        del pretrained_state_dict[key]

    # Drop parameters with shape mismatch when config differs from checkpoint
    model_state_dict = model.state_dict()
    keys_mismatch = []

    # Iterate through weights to check for shape conflicts
    for key in list(pretrained_state_dict.keys()): 
        if key in model_state_dict:
            ckpt_shape = pretrained_state_dict[key].shape
            model_shape = model_state_dict[key].shape
            
            if ckpt_shape != model_shape:
                keys_mismatch.append(key)
                if rank == 0:
                    print(f"⚠️ Shape mismatch for {key}: Checkpoint {ckpt_shape} != Current Model {model_shape}. Dropping it.")
                del pretrained_state_dict[key]
    
    if rank == 0 and len(keys_mismatch) > 0:
        print(f"Dropped {len(keys_mismatch)} layers due to shape mismatch (likely W_O layer changes).")

    # Load backbone weights
    msg = model.load_state_dict(pretrained_state_dict, strict=False)
    
    if rank == 0:
        print("✓ Pretrained weights loaded successfully!")
        print(f"  Missing keys (initialized randomly): {len(msg.missing_keys)}")
        if 'epoch' in checkpoint:
            print(f"  Pretrained for {checkpoint['epoch']} epochs")

def freeze_layers(model, freeze_mode='none'):
    """
    Freeze specific layers (used for progressive fine-tuning).
    
    Args:
        model: EncryptedTrafficClassifier
        freeze_mode: 'none' | 'byte_encoder' | 'all_except_classifier'
    """
    if freeze_mode == 'none':
        return
    
    elif freeze_mode == 'byte_encoder':
        for param in model.byte_encoder.parameters():
            param.requires_grad = False
    
    elif freeze_mode == 'all_except_classifier':
        for param in model.byte_encoder.parameters():
            param.requires_grad = False
        for param in model.packet_encoder.parameters():
            param.requires_grad = False
    
    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")


def train_epoch(model, dataloader, criterion, optimizer, device, rank, world_size):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc='Training')
    else:
        pbar = dataloader
    
    for i, batch in enumerate(pbar):
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(batch_data)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
    
    # Synchronize metrics across all GPUs
    metrics = torch.tensor([total_loss * len(dataloader), correct, total], 
                           dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0].item() / (len(dataloader) * world_size)
    accuracy = 100. * metrics[1].item() / metrics[2].item()
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, rank, world_size):
    """
    Evaluate the model, calculating loss, accuracy, precision, recall, and f1.
    Returns global metrics aggregated from all GPUs.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc='Evaluating')
        else:
            pbar = dataloader
            
        for batch in pbar:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            logits = model(batch_data)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            
            all_preds.append(predicted)
            all_labels.append(labels)
    
    # Merge predictions and labels from all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Gather predictions and labels from all GPUs
    local_size = torch.tensor([all_preds.size(0)], device=device, dtype=torch.long)
    size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Calculate max size for padding
    max_size = max([s.item() for s in size_list])
    
    # Pad to the same length
    if all_preds.size(0) < max_size:
        padding = max_size - all_preds.size(0)
        all_preds = torch.cat([all_preds, torch.zeros(padding, dtype=all_preds.dtype, device=device)])
        all_labels = torch.cat([all_labels, torch.zeros(padding, dtype=all_labels.dtype, device=device)])
    
    # Collect data from all GPUs
    gathered_preds = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(world_size)]
    gathered_labels = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)
    
    # Compute metrics only on rank 0
    if rank == 0:
        # Remove padding and combine data
        final_preds = []
        final_labels = []
        for i in range(world_size):
            size = size_list[i].item()
            final_preds.append(gathered_preds[i][:size])
            final_labels.append(gathered_labels[i][:size])
        
        final_preds = torch.cat(final_preds).cpu().numpy()
        final_labels = torch.cat(final_labels).cpu().numpy()
        
        # Calculate scores
        accuracy = accuracy_score(final_labels, final_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            final_labels, final_preds, average='macro', zero_division=0
        )
        
        # Calculate average loss
        loss_tensor = torch.tensor([total_loss * len(dataloader)], device=device)
    else:
        loss_tensor = torch.tensor([total_loss * len(dataloader)], device=device)
        accuracy = precision = recall = f1 = 0
    
    # Synchronize loss
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor.item() / (len(dataloader) * world_size)
    
    # Broadcast metrics to all processes
    metrics = torch.tensor([accuracy, precision * 100, recall * 100, f1 * 100], 
                          dtype=torch.float32, device=device)
    dist.broadcast(metrics, src=0)
    
    return avg_loss, metrics[0].item(), metrics[1].item(), metrics[2].item(), metrics[3].item()


def main(rank, world_size, args):
    """Main function - Integrated with Stochastic Weight Averaging (SWA)."""
    set_seed(args.seed, rank)
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # ===== Load Data =====
    data_dir = Path(args.data_dir)
    train_h5 = data_dir / "train_data.h5"
    val_h5 = data_dir / "val_data.h5"
    label_map_file = data_dir / "label_mapping.json"
    
    if not val_h5.exists():
        raise FileNotFoundError(f"Validation data file not found: {val_h5}")
    
    with open(label_map_file, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    config_file_path = args.config_path
    
    # ===== Flexible Config Loading =====
    if config_file_path and Path(config_file_path).exists():
        model_config, data_max_bytes, data_max_packets = load_config_from_file(config_file_path)
    else:
        raise ValueError("Must provide --config_path or a valid --pretrain_path to load backbone_config.py")
    
    # ========== Handle Ablation Study Parameters ==========
    disabled_head_indices = []
    if args.disable_heads:
        # Convert string "1,2,4" to list [1, 2, 4]
        try:
            disabled_head_indices = [int(x.strip()) for x in args.disable_heads.split(',') if x.strip()]
            if rank == 0:
                print(f"\n{'!'*40}")
                print(f" Ablation Study Active: Disabling Bias for Heads: {disabled_head_indices}")
                print(f"{'!'*40}\n")
        except ValueError:
            raise ValueError("Error parsing --disable_heads. Ensure format is like '1,2,4'")
    
    # Add ablation parameters to model_config
    model_config['disabled_head_indices'] = disabled_head_indices
    
    if rank == 0:
        print(f"Number of classes: {num_classes}")
        print(f"SWA enabled: True")
        print(f"SWA will start at epoch: {args.epochs - args.swa_epochs + 1}")
    
    # Use augmentation for training, none for validation
    train_dataset = TrafficDataset(
        train_h5, 
        use_stats=model_config['use_stats'], 
        augmentation=True, 
        aug_prob=0.8,
        max_bytes=data_max_bytes,
        max_packets=data_max_packets
    )
    val_dataset = TrafficDataset(
        val_h5, 
        use_stats=model_config['use_stats'], 
        augmentation=False,
        max_bytes=data_max_bytes,
        max_packets=data_max_packets
    )
    
    # Print truncation info
    if rank == 0:
        print(f"\nData truncation info:")
        print(f"  Original data shape: packets={train_dataset.original_packets}, bytes={train_dataset.original_bytes}")
        print(f"  Truncated to: packets={data_max_packets}, bytes={data_max_bytes}")
        if train_dataset.original_packets and train_dataset.original_packets > data_max_packets:
            print(f"  ⚠️  Packets will be truncated from {train_dataset.original_packets} to {data_max_packets}")
        if train_dataset.original_bytes and train_dataset.original_bytes > data_max_bytes:
            print(f"  ⚠️  Bytes will be truncated from {train_dataset.original_bytes} to {data_max_bytes}")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    g = torch.Generator()
    g.manual_seed(args.seed + rank)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=8, collate_fn=collate_fn, pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=8, collate_fn=collate_fn, pin_memory=True)
    
    # ===== Create Model =====
    model = EncryptedTrafficClassifier(
        num_classes=num_classes,
        **model_config
    ).to(device)
    
    # ===== Load Pre-trained Weights =====
    if args.pretrain_path and Path(args.pretrain_path).exists():
        load_pretrained_weights(model, args.pretrain_path, rank)
        
        if args.disable_heads and args.reset_weights:
            if rank == 0:
                print(f"\n{'!'*50}")
                print(f"Resetting weights for heads: {disabled_head_indices}")
                print(f"{'!'*50}")
            
            # Iterate through layers and call weight reset function in attention.py
            for i, layer in enumerate(model.packet_encoder.layers):
                layer.attention.reset_ablated_heads_weights()
            
            if rank == 0:
                print("✓ All specified heads have been re-initialized to random noise.\n")
    else:
        if rank == 0:
            print("No pretrained weights specified, training from scratch...")
    
    # ===== Progressive Unfreezing (Optional) =====
    if args.freeze_mode != 'none':
        freeze_layers(model, args.freeze_mode)
        if rank == 0:
            print(f"Freeze mode: {args.freeze_mode}")
            
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # ===== Optimizer and Scheduler =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Standard LR scheduler
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, 
                                     total_iters=args.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=0)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                 milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    # ===== SWA Setup =====
    swa_model = swa_utils.AveragedModel(model)
    swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=args.swa_lr)
    swa_start_epoch = args.epochs - args.swa_epochs
    
    if rank == 0:
        print(f"\nSWA Configuration:")
        print(f"  - SWA start epoch: {swa_start_epoch + 1}")
        print(f"  - SWA learning rate: {args.swa_lr}")
        print(f"  - Total SWA epochs: {args.swa_epochs}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*40}")
        print(f"Model Statistics:")
        print(f"{'='*40}")
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Trainable Ratio:      {100 * trainable_params / total_params:.2f}%")
        print(f"{'='*40}\n")
    
    # ===== Training Loop =====
    best_val_f1 = 0
    best_metrics = None
    best_swa_f1 = 0
    best_swa_metrics = None
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        g.manual_seed(args.seed + rank + epoch)
        
        if rank == 0:
            swa_status = f" [SWA Active]" if epoch >= swa_start_epoch else ""
            print(f"\n{'='*60}\nEpoch {epoch+1}/{args.epochs}{swa_status}\n{'='*60}")
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                            device, rank, world_size)
        
        # Update SWA parameters
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if rank == 0:
                print(f"✓ SWA model updated (averaging {epoch - swa_start_epoch + 1} checkpoints)")
        else:
            scheduler.step()
        
        # Regular Model Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, device, rank, world_size
        )
        
        if rank == 0:
            print(f"\n[Regular Model]")
            print(f"  Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%, F1: {val_f1:.2f}%")
        
        # Save best regular model checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if rank == 0:
                best_metrics = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1
                }
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': best_metrics,
                }, args.save_path)
                
                print(f"\n{'*'*60}")
                print(f"✓ New Best Regular Model! (Val F1: {val_f1:.2f}%)")
                print(f"Model saved to: {args.save_path}")
                print(f"{'*'*60}\n")
        
        # SWA Model Evaluation (once SWA is active)
        if epoch >= swa_start_epoch:
            if rank == 0:
                print(f"\n[SWA Model] Updating BN statistics...")
            
            # Update BatchNorm statistics using a subset of the training set
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            
            swa_val_loss, swa_val_acc, swa_val_p, swa_val_r, swa_val_f1 = evaluate(
                swa_model, val_loader, criterion, device, rank, world_size
            )
            
            if rank == 0:
                print(f"[SWA Model]")
                print(f"  Validation - Loss: {swa_val_loss:.4f}, Acc: {swa_val_acc:.2f}%, "
                      f"Precision: {swa_val_p:.2f}%, Recall: {swa_val_r:.2f}%, F1: {swa_val_f1:.2f}%")
            
            # Save best SWA model checkpoint
            if swa_val_f1 > best_swa_f1:
                best_swa_f1 = swa_val_f1
                if rank == 0:
                    best_swa_metrics = {
                        'epoch': epoch,
                        'val_loss': swa_val_loss,
                        'val_acc': swa_val_acc,
                        'val_precision': swa_val_p,
                        'val_recall': swa_val_r,
                        'val_f1': swa_val_f1
                    }
                    swa_save_path = args.save_path.replace('.pth', '_swa.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': swa_model.module.state_dict(),
                        'val_metrics': best_swa_metrics,
                    }, swa_save_path)
                    
                    print(f"\n{'*'*60}")
                    print(f"✓ New Best SWA Model! (Val F1: {swa_val_f1:.2f}%)")
                    print(f"SWA Model saved to: {swa_save_path}")
                    print(f"{'*'*60}\n")
    
    # ===== Final Evaluation Summary =====
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{'TRAINING COMPLETED':^70}")
        print(f"{'='*70}\n")
        
        print(f"{'Regular Model Best Performance':^70}")
        print(f"{'-'*70}")
        print(f"  Epoch:     {best_metrics['epoch']+1}/{args.epochs}")
        print(f"  Loss:      {best_metrics['val_loss']:.4f}")
        print(f"  Accuracy:  {best_metrics['val_acc']:.2f}%")
        print(f"  Precision: {best_metrics['val_precision']:.2f}%")
        print(f"  Recall:    {best_metrics['val_recall']:.2f}%")
        print(f"  F1 Score:  {best_metrics['val_f1']:.2f}%")
        print(f"  Saved to:  {args.save_path}")
        
        if best_swa_metrics:
            print(f"\n{'SWA Model Best Performance':^70}")
            print(f"{'-'*70}")
            print(f"  Epoch:     {best_swa_metrics['epoch']+1}/{args.epochs}")
            print(f"  Loss:      {best_swa_metrics['val_loss']:.4f}")
            print(f"  Accuracy:  {best_swa_metrics['val_acc']:.2f}%")
            print(f"  Precision: {best_swa_metrics['val_precision']:.2f}%")
            print(f"  Recall:    {best_swa_metrics['val_recall']:.2f}%")
            print(f"  F1 Score:  {best_swa_metrics['val_f1']:.2f}%")
            swa_save_path = args.save_path.replace('.pth', '_swa.pth')
            print(f"  Saved to:  {swa_save_path}")
            
            print(f"\n{'Model Comparison':^70}")
            print(f"{'-'*70}")
            f1_diff = best_swa_metrics['val_f1'] - best_metrics['val_f1']
            if f1_diff > 0:
                print(f"  💡 SWA model is BETTER by {f1_diff:.2f}% F1")
                print(f"  ✓ Recommendation: Use SWA model ({swa_save_path})")
            elif f1_diff < -0.1:
                print(f"  ⚠️  Regular model is better by {-f1_diff:.2f}% F1")
                print(f"  ✓ Recommendation: Use Regular model ({args.save_path})")
            else:
                print(f"  ≈ Models are similar (diff: {f1_diff:.2f}% F1)")
                print(f"  ✓ Recommendation: Use either model")
        
        print(f"\n{'='*70}\n")
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning with Pretrained Weights')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory for downstream task data')
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help='Path to pre-trained weights (Optional, trains from scratch if omitted)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to backbone_config.py (Required if pretrain_path is not provided)')
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Fine-tuning strategy
    parser.add_argument('--freeze_mode', type=str, default='none',
                        choices=['none', 'byte_encoder', 'all_except_classifier'],
                        help='Layer freezing strategy')
    # SWA parameters
    parser.add_argument('--swa_epochs', type=int, default=10,
                        help='Number of epochs to use SWA at the end (default: 10)')
    parser.add_argument('--swa_lr', type=float, default=1e-3,
                        help='Learning rate for SWA (default: 1e-3)')
    # Saving arguments
    parser.add_argument('--save_path', type=str, default='finetuned_model.pth')
    
    # GPU arguments
    parser.add_argument('--gpus', type=str, default='0')
    
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
    parser.add_argument('--disable_heads', type=str, default='',
                        help='Indices of heads to disable bias, separated by commas. e.g., "1,3" for Time(1) and Direction(3). Options: 1,2,3,4')
    
    # [New] Reset weights switch
    parser.add_argument('--reset_weights', action='store_true', help='Force reset weights of ablated heads.')
    
    args = parser.parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    
    import torch.multiprocessing as mp
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)