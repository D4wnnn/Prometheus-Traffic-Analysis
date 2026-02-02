# File: distillation/train_light_offline_ddp_global.py

"""
Distillation training using a global selector.
Key differences from the original version:
1. Uses global_selection_path instead of per-sample selection.
2. Teacher and Student use the same global selector, but with different geometries.
3. Reuses core training loops from train_light_offline_ddp.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from models import EncryptedTrafficClassifier
from models.light_model import LightTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed
from distillation_utils import DistillationLoss

# Note: train_epoch_distill maintains the same functionality as train_epoch, 
# with added detailed stats for CE/KD loss.
from train_light_offline_ddp import cleanup, train_epoch_distill, setup


# === Modification 2: Removed local cleanup(), directly using the imported version ===


def build_teacher(num_classes, teacher_path, device, rank, max_packets, max_bytes):
    """Constructs the Teacher model (using full view)."""
    teacher = EncryptedTrafficClassifier(
        num_classes=num_classes,
        d_byte=128,
        d_packet=260,
        byte_layers=3,
        packet_layers=4,
        num_heads=5,
        d_ff=1024,
        max_bytes=max_bytes,
        max_packets=100,
        dropout=0.1,
        use_stats=False,
        use_adaptive_gating=True
    ).to(device)
    
    if rank == 0:
        print(f"Loading teacher from {teacher_path}")
    
    ckpt = torch.load(teacher_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    teacher.load_state_dict(state_dict, strict=True)
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    
    return teacher


def build_student(num_classes, device, rank):
    """Constructs the Student model."""
    student = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)
    
    if rank == 0:
        params = sum(p.numel() for p in student.parameters())
        print(f"Student parameters: {params:,}")
    
    return student


@torch.no_grad()
def evaluate(student, dataloader, device, rank):
    """
    Validation function (kept here as train_light_offline_ddp.py handles validation within main).
    """
    student.eval()
    correct = 0
    total = 0
    
    for batch in dataloader:
        labels = batch["label"].to(device)
        batch_data = {k: v.to(device) for k, v in batch.items() if k != "label"}
        
        logits = student(batch_data)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    metrics = torch.tensor([correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    return 100.0 * metrics[0].item() / max(metrics[1].item(), 1)


def main(rank, world_size, args):
    setup(rank, world_size)
    set_seed(args.seed, rank=rank)
    device = torch.device(f'cuda:{rank}')
    
    # Load global selection configuration
    selector_data = np.load(args.global_selector, allow_pickle=True)
    global_config = json.loads(str(selector_data["config"]))
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Global Selection Config:")
        print(f"  Strategy: {global_config['strategy']}")
        print(f"  Target packets: {global_config['top_packets']}")
        if 'byte_range' in global_config:
            br = global_config['byte_range']
            print(f"  Byte range: [{br['start']}, {br['end']})")
        print("=" * 60 + "\n")
    
    # Data paths
    train_h5 = os.path.join(args.data_dir, "train_data.h5")
    val_h5 = os.path.join(args.data_dir, "val_data.h5")
    
    with open(os.path.join(args.data_dir, "label_mapping.json"), "r") as f:
        num_classes = len(json.load(f))
    
    # Dataset construction
    # Teacher Perspective: Full data (no selector)
    teacher_train_ds = TrafficDataset(
        train_h5,
        augmentation=False,
        max_packets=args.teacher_packets,
        max_bytes=args.teacher_bytes,
        global_selection_path=None  # Teacher views full data
    )
    
    # Student Perspective: Uses global selector
    student_train_ds = TrafficDataset(
        train_h5,
        augmentation=args.student_aug,
        max_packets=args.teacher_packets,  # Load teacher size first
        max_bytes=args.teacher_bytes,
        global_selection_path=args.global_selector  # Then crop with selector
    )
    
    student_val_ds = TrafficDataset(
        val_h5,
        augmentation=False,
        max_packets=args.teacher_packets,
        max_bytes=args.teacher_bytes,
        global_selection_path=args.global_selector
    )
    
    train_sampler = DistributedSampler(student_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(student_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        student_train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        student_val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model construction
    student_packets = global_config['top_packets']
    if 'byte_range' in global_config:
        student_bytes = global_config['byte_range']['end'] - global_config['byte_range']['start']
    else:
        student_bytes = global_config['window_len'] * global_config['windows_per_packet']
    
    if rank == 0:
        print(f"Student geometry: {student_packets} packets × {student_bytes} bytes")
    
    # Teacher uses full view
    teacher = build_teacher(
        num_classes, args.teacher_path, device, rank,
        max_packets=args.teacher_packets,
        max_bytes=args.teacher_bytes
    )
    
    student = build_student(num_classes, device, rank)
    student = DDP(student, device_ids=[rank], output_device=rank)
    
    # Training setup
    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # === Modification 4: Use imported train_epoch_distill ===
        train_loss, train_acc = train_epoch_distill(
            student, teacher, train_loader, criterion, optimizer, device, rank
        )
        val_acc = evaluate(student, val_loader, device, rank)
        
        scheduler.step()
        
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(args.save_dir, "light_model_global.pth")
                torch.save({
                    'model_state_dict': student.module.state_dict(),
                    'global_config': global_config,
                    'best_acc': best_acc
                }, save_path)
                print(f"✓ Best model saved (Val Acc: {best_acc:.2f}%)")
    
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--global_selector', type=str, required=True,
                        help='Path to global selector .npz file')
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    
    parser.add_argument('--teacher_packets', type=int, default=10)
    parser.add_argument('--teacher_bytes', type=int, default=300)
    
    parser.add_argument('--student_aug', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--master_port', type=int, default=12360)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)