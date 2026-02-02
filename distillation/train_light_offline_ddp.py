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
from tqdm import tqdm
import json
import h5py

sys.path.append('..')
from models import EncryptedTrafficClassifier
from models.light_model import LightTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed
from distillation_utils import DistillationLoss

def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358' # Use a port different from finetuning
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_epoch_distill(student, teacher, dataloader, criterion, optimizer, device, rank):
    """Executes one epoch of distillation training (DDP version)."""
    student.train()
    teacher.eval()
    
    total_loss = 0
    total_ce = 0
    total_kd = 0
    correct = 0
    total = 0
    
    # Progress bar displayed only on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc='Distilling')
    else:
        pbar = dataloader
    
    for batch in pbar:
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        
        # 1. Teacher Inference (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(batch_data)
            
        # 2. Student Forward Pass
        optimizer.zero_grad()
        student_logits = student(batch_data)
        
        # 3. Compute Loss
        loss, ce_item, kd_item = criterion(student_logits, teacher_logits, labels)
        
        # 4. Backward Pass (DDP automatically synchronizes student gradients)
        loss.backward()
        optimizer.step()
        
        # 5. Local Statistics
        total_loss += loss.item()
        total_ce += ce_item
        total_kd += kd_item
        
        _, predicted = student_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
    # 6. Aggregate metrics from all GPUs for logging
    metrics = torch.tensor([total_loss, total_ce, total_kd, correct, total], 
                           dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    # Compute global averages
    global_total = metrics[4].item()
    avg_loss = metrics[0].item() / len(dataloader) / dist.get_world_size() # Approximate average
    avg_acc = 100. * metrics[3].item() / global_total
    
    return avg_loss, avg_acc

def main(rank, world_size, args):
    setup(rank, world_size)
    set_seed(args.seed)
    device = torch.device(f'cuda:{rank}')
    
    # ===== 1. Data Loading (Using DistributedSampler) =====
    train_h5 = os.path.join(args.data_dir, "train_data.h5")
    val_h5 = os.path.join(args.data_dir, "val_data.h5")
    
    with open(os.path.join(args.data_dir, "label_mapping.json"), 'r') as f:
        num_classes = len(json.load(f))
        
    train_dataset = TrafficDataset(train_h5, augmentation=True, max_bytes=150, max_packets=10)
    val_dataset = TrafficDataset(val_h5, augmentation=False, max_bytes=150, max_packets=10)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # ===== 2. Load Teacher (Heavy Model) =====
    # Teacher does not need DDP wrapping as it does not update parameters
    teacher = EncryptedTrafficClassifier(
        num_classes=num_classes,
        d_byte=128,
        d_packet=260,
        byte_layers=3,
        packet_layers=4,
        num_heads=5,
        d_ff=1024,
        max_bytes=300,
        max_packets=100,
        dropout=0.1,
        use_stats=False,
        stats_dim=36,
        use_adaptive_gating=True
    ).to(device)

    # Handle weight loading (stripping possible 'module.' prefixes)
    if os.path.exists(args.teacher_path):
        if rank == 0:
            print(f"Loading teacher weights from {args.teacher_path}")
        checkpoint = torch.load(args.teacher_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        teacher.load_state_dict(new_state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Teacher weights not found: {args.teacher_path}")
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # ===== 3. Load Student (Light Model) =====
    student = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)
    if rank == 0:
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        
        print(f"\n{'='*50}")
        print(f"Distillation Model Statistics:")
        print(f"{'='*50}")
        print(f"Teacher (Heavy) Params: {teacher_params:,}")
        print(f"Student (Light) Params: {student_params:,}")
        print(f"{'-'*50}")
        print(f"Compression Ratio:      {teacher_params / student_params:.2f}x smaller")
        print(f"Parameter Reduction:    {100 * (1 - student_params / teacher_params):.2f}% reduced")
        print(f"{'='*50}\n")

    # Student requires DDP wrapping
    student = DDP(student, device_ids=[rank], output_device=rank)
    
    # ===== 4. Training Setup =====
    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
        # Training phase
        train_loss, train_acc = train_epoch_distill(
            student, teacher, train_loader, criterion, optimizer, device, rank
        )
        
        # Validation phase
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(device)
                
                logits = student(batch_data)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        # Aggregate validation results
        metrics = torch.tensor([correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_acc = 100. * metrics[0].item() / metrics[1].item()
        
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        # Save best model (Rank 0 only)
        if rank == 0 and val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, "light_model_initial.pth")
            torch.save(student.module.state_dict(), save_path)
            print(f"✓ Best model saved to {args.save_dir}")
            
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=str, default='0,1,2,3') # Default to multi-GPU
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)