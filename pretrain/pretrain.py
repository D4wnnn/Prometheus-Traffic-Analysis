"""
Main Pre-training Script
Supports Distributed Data Parallel (DDP) + checkpoint resume + step-based saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from pretrain_dataset import PretrainDataset, collate_fn
from pretrain_model import PretrainWrapper


def setup_debugger(rank):
    """Enable debugger on rank 0 only (only when --debug is passed)."""
    if rank != 0:
        return
    try:
        import debugpy
        debugpy.listen(("localhost", 9505))
        print("Waiting for debugger attach on rank 0")
        debugpy.wait_for_client()
        print("Debugger attached to rank 0")
    except Exception as e:
        print(f"Debugger setup failed: {e}")


def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
                    batch_idx, losses, save_path, is_step_ckpt=False):
    """
    Save training checkpoint
    
    Args:
        model: DDP model
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        global_step: Total global steps
        batch_idx: Current batch index within the epoch
        losses: Dictionary of current losses
        save_path: Path to save the checkpoint
        is_step_ckpt: Whether this is a step-based checkpoint
    """
    backbone = model.module.backbone
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'batch_idx': batch_idx,  # Used for intra-epoch recovery
        'backbone_state_dict': backbone.state_dict(),
        'model_state_dict': model.module.state_dict(),  # Save full pre-training model
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
    }
    
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, rank):
    """
    Load training checkpoint
    
    Returns:
        start_epoch: Epoch to resume from
        global_step: Global step to resume from
        start_batch_idx: Batch index within epoch to resume from
    """
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load to CPU first to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Compatibility for older checkpoints with backbone only
        model.module.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    
    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    start_batch_idx = checkpoint.get('batch_idx', 0)
    
    if rank == 0:
        print(f"Resumed from epoch {start_epoch}, global_step {global_step}, batch_idx {start_batch_idx}")
    
    return start_epoch, global_step, start_batch_idx


def find_latest_checkpoint(save_dir):
    """
    Find the most recent checkpoint based on global_step.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    
    ckpt_files = list(save_dir.glob('pretrain_*.pth'))
    
    if not ckpt_files:
        return None
    
    latest_ckpt = None
    max_step = -1
    
    for ckpt_file in ckpt_files:
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            step = ckpt.get('global_step', 0)
            if step > max_step:
                max_step = step
                latest_ckpt = ckpt_file
        except Exception as e:
            print(f"Warning: Failed to load {ckpt_file}: {e}")
            continue
    
    return latest_ckpt


def train_epoch(model, dataloader, optimizer, scheduler, device, rank, 
                task_weights, global_step, epoch, args, 
                start_batch_idx=0):
    """
    Train for one epoch
    
    Args:
        start_batch_idx: Batch index to start from (for resuming)
    """
    model.train()
    
    total_losses = {}
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}', initial=start_batch_idx)
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        # Skip processed batches during resume
        if batch_idx < start_batch_idx:
            continue
        
        # Move data to device
        batch_data = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        losses = model(batch_data, task='all')
        
        # Weighted loss combination
        total_loss = 0
        for task_name, weight in task_weights.items():
            for key, value in losses.items():
                if key.startswith(task_name):
                    total_loss += weight * value
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update LR every step
        scheduler.step()
        global_step += 1
        
        # Statistics
        num_batches += 1
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value.item()
        
        if rank == 0:
            display_dict = {
                'step': global_step,
                'total': f'{total_loss.item():.4f}',
                'mpm': f'{losses.get("mpm_packet_recon", torch.tensor(0)).item():.4f}',
                'cl': f'{losses.get("contrastive", torch.tensor(0)).item():.4f}',
                'stats_iat': f'{losses.get("stats_iat_stats", torch.tensor(0)).item():.4f}',
                'mm_type': f'{losses.get("mm_packet_type", torch.tensor(0)).item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            }
            pbar.set_postfix(display_dict)
        
        # ===== Step-based checkpoint saving =====
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            if rank == 0:
                save_path = Path(args.save_dir) / f'pretrain_step_{global_step}.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                avg_losses_so_far = {k: v / num_batches for k, v in total_losses.items()}
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, global_step, batch_idx + 1,
                    avg_losses_so_far, save_path, is_step_ckpt=True
                )
                print(f"\n✓ Step checkpoint saved to {save_path}")
                
                if args.max_step_ckpts > 0:
                    cleanup_old_checkpoints(
                        args.save_dir, 
                        prefix='pretrain_step_', 
                        max_keep=args.max_step_ckpts
                    )
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()} if num_batches > 0 else {}
    
    return avg_losses, global_step


def cleanup_old_checkpoints(save_dir, prefix, max_keep):
    """Cleanup old checkpoints, keeping only the most recent N files"""
    save_dir = Path(save_dir)
    ckpt_files = sorted(
        save_dir.glob(f'{prefix}*.pth'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    for ckpt_file in ckpt_files[max_keep:]:
        try:
            ckpt_file.unlink()
            print(f"  Deleted old checkpoint: {ckpt_file.name}")
        except Exception as e:
            print(f"  Warning: Failed to delete {ckpt_file}: {e}")


def main(rank, world_size, args):
    """Main training function"""
    if getattr(args, 'debug', False):
        setup_debugger(rank)
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print("="*60)
        print("Pre-training Configuration")
        print("="*60)
        print(f"Data: {args.data_file}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Mask probability: {args.mask_prob}")
        print(f"Save steps: {args.save_steps}")
        print(f"Resume: {args.resume}")
        print("="*60)
    
    backbone_config = {
        'd_byte': 128,
        'byte_layers': 3,
        'max_bytes': 350,
        'd_packet': 260,
        'packet_layers': 4,
        'd_ff': 1024,
        'max_packets': 10,
        'num_heads': 5,
        'dropout': 0.1,
        'use_stats': False,
        'stats_dim': 36,
        'use_adaptive_gating': True
    }
    
    # ===== Data Loading =====
    dataset = PretrainDataset(
        args.data_file,
        max_packets=backbone_config['max_packets'],
        max_bytes=backbone_config['max_bytes'],
        augmentation=True,
        aug_prob=0.5
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=32,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    model = PretrainWrapper(
        backbone_config=backbone_config,
        mask_prob=args.mask_prob,
        use_byte_mask=False
    ).to(device)
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    # ===== Optimizer =====
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # ===== Step Calculation =====
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs
    
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = max(2000, int(total_steps * 0.02))
    
    if rank == 0:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
    
    # ===== Learning Rate Scheduler =====
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
    
    # ===== Resume Logic =====
    start_epoch = 0
    global_step = 0
    start_batch_idx = 0
    
    if args.resume:
        if args.resume_path:
            checkpoint_path = args.resume_path
        else:
            checkpoint_path = find_latest_checkpoint(args.save_dir)
        
        if checkpoint_path is not None:
            start_epoch, global_step, start_batch_idx = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device, rank
            )
            
            if start_batch_idx >= steps_per_epoch:
                start_epoch += 1
                start_batch_idx = 0
                if rank == 0:
                    print(f"Epoch {start_epoch} completed, starting from epoch {start_epoch + 1}")
        else:
            if rank == 0:
                print("No checkpoint found, starting from scratch")
    
    # ===== Task Weights =====
    task_weights = {
        'mpm': 1.0,
        'mm': 0.5,
        'contrastive': 0.3,
        'stats': 0.2
    }
    
    # ===== Training Loop =====
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*60}")
        
        current_start_batch = start_batch_idx if epoch == start_epoch else 0
        
        losses, global_step = train_epoch(
            model, dataloader, optimizer, scheduler,
            device, rank, task_weights, global_step, epoch, args,
            start_batch_idx=current_start_batch
        )
        
        if rank == 0:
            print("\nEpoch Losses:")
            for key, value in losses.items():
                print(f"  {key}: {value:.4f}")
            
            save_path = Path(args.save_dir) / f'pretrain_epoch_{epoch+1}.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, global_step, 0,
                losses, save_path, is_step_ckpt=False
            )
            print(f"✓ Epoch checkpoint saved to {save_path}")
    
    if rank == 0:
        print(f"\n{'='*60}\nPre-training completed!\n{'='*60}")
    
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training for Encrypted Traffic Classification')
    
    # Data arguments
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to pre-training data file (HDF5 format)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps (0 for automatic calculation at 2%% of total steps)')
    
    # Pre-training arguments
    parser.add_argument('--mask_prob', type=float, default=0.15,
                        help='Packet masking probability')
    
    # Saving arguments
    parser.add_argument('--save_dir', type=str, default='./pretrain_checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Interval (in steps) to save checkpoints (0 to disable)')
    parser.add_argument('--max_step_ckpts', type=int, default=0,
                        help='Maximum number of step checkpoints to keep (0 for no limit)')
    
    # Resume arguments
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Specific checkpoint path to resume from')
    
    # GPU arguments
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='GPU indices to use, comma-separated')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugpy listener on rank 0 (for IDE attach)')

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    
    import torch.multiprocessing as mp
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)