import torch
import torch.optim as optim
import torch.nn as nn
import zmq
import argparse
import os
import sys
import json
import numpy as np

sys.path.append('../..')
from models.light_model import LightTrafficClassifier
from distillation.distillation_utils import DistillationLoss
from finetune.finetune import load_config_from_folder

PORT_RECV_DATA = 5556
PORT_PUB_WEIGHTS = 5557
UPDATE_INTERVAL = 5 # Frequency of broadcasting model weights

def apply_tensor_selection(batch_data, config):
    """Applies feature selection strategy to input tensors"""
    if config is None: return batch_data
    strategy = config.get("strategy", "contiguous")
    new_batch = {k: v for k, v in batch_data.items()}
    
    if strategy == "contiguous":
        pkt_indices = config["packet_indices"]
        byte_start, byte_end = config["byte_range"]["start"], config["byte_range"]["end"]
        if 'bytes_nlp' in new_batch: new_batch['bytes_nlp'] = new_batch['bytes_nlp'][:, pkt_indices, :][:, :, byte_start:byte_end]
        if 'seq_nlp' in new_batch: new_batch['seq_nlp'] = new_batch['seq_nlp'][:, pkt_indices, :]
        if 'tcp_data' in new_batch: new_batch['tcp_data'] = new_batch['tcp_data'][:, pkt_indices, :]
    elif strategy == "per_packet":
        packet_configs = config["packet_configs"]
        sb, ss, st = [], [], []
        for pc in packet_configs:
            pidx, wins = pc["packet_idx"], pc["windows"]
            if 'bytes_nlp' in new_batch: sb.append(torch.cat([new_batch['bytes_nlp'][:, pidx, w['start']:w['end']] for w in wins], dim=1))
            if 'seq_nlp' in new_batch: ss.append(new_batch['seq_nlp'][:, pidx, :])
            if 'tcp_data' in new_batch: st.append(new_batch['tcp_data'][:, pidx, :])
        if sb: new_batch['bytes_nlp'] = torch.stack(sb, dim=1)
        if ss: new_batch['seq_nlp'] = torch.stack(ss, dim=1)
        if st: new_batch['tcp_data'] = torch.stack(st, dim=1)
    return new_batch


class StableDistillationLoss(nn.Module):
    """Stable distillation loss implementation to prevent gradient explosion"""
    def __init__(self, alpha=0.3, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # 1. Hard Label Loss (Cross Entropy)
        ce = self.ce_loss(student_logits, labels)
        
        # 2. Soft Label Loss (KL Divergence with Normalization)
        # Normalize logits to prevent scale-induced instability
        t_std = teacher_logits.std(dim=-1, keepdim=True).clamp(min=1.0)
        s_std = student_logits.std(dim=-1, keepdim=True).clamp(min=1.0)
        
        teacher_norm = teacher_logits / t_std
        student_norm = student_logits / s_std
        
        soft_teacher = torch.softmax(teacher_norm / self.temperature, dim=-1)
        soft_student = torch.log_softmax(student_norm / self.temperature, dim=-1)
        
        kl = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 3. Combine losses (clamping KL contribution for stability)
        kl = kl.clamp(max=10.0)
        
        loss = (1 - self.alpha) * ce + self.alpha * kl
        return loss, ce, kl


def online_trainer(args):
    context = zmq.Context()
    
    socket_pull = context.socket(zmq.PULL)
    socket_pull.bind(f"tcp://*:{PORT_RECV_DATA}")
    
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind(f"tcp://*:{PORT_PUB_WEIGHTS}")
    
    device = torch.device(f'cuda:{args.gpu}')
    print(f"[Trainer] Running on {device}")
    
    with open(os.path.join(args.data_dir, "label_mapping.json"), 'r') as f:
        num_classes = len(json.load(f))
        
    model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)
    
    if os.path.exists(args.light_path):
        checkpoint = torch.load(args.light_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("[Trainer] Initial weights loaded.")
    
    # --- Critical Setup 1: Train Mode and BN Freezing ---
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval() # Keep BN statistics fixed during online learning
            for param in m.parameters():
                param.requires_grad = False
    
    # --- Critical Setup 2: Conservative Optimizer Settings ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-6,           # Low learning rate for stability
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # --- Critical Setup 3: Stable Distillation Loss ---
    criterion = StableDistillationLoss(alpha=0.3, temperature=2.0)
    
    selection_config = None
    if args.global_selector:
        data = np.load(args.global_selector, allow_pickle=True)
        selection_config = json.loads(str(data["config"]))

    step_counter = 0
    total_loss = 0.0
    
    # --- Critical Setup 4: Monitoring Variables ---
    initial_weight_norm = sum(p.norm().item() for p in model.parameters())
    
    print(f"[Trainer] Initial weight norm: {initial_weight_norm:.4f}")
    print("[Trainer] Ready. Waiting for data...")

    while True:
        try:
            packet = socket_pull.recv_pyobj()
            
            if packet == "STOP":
                print("[Trainer] Received STOP signal. Exiting.")
                break
            
            data_cpu = packet["data"]
            teacher_logits = packet["teacher_logits"].to(device)
            labels = packet["labels"].to(device)
            
            data_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in data_cpu.items()}
            
            light_inputs = apply_tensor_selection(data_gpu, selection_config)
            
            optimizer.zero_grad()
            if light_inputs['bytes_nlp'].sum() == 0:
                print("!!! CRITICAL WARNING: Trainer inputs are all ZEROS !!!")
            
            student_logits = model(light_inputs)
            
            loss, ce_loss, kl_loss = criterion(student_logits, teacher_logits, labels)

            if torch.isnan(loss) or loss.item() > 100:
                print(f"[WARNING] Abnormal loss detected: {loss.item():.4f}, skipping this batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            step_counter += 1
            total_loss += loss.item()
            
            # --- Weight Broadcasting ---
            if step_counter % UPDATE_INTERVAL == 0:
                # --- Critical Setup 8: Weight Integrity Check ---
                current_weight_norm = sum(p.norm().item() for p in model.parameters())
                weight_ratio = current_weight_norm / initial_weight_norm
                
                avg_loss = total_loss / UPDATE_INTERVAL
                print(f"[Trainer] Step {step_counter}: Avg Loss {avg_loss:.4f}, Weight norm ratio: {weight_ratio:.4f}")
                
                if weight_ratio < 0.1 or weight_ratio > 10:
                    print(f"[WARNING] Weight norm changed dramatically! Ratio: {weight_ratio:.4f}")
                
                state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                socket_pub.send_string("weights", flags=zmq.SNDMORE)
                socket_pub.send_pyobj(state_dict_cpu)
                
                total_loss = 0.0
                
        except KeyboardInterrupt:
            print("\n[Trainer] Interrupted.")
            break
        except Exception as e:
            print(f"[Trainer] Error: {e}")
            traceback.print_exc()
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--light_path', type=str, required=True)
    parser.add_argument('--global_selector', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    online_trainer(args)