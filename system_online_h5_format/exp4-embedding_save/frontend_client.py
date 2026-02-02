import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import argparse
import os
import sys
import time
import json
import numpy as np
import zmq
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

# Path configuration
sys.path.append('../..')
from models.light_model import LightTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, load_config_from_folder

# ================= Configuration Constants =================
BACKEND_ADDR = "tcp://localhost:5555"
TRAINER_BROADCAST_ADDR = "tcp://localhost:5557"

# ================= Reproducibility Seeds =================
np.random.seed(42)
torch.manual_seed(42)

# ================= Helper Functions =================
def apply_tensor_selection(batch_data, config):
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

# ================= Core Inference Engine =================
def run_inference_engine(loader, model, device, socket_push, socket_sub, selection_config, 
                         conf_threshold, enable_learning=True, stage_name="Train"):
    heavy_sent_count = 0
    correct_count = 0
    total_count = 0
    
    # Progress bar via tqdm
    pbar = tqdm(loader, desc=f"[{stage_name}]", leave=True)
    update_count = 0
    
    with torch.no_grad():
        for batch in pbar:
            # 1. Attempt to receive weights (in learning mode)
            try:
                # Flush the queue to ensure the most recent weights are loaded
                while True:
                    topic = socket_sub.recv_string(flags=zmq.NOBLOCK)
                    if topic == "weights":
                        state_dict = socket_sub.recv_pyobj()
                        model.load_state_dict(state_dict)
                        update_count += 1
            except zmq.Again: pass

            # 2. Inference
            labels = batch['label']
            inputs_cpu = {k: v for k, v in batch.items() if k != 'label'}
            inputs_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs_cpu.items()}
            
            light_inputs = apply_tensor_selection(inputs_gpu, selection_config)
            logits = model(light_inputs)
            
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(1)
            
            low_conf_mask = conf < conf_threshold
            
            # 3. Statistics
            preds_cpu = preds.cpu()
            correct_count += (preds_cpu == labels).sum().item()
            total_count += labels.size(0)
            
            # 4. Offload difficult cases
            if enable_learning:
                num_heavy = low_conf_mask.sum().item()
                if num_heavy > 0:
                    heavy_sent_count += num_heavy
                    mask_tensor_cpu = low_conf_mask.cpu()
                    socket_push.send_pyobj({
                        "data": {k: v[mask_tensor_cpu] for k, v in inputs_cpu.items()},
                        "indices": torch.zeros(num_heavy), 
                        "labels": labels[mask_tensor_cpu]
                    })
            else:
                heavy_sent_count += low_conf_mask.sum().item()
            
            # Real-time progress bar update
            curr_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
            pbar.set_postfix({"Acc": f"{curr_acc:.2f}%"})

    acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    offload_rate = (heavy_sent_count / total_count) * 100 if total_count > 0 else 0
    return acc, offload_rate, update_count

# ================= Main Program =================
def main(args):
    context = zmq.Context()
    socket_push = context.socket(zmq.PUSH)
    socket_push.connect(BACKEND_ADDR)
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect(TRAINER_BROADCAST_ADDR)
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, "weights")

    device = torch.device(f'cuda:{args.gpu}')
    
    # Set checkpoint directory
    ckpt_dir = os.path.join("checkpoints", args.log_file_prefix)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[Setup] Checkpoints will be saved to: {ckpt_dir}")

    # Setup configurations
    map_path = os.path.join(args.base_data_dir, "label_mapping.json")
    with open(map_path, 'r') as f: num_classes = len(json.load(f))
    
    try:
        _, heavy_max_bytes, heavy_max_packets = load_config_from_folder(args.pretrain_checkpoint_path)
    except:
        heavy_max_bytes, heavy_max_packets = 1500, 100

    selection_config = None
    if args.global_selector:
        selection_config = json.loads(str(np.load(args.global_selector, allow_pickle=True)["config"]))

    # Datasets
    print(f"\n[Setup] Loading datasets...")
    base_h5 = os.path.join(args.base_data_dir, "test_data.h5")
    ds_base = TrafficDataset(base_h5, augmentation=False, max_packets=heavy_max_packets, max_bytes=heavy_max_bytes)
    N = 200 
    indices = np.random.choice(len(ds_base), N, replace=False)
    sampler_base = SubsetRandomSampler(indices)
    loader_base = DataLoader(ds_base, batch_size=32, collate_fn=collate_fn, sampler=sampler_base, shuffle=False)

    drift_h5 = os.path.join(args.drift_data_dir, "train_data.h5")
    ds_drift = TrafficDataset(drift_h5, augmentation=False, max_packets=heavy_max_packets, max_bytes=heavy_max_bytes)
    drift_total = len(ds_drift)
    all_indices = np.random.permutation(drift_total)
    drift_probe_size = int(drift_total * 0.1)
    drift_train_size = drift_total - drift_probe_size
    train_indices = all_indices[:drift_train_size]
    probe_indices = all_indices[drift_train_size:]
    ds_drift_probe = Subset(ds_drift, probe_indices)
    loader_drift_probe = DataLoader(ds_drift_probe, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)
    if os.path.exists(args.light_path):
        checkpoint = torch.load(args.light_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    exp_logs = []

    # ================= PHASE 1: Base Baseline =================
    print("\n" + "="*50)
    print(f">>> PHASE 1: Base Dataset Processing ({args.phase1_epochs} Epochs)")
    print("="*50)
    
    model.eval()
    for epoch in range(args.phase1_epochs):
        stage_str = f"P1-Ep{epoch+1}"
        acc_p1, off_p1, update_count = run_inference_engine(loader_base, model, device, socket_push, socket_sub, 
                                                            selection_config, args.conf_threshold, 
                                                            enable_learning=True, 
                                                            stage_name=stage_str)
        exp_logs.append({"phase": 1, "epoch": epoch + 1, "desc": "base_baseline", "acc": acc_p1, "offload": off_p1, "updates": update_count})
        print(f"[{stage_str}] Acc: {acc_p1:.2f}%, Offload: {off_p1:.2f}%")
        
    # Save baseline model after Phase 1
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "phase1_baseline.pth"))
    time.sleep(2)

    # ================= PHASE 2: Concept Drift =================
    print("\n" + "="*50)
    print(f">>> PHASE 2: Concept Drift (Drift Dataset) - {args.drift_epochs_per_stage} Epochs per Stage")
    print("="*50)

    part_len = drift_train_size // args.num_parts
    update_count = 0
    for p_idx in range(args.num_parts):
        # 2.1 Exam (Evaluation)
        print(f"\n--- Phase 2 | Stage {p_idx+1}/{args.num_parts} : Exam ---")
        acc_exam, off_exam, _ = run_inference_engine(loader_drift_probe, model, device, socket_push, socket_sub,
                                                      selection_config, args.conf_threshold,
                                                      enable_learning=False, stage_name=f"P2-S{p_idx+1}-Exam")
        
        log_entry = {
            "phase": 2, "stage": p_idx + 1, "epoch": 0,
            "cumulative_samples": p_idx * part_len,
            "probe_acc": acc_exam, "probe_offload": off_exam, "updates": update_count
        }
        exp_logs.append(log_entry)
        print(f"[Exam Result] Acc: {acc_exam:.2f}%, Offload: {off_exam:.2f}%")

        # 2.2 Learn (Update)
        print(f"--- Phase 2 | Stage {p_idx+1}/{args.num_parts} : Learning ({args.drift_epochs_per_stage} Local Epochs) ---")
        start_idx = p_idx * part_len
        end_idx = (p_idx + 1) * part_len
        if p_idx == args.num_parts - 1: end_idx = drift_train_size
        
        current_part_indices = train_indices[start_idx : end_idx]
        curr_subset = Subset(ds_drift, current_part_indices)
        curr_loader = DataLoader(curr_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        
        for drift_ep in range(args.drift_epochs_per_stage):
            stage_name_learn = f"P2-S{p_idx+1}-L-Ep{drift_ep+1}"
            acc_learn, off_learn, update_count = run_inference_engine(curr_loader, model, device, socket_push, socket_sub,
                                                              selection_config, args.conf_threshold,
                                                              enable_learning=True, stage_name=stage_name_learn)
            
            # Sync weights from Trainer
            time.sleep(1.0) 
            
            # Force flush message queue for latest weights
            try:
                while True:
                    topic = socket_sub.recv_string(flags=zmq.NOBLOCK)
                    if topic == "weights":
                        state_dict = socket_sub.recv_pyobj()
                        model.load_state_dict(state_dict)
            except zmq.Again:
                pass 
            
            # Save current Stage/Epoch model
            save_name = f"light_stage{p_idx+1}_epoch{drift_ep+1}.pth"
            save_path = os.path.join(ckpt_dir, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"  [Checkpoint] Model saved to {save_path}")

    # ================= PHASE 3: Replay =================
    print("\n" + "="*50)
    print(f">>> PHASE 3: Replay Base Dataset ({args.phase3_epochs} Epochs)")
    print("="*50)

    for epoch in range(args.phase3_epochs):
        stage_str = f"P3-Ep{epoch+1}"
        acc_p3, off_p3, update_count = run_inference_engine(loader_base, model, device, socket_push, socket_sub,
                                                            selection_config, args.conf_threshold,
                                                            enable_learning=True, 
                                                            stage_name=stage_str)
        exp_logs.append({"phase": 3, "epoch": epoch + 1, "desc": "replay_recovery", "acc": acc_p3, "offload": off_p3, "updates": update_count})
        print(f"[{stage_str}] Acc: {acc_p3:.2f}%, Offload: {off_p3:.2f}%")
        
        # Save Replay Phase model
        save_name = f"light_replay_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), os.path.join(ckpt_dir, save_name))

    # --- Save Logs ---
    save_path = f"results/{args.log_file_prefix}_{args.drift_epochs_per_stage}.json"
    with open(save_path, "w") as f: json.dump(exp_logs, f, indent=4)
    print(f"\n[Experiment Finished] Logs saved to {save_path}")
    socket_push.send_pyobj("STOP")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_dir', type=str, required=True)
    parser.add_argument('--drift_data_dir', type=str, required=True)
    parser.add_argument('--light_path', type=str, required=True)
    parser.add_argument('--global_selector', type=str, required=True)
    parser.add_argument('--pretrain_checkpoint_path', type=str, required=True)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--conf_threshold', type=float, default=0.6)
    parser.add_argument('--num_parts', type=int, default=10)
    
    parser.add_argument('--phase1_epochs', type=int, default=3)
    parser.add_argument('--phase3_epochs', type=int, default=3)
    parser.add_argument('--log_file_prefix', type=str, default="experiment_log",
                        help="Prefix for the log file name")
    
    parser.add_argument('--drift_epochs_per_stage', type=int, default=1, 
                        help="Iterations over stage data for trainer catch-up")
    
    args = parser.parse_args()
    main(args)