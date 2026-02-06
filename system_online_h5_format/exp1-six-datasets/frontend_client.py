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

sys.path.append('../..')
from models.light_model import LightTrafficClassifier
from models import EncryptedTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, load_config_from_folder

# ================= Configuration Constants =================
BACKEND_ADDR = "tcp://localhost:5555"
TRAINER_BROADCAST_ADDR = "tcp://localhost:5557"

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

# ================= Core Inference Loop =================
def run_epoch(loader, light_model, heavy_model, device, socket_push, socket_sub, selection_config, 
                    conf_threshold, mode, epoch_idx):
    
    # Metrics tracking
    system_correct_count = 0  # Accuracy after hybrid decision
    light_correct_count = 0   # Baseline accuracy of Pure Light model
    total_count = 0
    offload_count = 0
    update_count = 0
    
    light_model.eval()
    if heavy_model:
        heavy_model.eval()

    pbar = tqdm(loader, desc=f"[Epoch {epoch_idx}] Mode: {mode}", leave=True)

    for batch in pbar:
        # --- 1. System Mode: Check for model weight updates from Trainer ---
        if mode == "system":
            try:
                while True: 
                    topic = socket_sub.recv_string(flags=zmq.NOBLOCK)
                    if topic == "weights":
                        state_dict = socket_sub.recv_pyobj()
                        light_model.load_state_dict(state_dict)
                        update_count += 1
            except zmq.Again:
                pass

        # --- 2. Data Preparation ---
        labels = batch['label'].to(device)
        inputs_cpu = {k: v for k, v in batch.items() if k != 'label'}
        inputs_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs_cpu.items()}
        
        # --- 3. Light Model Inference ---
        with torch.no_grad():
            light_inputs = apply_tensor_selection(inputs_gpu, selection_config)
            logits = light_model(light_inputs)
            probs = F.softmax(logits, dim=1)
            conf, light_preds = probs.max(1)

        # --- 4. Decision & Offloading Logic ---
        
        # Determine samples that require offloading based on confidence
        if mode == "system":
            low_conf_mask = conf < conf_threshold 
        else:
            low_conf_mask = torch.zeros_like(conf, dtype=torch.bool)

        num_offload = low_conf_mask.sum().item()
        offload_count += num_offload

        final_preds = light_preds.clone() 

        # If samples need offloading, calculate heavy results locally to simulate Oracle
        if num_offload > 0 and heavy_model is not None:
            with torch.no_grad():
                heavy_inputs = {k: v[low_conf_mask] for k, v in inputs_gpu.items()}
                heavy_logits = heavy_model(heavy_inputs)
                _, heavy_preds = heavy_logits.max(1)
                
                # Override light prediction with heavy prediction
                final_preds[low_conf_mask] = heavy_preds

        # Statistics
        system_correct_count += (final_preds == labels).sum().item()
        light_correct_count += (light_preds == labels).sum().item()
        total_count += labels.size(0)

        # --- 5. System Mode: Send offloaded data to Backend for distillation ---
        if mode == "system" and num_offload > 0:
            mask_tensor_cpu = low_conf_mask.cpu()
            labels_cpu = labels.cpu()
            try:
                socket_push.send_pyobj({
                    "data": {k: v[mask_tensor_cpu] for k, v in inputs_cpu.items()},
                    "indices": torch.zeros(num_offload), 
                    "labels": labels_cpu[mask_tensor_cpu]
                })
            except Exception as e:
                print(f"ZMQ Send Error: {e}")

        # --- 6. Real-time Progress Monitoring ---
        sys_acc = (system_correct_count / total_count) * 100
        off_rate = (offload_count / total_count) * 100
        
        pbar.set_postfix({
            "SysAcc": f"{sys_acc:.2f}%", 
            "Off": f"{off_rate:.2f}%", 
            "Upds": update_count
        })

    final_sys_acc = (system_correct_count / total_count) * 100
    final_light_acc = (light_correct_count / total_count) * 100
    final_off = (offload_count / total_count) * 100
    
    return {
        "epoch": epoch_idx,
        "acc": final_sys_acc,       
        "light_acc": final_light_acc, 
        "offload": final_off,
        "updates": update_count
    }

# ================= Main Execution =================
def main(args):
    socket_push = None
    socket_sub = None
    
    if args.mode == "system":
        context = zmq.Context()
        print("[Frontend] Connecting to Backend...")
        socket_push = context.socket(zmq.PUSH)
        socket_push.connect(BACKEND_ADDR)
        
        print("[Frontend] Connecting to Trainer Broadcast...")
        socket_sub = context.socket(zmq.SUB)
        socket_sub.connect(TRAINER_BROADCAST_ADDR)
        socket_sub.setsockopt_string(zmq.SUBSCRIBE, "weights")
        time.sleep(0.5) 
        # Clear existing weight buffer
        try:
            while True:
                socket_sub.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    device = torch.device(f'cuda:{args.gpu}')
    
    # --- Load Configuration ---
    map_path = os.path.join(args.data_dir, "label_mapping.json")
    with open(map_path, 'r') as f: num_classes = len(json.load(f))
    
    try:
        model_config, heavy_max_bytes, heavy_max_packets = load_config_from_folder(args.pretrain_checkpoint_path)
    except:
        model_config = {} 
        heavy_max_bytes, heavy_max_packets = 1500, 100

    selection_config = None
    if args.global_selector:
        selection_config = json.loads(str(np.load(args.global_selector, allow_pickle=True)["config"]))

    # --- Load Dataset ---
    print(f"\n[Setup] Loading dataset from {args.data_dir}...")
    data_path = os.path.join(args.data_dir, "test_data.h5")
    dataset = TrafficDataset(data_path, augmentation=False, max_packets=heavy_max_packets, max_bytes=heavy_max_bytes)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # --- Initialize Light Model ---
    print(f"[Setup] Loading Light Model from {args.light_path}...")
    light_model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)
    if os.path.exists(args.light_path):
        checkpoint = torch.load(args.light_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        light_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    # --- Initialize Heavy Model (Local Oracle for accuracy benchmarking) ---
    heavy_model = None
    if args.mode == "system" and args.heavy_path:
        print(f"[Setup] Loading Heavy Model (Oracle) from {args.heavy_path}...")
        heavy_model = EncryptedTrafficClassifier(num_classes=num_classes, **model_config).to(device)
        
        if os.path.exists(args.heavy_path):
            checkpoint = torch.load(args.heavy_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            heavy_model.load_state_dict(new_state_dict, strict=True)
            heavy_model.eval()
        else:
            print("Warning: Heavy model path provided but file not found!")

    logs = []
    
    print("\n" + "="*60)
    print(f"🚀 STARTING EXPERIMENT | Mode: {args.mode} | Dataset: {args.data_dir}")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        res = run_epoch(loader, light_model, heavy_model, device, socket_push, socket_sub, selection_config, 
                        args.conf_threshold, args.mode, epoch)
        logs.append(res)
        print(f"   >>> Epoch {epoch} Result: SysAcc={res['acc']:.2f}% (LightAcc={res['light_acc']:.2f}%), Offload={res['offload']:.2f}%, Updates={res['updates']}")

    save_path = f"{args.log_file}.json"
    with open(save_path, "w") as f: json.dump(logs, f, indent=4)
    print(f"\n[Finished] Logs saved to {save_path}")

    if args.mode == "system":
        socket_push.send_pyobj("STOP")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--light_path', type=str, required=True)
    parser.add_argument('--global_selector', type=str, required=True)
    parser.add_argument('--pretrain_checkpoint_path', type=str, required=True)
    parser.add_argument('--heavy_path', type=str, default=None, help="Path to heavy model for local oracle accuracy calculation")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--conf_threshold', type=float, default=0.6)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mode', type=str, choices=['pure', 'system'], required=True)
    parser.add_argument('--log_file', type=str, default="result_log")
    args = parser.parse_args()
    main(args)