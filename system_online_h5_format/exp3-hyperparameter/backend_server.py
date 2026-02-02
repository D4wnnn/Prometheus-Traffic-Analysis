import torch
import zmq
import time
import argparse
import os
import sys
import json
import numpy as np
import traceback

# Add parent directory to import model definitions
sys.path.append('../..')
from models import EncryptedTrafficClassifier
from finetune.finetune import load_config_from_folder

# ================= Configuration Constants =================
BATCH_SIZE = 128      # Threshold for dynamic batching
TIMEOUT = 0.005        # Timeout threshold (5ms)
PORT_RECV = 5555      # Port for receiving frontend data
PORT_SEND_TRAIN = 5556 # Port for sending data to Trainer

# ================= Utility Functions =================
def fast_concat(batch_list):
    """Fast concatenation of multiple batch dictionaries."""
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    return {k: torch.cat([b[k] for b in batch_list], dim=0) for k in keys}

# ================= Main Worker Process =================
def backend_worker(args):
    context = zmq.Context()
    
    # 1. Receive frontend data (PULL)
    socket_pull = context.socket(zmq.PULL)
    socket_pull.bind(f"tcp://*:{PORT_RECV}")
    
    # 2. Send to Trainer (PUSH)
    socket_push_train = context.socket(zmq.PUSH)
    socket_push_train.connect(f"tcp://localhost:{PORT_SEND_TRAIN}")
    socket_push_train.set_hwm(1000)
    
    device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cudnn.benchmark = True
    print(f"[Backend] Listening on {PORT_RECV}, Sending to Trainer on {PORT_SEND_TRAIN} | Device: {device}")

    # --- Load Model ---
    map_path = os.path.join(args.data_dir, "label_mapping.json")
    if not os.path.exists(map_path):
        print(f"Error: {map_path} not found.")
        return
    with open(map_path, 'r') as f:
        num_classes = len(json.load(f))
        
    model_config, _, _ = load_config_from_folder(args.pretrain_checkpoint_path)
    model = EncryptedTrafficClassifier(num_classes=num_classes, **model_config).to(device)
    
    if os.path.exists(args.heavy_path):
        checkpoint = torch.load(args.heavy_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print("[Backend] Heavy Model loaded.")
    
    model.eval()

    # --- Dynamic Batching Buffer ---
    buffer_data = []    
    buffer_indices = [] 
    buffer_labels = []  
    
    current_size = 0
    last_process_time = time.time()
    
    results_log = []
    total_processed = 0

    try:
        while True:
            # Polling with 1ms timeout
            if socket_pull.poll(timeout=1): 
                packet = socket_pull.recv_pyobj()
                
                if packet == "STOP":
                    print("[Backend] Received STOP signal.")
                    break
                
                buffer_data.append(packet["data"])
                buffer_indices.append(packet["indices"])
                if "labels" in packet:
                    buffer_labels.append(packet["labels"])
                
                current_size += packet["indices"].size(0)
            
            # --- Inference Trigger Conditions ---
            time_diff = time.time() - last_process_time
            should_infer = (current_size >= BATCH_SIZE) or (current_size > 0 and time_diff > TIMEOUT)

            if should_infer:
                # A. Batch Concatenation (CPU)
                full_batch_cpu = fast_concat(buffer_data)
                full_indices_cpu = torch.cat(buffer_indices, dim=0)
                
                has_labels = len(buffer_labels) > 0
                full_labels_cpu = torch.cat(buffer_labels, dim=0) if has_labels else None
                
                # B. Transfer to GPU (for inference)
                full_batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                  for k, v in full_batch_cpu.items()}
                
                # C. Heavy Model Inference
                with torch.no_grad():
                    logits = model(full_batch_gpu)
                    _, preds = logits.max(1)
                if full_labels_cpu is not None:
                    acc = (preds.cpu() == full_labels_cpu).float().mean().item()
                    print(f"[Backend Check] Heavy Model Accuracy on current batch: {acc * 100:.2f}%")
                
                # D. Logging Results
                preds_np = preds.cpu().numpy()
                indices_np = full_indices_cpu.numpy()
                
                for i in range(len(preds_np)):
                    results_log.append({
                        "index": int(indices_np[i]),
                        "heavy_pred": int(preds_np[i])
                    })
                
                # E. Send to Trainer for online distillation
                try:
                    trainer_packet = {
                        "data": full_batch_cpu,       
                        "teacher_logits": logits.cpu(), 
                        "labels": full_labels_cpu     
                    }
                    socket_push_train.send_pyobj(trainer_packet, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass 
                except Exception as e:
                    print(f"[Backend] Warning: Failed to send to trainer: {e}")

                # F. Cleanup and Monitoring
                total_processed += current_size
                if len(results_log) % 1000 < BATCH_SIZE:
                    print(f"[Backend] Processed {total_processed} samples.")

                buffer_data = []
                buffer_indices = []
                buffer_labels = []
                current_size = 0
                last_process_time = time.time()

    except KeyboardInterrupt:
        print("\n[Backend] Interrupted.")
    except Exception as e:
        print(f"[Backend] CRITICAL ERROR: {e}")
        traceback.print_exc()
    finally:
        # Save results to the specified log file
        print(f"[Backend] Saving {len(results_log)} results to {args.log_file}...")
        with open(args.log_file, "w") as f:
            json.dump(results_log, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--heavy_path', type=str, required=True)
    parser.add_argument('--pretrain_checkpoint_path', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--log_file', type=str, default="results_heavy.json", help="Path to save heavy model prediction logs")
    args = parser.parse_args()
    backend_worker(args)