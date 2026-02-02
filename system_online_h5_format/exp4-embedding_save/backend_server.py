import torch
import zmq
import time
import argparse
import os
import sys
import json
import numpy as np
import traceback

# Add parent directory to sys.path for importing model definitions
sys.path.append('../..')
from models import EncryptedTrafficClassifier
from finetune.finetune import load_config_from_folder

# ================= Configuration Constants =================
BATCH_SIZE = 128      # Threshold for dynamic batching
TIMEOUT = 0.005       # Timeout threshold (5ms)
PORT_RECV = 5555      # Port to receive data from frontend
PORT_SEND_TRAIN = 5556 # Port to send data to Trainer
LOG_FILE = "results_heavy.json"

# ================= Helper Functions =================
def fast_concat(batch_list):
    """Efficiently concatenate multiple batch dictionaries."""
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    return {k: torch.cat([b[k] for b in batch_list], dim=0) for k in keys}

# ================= Main Worker Process =================
def backend_worker(args):
    context = zmq.Context()
    
    # 1. Receive data from frontend (PULL)
    socket_pull = context.socket(zmq.PULL)
    socket_pull.bind(f"tcp://*:{PORT_RECV}")
    
    # 2. Send data to Trainer (PUSH)
    # Using Connect mode, assuming Trainer will bind to this port
    socket_push_train = context.socket(zmq.PUSH)
    socket_push_train.connect(f"tcp://localhost:{PORT_SEND_TRAIN}")
    # Set HWM to prevent memory overflow if Trainer hangs
    socket_push_train.set_hwm(1000)
    
    device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cudnn.benchmark = True
    print(f"[Backend] Listening on {PORT_RECV}, Sending to Trainer on {PORT_SEND_TRAIN} | Device: {device}")

    model_config, _, _ = load_config_from_folder(args.pretrain_checkpoint_path)
    model = EncryptedTrafficClassifier(num_classes=41, **model_config).to(device)
    
    if os.path.exists(args.heavy_path):
        checkpoint = torch.load(args.heavy_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print("[Backend] Heavy Model loaded.")
    
    model.eval()

    # --- Dynamic Batching Buffers ---
    buffer_data = []    # Stores input features (dict)
    buffer_indices = [] # Stores global indices
    buffer_labels = []  # Stores labels (sent to Trainer for Loss calculation)
    
    current_size = 0
    last_process_time = time.time()
    
    results_log = []
    total_processed = 0

    try:
        while True:
            # 1ms timeout polling
            if socket_pull.poll(timeout=1): 
                packet = socket_pull.recv_pyobj()
                
                if packet == "STOP":
                    print("[Backend] Received STOP signal.")
                    break
                
                # packet: {'data': dict, 'indices': tensor, 'labels': tensor}
                buffer_data.append(packet["data"])
                buffer_indices.append(packet["indices"])
                # Optional: store labels if provided by frontend
                if "labels" in packet:
                    buffer_labels.append(packet["labels"])
                
                current_size += packet["indices"].size(0)
            
            # --- Inference Trigger Conditions ---
            time_diff = time.time() - last_process_time
            should_infer = (current_size >= BATCH_SIZE) or (current_size > 0 and time_diff > TIMEOUT)

            if should_infer:
                # A. Batching (CPU)
                full_batch_cpu = fast_concat(buffer_data)
                full_indices_cpu = torch.cat(buffer_indices, dim=0)
                
                has_labels = len(buffer_labels) > 0
                full_labels_cpu = torch.cat(buffer_labels, dim=0) if has_labels else None
                
                # B. Transfer to GPU (Inference)
                full_batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                  for k, v in full_batch_cpu.items()}
                
                # C. Heavy Model Inference
                with torch.no_grad():
                    logits = model(full_batch_gpu)
                    _, preds = logits.max(1)
                if full_labels_cpu is not None:
                    acc = (preds.cpu() == full_labels_cpu).float().mean().item()
                    print(f"[Backend Check] Heavy Model Accuracy on current batch: {acc * 100:.2f}%")
                
                # D. Log Results
                preds_np = preds.cpu().numpy()
                indices_np = full_indices_cpu.numpy()
                
                for i in range(len(preds_np)):
                    results_log.append({
                        "index": int(indices_np[i]),
                        "heavy_pred": int(preds_np[i])
                    })
                
                # E. Send to Trainer for Online Distillation
                # Payload: Original input (CPU), Teacher Logits (CPU), Labels (CPU)
                # This is a non-blocking send (Fire-and-forget)
                try:
                    trainer_packet = {
                        "data": full_batch_cpu,       # Trainer needs raw data for Student Forward
                        "teacher_logits": logits.cpu(), # Distillation target
                        "labels": full_labels_cpu     # Auxiliary Loss
                    }
                    socket_push_train.send_pyobj(trainer_packet, flags=zmq.NOBLOCK)
                except zmq.Again:
                    # If Trainer queue is full, discard the batch to prioritize Backend throughput
                    pass 
                except Exception as e:
                    print(f"[Backend] Warning: Failed to send to trainer: {e}")

                # F. Cleanup
                total_processed += current_size
                if len(results_log) % 1000 < BATCH_SIZE:
                    print(f"[Backend] Processed {total_processed} samples. Sent to Trainer.")

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
        # Save results
        print(f"[Backend] Saving {len(results_log)} results to {LOG_FILE}...")
        with open(LOG_FILE, "w") as f:
            json.dump(results_log, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--heavy_path', type=str, required=True)
    parser.add_argument('--pretrain_checkpoint_path', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    backend_worker(args)