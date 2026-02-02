import subprocess
import time
import os
import signal
import sys
import itertools

# ================= 1. Base Configuration =================

PATH_PREFIX = "../../" 

# Common Parameters
PRETRAIN_CHECKPOINT = os.path.join(PATH_PREFIX, "outputs/pretrain_checkpoints/10*300/10x300_pretrain_epoch_7.pth")
CONF_THRESHOLD = "0.5"
NUM_PARTS = "10"
GPU_FRONTEND = "0"
GPU_BACKEND = "1"
GPU_TRAINER = "2"

# Experimental Variables
# Define epochs to run: 1, then 5
DRIFT_EPOCHS_LIST = [1, 5]

# Dataset names for generating permutations
DATASETS = ["aes_128_gcm", "chacha20_poly1305", "mix"]

# ================= 2. Path Mapping =================

def get_dataset_paths(dataset_name):
    """Returns corresponding file paths for a given dataset name"""
    data_dir = os.path.join(PATH_PREFIX, f"1-data_preprocess/{dataset_name}_h5")
    
    heavy_path = os.path.join(PATH_PREFIX, f"outputs/finetuned_best/{dataset_name}/finetuned_best_{dataset_name}_swa.pth")
    light_path = os.path.join(PATH_PREFIX, f"outputs/light_model_best/{dataset_name}_global_per_packet_5x3x50_distilled/light_model_global.pth")
    selector_path = os.path.join(PATH_PREFIX, f"outputs/global_selectors/{dataset_name}_per_packet_5x3x50.npz")

    return {
        "name": dataset_name,
        "data_dir": data_dir,
        "heavy_path": heavy_path,
        "light_path": light_path,
        "selector_path": selector_path
    }

# ================= 3. Core Execution Logic =================

def run_process(cmd, description):
    """Start a background process"""
    print(f"    [Starting] {description}...")
    return subprocess.Popen(cmd)

def run_experiment_group(source_cfg, target_cfg, current_drift_epochs):
    """
    Run an experiment group: Source -> Target
    current_drift_epochs: Number of drift epochs for this run
    """
    
    exp_name = f"{source_cfg['name']}2{target_cfg['name']}"
    log_suffix = f"e{current_drift_epochs}" 
    
    print(f"\n{'='*60}")
    print(f" 🚀  Experiment: {source_cfg['name']} -> {target_cfg['name']} | Drift Epochs: {current_drift_epochs}")
    print(f"{'='*60}")

    # --- Phase 2: System Fix (Heavy + Trainer + Frontend) ---
    print(f"\n>>> [Phase 2] System Fix (Online Learning) | Epochs: {current_drift_epochs}")
    log_file_fix = f"{exp_name}_{log_suffix}_drift_results_fix"
    
    processes = []
    try:
        # 1. Start Heavy Backend
        cmd_heavy = [
            "python", "backend_server.py",
            "--heavy_path", source_cfg['heavy_path'],
            "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
            "--gpu", GPU_BACKEND
        ]
        processes.append(run_process(cmd_heavy, "Backend (Heavy)"))

        # 2. Start Online Trainer
        cmd_trainer = [
            "python", "online_trainer.py",
            "--data_dir", source_cfg['data_dir'], 
            "--light_path", source_cfg['light_path'],
            "--global_selector", source_cfg['selector_path'],
            "--gpu", GPU_TRAINER
        ]
        processes.append(run_process(cmd_trainer, "Online Trainer"))

        # 3. Initialization wait
        print("    Waiting 10 seconds for initialization...")
        time.sleep(10)

        # 4. Start Frontend Client
        cmd_frontend_fix = [
            "python", "frontend_client.py",
            "--base_data_dir", source_cfg['data_dir'],
            "--drift_data_dir", target_cfg['data_dir'],
            "--light_path", source_cfg['light_path'],
            "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
            "--conf_threshold", CONF_THRESHOLD,
            "--global_selector", source_cfg['selector_path'],
            "--num_parts", NUM_PARTS,
            "--gpu", GPU_FRONTEND,
            "--log_file_prefix", log_file_fix,
            "--drift_epochs_per_stage", str(current_drift_epochs)
        ]
        
        p_frontend = subprocess.Popen(cmd_frontend_fix)
        processes.append(p_frontend)
        
        exit_code = p_frontend.wait()
        
        if exit_code == 0:
            print(f">>> Phase 2 Completed Successfully.")
        else:
            print(f">>> Phase 2 Failed with code {exit_code}.")

    except Exception as e:
        print(f"!!! Error: {e}")
    finally:
        cleanup_processes(processes)
        print(f"--- Cleanup done ---\n")
        time.sleep(5)

def cleanup_processes(proc_list):
    """Cleanup all processes in the list"""
    for p in proc_list:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
            except Exception:
                pass

# ================= 4. Main Entry Point =================

if __name__ == "__main__":
    permutations = list(itertools.permutations(DATASETS, 2))
    total_runs = len(DRIFT_EPOCHS_LIST) * len(permutations)
    
    print(f"Plan to run {total_runs} experiment groups.")
    print(f"Drift Epochs Schedule: {DRIFT_EPOCHS_LIST}")
    print(f"Dataset Permutations: {len(permutations)}")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    # Outer loop: Run all epoch=1 experiments, then all epoch=5
    for epoch_val in DRIFT_EPOCHS_LIST:
        print(f"\n######################################################")
        print(f"### STARTING BATCH WITH DRIFT_EPOCHS = {epoch_val} ###")
        print(f"######################################################\n")
        
        for src_name, dst_name in permutations:
            src_cfg = get_dataset_paths(src_name)
            dst_cfg = get_dataset_paths(dst_name)
            
            run_experiment_group(src_cfg, dst_cfg, epoch_val)
        
    print("\n✅ All batches finished! Go check your papers.")