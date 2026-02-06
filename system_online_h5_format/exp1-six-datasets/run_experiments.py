import subprocess
import time
import os
import signal
import sys

# ================= 1. Basic Configuration =================

PATH_PREFIX = "../../" 

# Common Parameters
PRETRAIN_CHECKPOINT = os.path.join(PATH_PREFIX, "outputs/pretrain_checkpoints/10*300/10x300_pretrain_epoch_7.pth")
CONF_THRESHOLD = "0.8"
GPU_FRONTEND = "0"
GPU_BACKEND = "1"
GPU_TRAINER = "2"
EPOCHS = "1" # Number of epochs per experiment

# Datasets to be evaluated
DATASETS = ["aes_128_gcm", "aes_256_gcm", "chacha20_poly1305", "mix", "cstnet", "datacon2021_part1"]

# ================= 2. Path Mapping Table =================

def get_dataset_config(dataset_name):
    """Returns dataset configuration and model paths based on dataset name"""
    data_dir = os.path.join(PATH_PREFIX, f"1-data_preprocess/{dataset_name}_h5")
    
    # Heavy Model: Fine-tuned on the specific dataset
    heavy_path = os.path.join(PATH_PREFIX, f"outputs/finetuned_best/{dataset_name}/finetuned_best_{dataset_name}_swa.pth")
    
    # Light Model and Selector paths
    light_path = os.path.join(PATH_PREFIX, f"outputs/light_model_best/{dataset_name}_global_per_packet_5x3x50_distilled/light_model_global.pth")
    selector_path = os.path.join(PATH_PREFIX, f"outputs/global_selectors/{dataset_name}_per_packet_5x3x50.npz")
    
    if dataset_name == "datacon2021_part1":
        light_path = os.path.join(PATH_PREFIX, f"outputs/light_model_best/{dataset_name}_global_contiguous_5x3x50_distilled/light_model_global.pth")
        selector_path = os.path.join(PATH_PREFIX, f"outputs/global_selectors/{dataset_name}_contiguous_5x3x50.npz")

    return {
        "name": dataset_name,
        "data_dir": data_dir,
        "heavy_path": heavy_path,
        "light_path": light_path,
        "selector_path": selector_path
    }

# ================= 3. Process Management =================

def run_process(cmd, description):
    print(f"    [Starting] {description}...")
    return subprocess.Popen(cmd)

def cleanup_processes(proc_list):
    for p in proc_list:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=2)
            except:
                p.kill()

# ================= 4. Experiment Logic =================

def run_pure_experiment(cfg):
    """Runs tests for Pure Light Model mode"""
    print(f"\n>>> [Pure Light] Testing {cfg['name']}...")
    log_file = f"results/pure_{cfg['name']}"
    
    cmd = [
        "python", "frontend_client.py",
        "--data_dir", cfg['data_dir'],
        "--light_path", cfg['light_path'],
        "--global_selector", cfg['selector_path'],
        "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
        "--conf_threshold", CONF_THRESHOLD,
        "--gpu", GPU_FRONTEND,
        "--mode", "pure",
        "--epochs", "1", # Single epoch is sufficient for non-updating Pure mode
        "--log_file", log_file
    ]
    
    p = subprocess.Popen(cmd)
    p.wait()
    print(f">>> Pure Test for {cfg['name']} Finished.")

def run_system_experiment(cfg):
    """Runs tests for full System mode (Light + Heavy + Trainer) with Online Learning"""
    print(f"\n>>> [System Mode] Testing {cfg['name']} (Online Learning)...")
    log_file = f"results/system_{cfg['name']}"
    
    processes = []
    try:
        # 1. Start Backend Server
        cmd_backend = [
            "python", "backend_server.py",
            "--heavy_path", cfg['heavy_path'],
            "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
            "--gpu", GPU_BACKEND,
            "--data_dir", cfg['data_dir']
        ]
        processes.append(run_process(cmd_backend, "Backend"))

        # 2. Start Online Trainer
        cmd_trainer = [
            "python", "online_trainer.py",
            "--data_dir", cfg['data_dir'],
            "--light_path", cfg['light_path'],
            "--global_selector", cfg['selector_path'],
            "--gpu", GPU_TRAINER
        ]
        processes.append(run_process(cmd_trainer, "Trainer"))

        # 3. Warm-up wait
        time.sleep(8)

        # 4. Start Frontend Client
        cmd_frontend = [
            "python", "frontend_client.py",
            "--data_dir", cfg['data_dir'],
            "--light_path", cfg['light_path'],
            "--heavy_path", cfg['heavy_path'], 
            "--global_selector", cfg['selector_path'],
            "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
            "--conf_threshold", CONF_THRESHOLD,
            "--gpu", GPU_FRONTEND,
            "--mode", "system",
            "--epochs", EPOCHS,
            "--log_file", log_file
        ]
        p_frontend = subprocess.Popen(cmd_frontend)
        processes.append(p_frontend)
        
        exit_code = p_frontend.wait()
        if exit_code == 0:
            print(">>> System Test Finished Successfully.")
        else:
            print(">>> System Test Failed.")

    except Exception as e:
        print(f"Error during system experiment: {e}")
    finally:
        cleanup_processes(processes)
        time.sleep(3)

# ================= 5. Main Entry =================

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    print(f"Datasets to test: {DATASETS}")
    print("Starting in 3 seconds...")
    time.sleep(3)

    for ds_name in DATASETS:
        cfg = get_dataset_config(ds_name)
        
        print(f"\n{'#'*60}")
        print(f"### DATASET: {ds_name} ###")
        print(f"{'#'*60}")
        
        # Step 1: Run Baseline (Pure Light)
        run_pure_experiment(cfg)
        
        time.sleep(2)
        
        # Step 2: Run Full System (Online Distillation)
        run_system_experiment(cfg)
        
    print("\n✅ All experiments finished.")