import subprocess
import time
import os
import signal
import sys

# ================= 1. Basic Configuration =================

PATH_PREFIX = "../../" 

# Common parameters
PRETRAIN_CHECKPOINT = os.path.join(PATH_PREFIX, "outputs/pretrain_checkpoints/10*300/10x300_pretrain_epoch_7.pth")
GPU_FRONTEND = "0"
GPU_BACKEND = "1"
GPU_TRAINER = "2"
EPOCHS = "1" # Number of epochs per experiment

# Hyperparameters: Confidence thresholds to be tested
CONF_THRESHOLDS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# List of datasets for evaluation
DATASETS = ["aes_128_gcm", "aes_256_gcm", "chacha20_poly1305", "mix", "cstnet", "datacon2021_part1"]

# ================= 2. Path Mapping =================

def get_dataset_config(dataset_name):
    """Returns a dictionary of file paths corresponding to the dataset name."""
    data_dir = os.path.join(PATH_PREFIX, f"1-data_preprocess/{dataset_name}_h5")
    
    heavy_path = os.path.join(PATH_PREFIX, f"outputs/finetuned_best/{dataset_name}/finetuned_best_{dataset_name}_swa.pth")
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

def run_pure_experiment(cfg, output_dir):
    """Runs Pure Light Model test (Baseline, unaffected by threshold)."""
    print(f"\n>>> [Pure Light] Testing {cfg['name']}...")
    
    # Log path: output_dir/pure_{dataset}.json
    log_file = os.path.join(output_dir, f"pure_{cfg['name']}")
    
    cmd = [
        "python", "frontend_client.py",
        "--data_dir", cfg['data_dir'],
        "--light_path", cfg['light_path'],
        "--global_selector", cfg['selector_path'],
        "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
        "--conf_threshold", "0.0", # Threshold is arbitrary in Pure mode
        "--gpu", GPU_FRONTEND,
        "--mode", "pure",
        "--epochs", "1",
        "--log_file", log_file
    ]
    
    p = subprocess.Popen(cmd)
    p.wait()
    print(f">>> Pure Test for {cfg['name']} Finished.")

def run_system_experiment(cfg, threshold, output_dir):
    """Runs Full System (Light + Heavy + Trainer) evaluation."""
    print(f"\n>>> [System Mode] Testing {cfg['name']} with Conf={threshold}...")
    
    log_file_frontend = os.path.join(output_dir, f"system_{cfg['name']}")
    log_file_backend = os.path.join(output_dir, f"backend_heavy_log_{cfg['name']}.json")
    
    processes = []
    try:
        # 1. Start Backend
        cmd_backend = [
            "python", "backend_server.py",
            "--heavy_path", cfg['heavy_path'],
            "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
            "--gpu", GPU_BACKEND,
            "--data_dir", cfg['data_dir'],
            "--log_file", log_file_backend
        ]
        processes.append(run_process(cmd_backend, "Backend"))

        # 2. Start Trainer
        cmd_trainer = [
            "python", "online_trainer.py",
            "--data_dir", cfg['data_dir'],
            "--light_path", cfg['light_path'],
            "--global_selector", cfg['selector_path'],
            "--gpu", GPU_TRAINER
        ]
        processes.append(run_process(cmd_trainer, "Trainer"))

        # 3. Wait for backend initialization
        time.sleep(8)

        # 4. Start Frontend
        cmd_frontend = [
            "python", "frontend_client.py",
            "--data_dir", cfg['data_dir'],
            "--light_path", cfg['light_path'],
            "--heavy_path", cfg['heavy_path'],
            "--global_selector", cfg['selector_path'],
            "--pretrain_checkpoint_path", PRETRAIN_CHECKPOINT,
            "--conf_threshold", str(threshold),
            "--gpu", GPU_FRONTEND,
            "--mode", "system",
            "--epochs", EPOCHS,
            "--log_file", log_file_frontend
        ]
        p_frontend = subprocess.Popen(cmd_frontend)
        processes.append(p_frontend)
        
        exit_code = p_frontend.wait()
        if exit_code == 0:
            print(">>> System Test Finished Successfully.")
        else:
            print(">>> System Test Failed.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_processes(processes)
        # Delay for port release to avoid "Address already in use"
        time.sleep(5) 

# ================= 5. Main Entry =================

if __name__ == "__main__":
    base_result_dir = "results"
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)

    print(f"Datasets to test: {DATASETS}")
    print(f"Thresholds to test: {CONF_THRESHOLDS}")
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Outer loop: Iterate through Hyperparameter (Threshold)
    for threshold in CONF_THRESHOLDS:
        print(f"\n{'='*80}")
        print(f"### STARTING BATCH FOR THRESHOLD: {threshold} ###")
        print(f"{'='*80}")
        
        # Create independent folder for current threshold
        current_save_dir = os.path.join(base_result_dir, f"conf_{threshold}")
        os.makedirs(current_save_dir, exist_ok=True)
        
        # Inner loop: Iterate through Datasets
        for ds_name in DATASETS:
            cfg = get_dataset_config(ds_name)
            print(f"\n--- Processing {ds_name} [Conf: {threshold}] ---")
            
            # Execute System evaluation
            run_system_experiment(cfg, threshold, current_save_dir)
            
    print("\n✅ All experiments for all thresholds finished.")