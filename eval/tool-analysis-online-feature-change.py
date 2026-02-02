import torch
import sys
import os

# 引入你的模型定义
sys.path.append('..')
from models.light_model import LightTrafficClassifier

def check_weight_change(ckpt_pre_path, ckpt_post_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载两个 Checkpoint
    # 注意处理 state_dict 的 key (去掉 module. 前缀)
    def load_sd(path):
        ckpt = torch.load(path, map_location=device)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        return {k.replace('module.', ''): v for k, v in sd.items()}

    sd_pre = load_sd(ckpt_pre_path)
    sd_post = load_sd(ckpt_post_path)
    
    print(f"{'Layer Name':<40} | {'Status':<15} | {'Diff (L1 Norm)':<15}")
    print("-" * 80)
    
    for key in sd_pre:
        if key not in sd_post:
            continue
            
        param_pre = sd_pre[key].float()
        param_post = sd_post[key].float()
        
        diff = torch.sum(torch.abs(param_pre - param_post)).item()
        
        status = "CHANGED" if diff > 1e-6 else "FROZEN/SAME"
        print(f"{key:<40} | {status:<15} | {diff:.6f}")

# 填入你的路径
pre_path = '../outputs/light_model_best/aes_128_gcm_global_per_packet_5x3x50_distilled/light_model_global.pth'
post_path = '../system_online_h5_format/test4-embedding_save/checkpoints/aes_128_gcm2chacha20_poly1305_e5_drift_results_fix/light_stage10_epoch5.pth'

check_weight_change(pre_path, post_path)