"""
File: 52-final-speed/eval/vis_logits.py
方案 A: 可视化 Logits (Softmax概率) 分布
展示模型最终输出的置信度分布变化
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import json
import os
import sys
from tqdm import tqdm

# 确保能导入上级目录的模块
sys.path.append('..')
from models.light_model import LightTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed

# ================= 配置区域 =================
ARGS = {
    'data_dir': '../1-data_preprocess/chacha20_poly1305_h5',
    'ckpt_pre': '../outputs/light_model_best/aes_128_gcm_global_per_packet_5x3x50_distilled/light_model_global.pth',
    'ckpt_post': '../system_online_h5_format/test4-embedding_save/checkpoints/aes_128_gcm2chacha20_poly1305_e5_drift_results_fix/light_stage10_epoch5.pth',
    'global_selector': '../outputs/global_selectors/aes_128_gcm_per_packet_5x3x50.npz',
    'gpu': 0,
    'batch_size': 128,
    'tsne_samples': 1000,
    # 这里填你之前分析出的提升最大的几个类别 ID
    'target_classes': [11, 19, 8]  # google-analytics, hubspot, garmin
}
# ===========================================

def get_logits_data(model, loader, device):
    """只获取 Softmax 后的概率分布"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['label'].to(device)
            
            # 不需要 return_feats，只要 logits
            logits = model(batch_data, return_feats=False)
            probs = F.softmax(logits, dim=1) # 关键：转换到概率空间
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_labels), np.concatenate(all_probs)

def plot_logits_tsne(labels_pre, probs_pre, labels_post, probs_post, id_to_name):
    # 1. 筛选目标类别
    mask_pre = np.isin(labels_pre, ARGS['target_classes'])
    mask_post = np.isin(labels_post, ARGS['target_classes'])
    
    # 2. 采样函数
    def sample(l, p, n=ARGS['tsne_samples']):
        if len(l) > n:
            idx = np.random.choice(len(l), n, replace=False)
            return l[idx], p[idx]
        return l, p

    l_pre, p_pre = sample(labels_pre[mask_pre], probs_pre[mask_pre])
    l_post, p_post = sample(labels_post[mask_post], probs_post[mask_post])
    
    print(f"Running t-SNE on Probabilities (Pre={len(l_pre)}, Post={len(l_post)})...")
    
    # 3. 合并做 t-SNE (保证空间一致)
    combined_probs = np.vstack([p_pre, p_post])
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    combined_2d = tsne.fit_transform(combined_probs)
    
    tsne_pre = combined_2d[:len(l_pre)]
    tsne_post = combined_2d[len(l_pre):]
    
    # 4. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    unique_ids = sorted(ARGS['target_classes'])
    palette = sns.color_palette("bright", len(unique_ids))
    color_map = {uid: palette[i] for i, uid in enumerate(unique_ids)}
    
    # 子图循环
    datas = [tsne_pre, tsne_post]
    lbls_list = [l_pre, l_post]
    titles = ["Before Finetune (Output Probability)", "After Finetune (Output Probability)"]
    
    for i in range(2):
        ax = axes[i]
        curr_data = datas[i]
        curr_lbls = lbls_list[i]
        
        for uid in unique_ids:
            idx = curr_lbls == uid
            ax.scatter(curr_data[idx, 0], curr_data[idx, 1], 
                       label=id_to_name[str(uid)], s=25, alpha=0.7, c=[color_map[uid]])
            
        ax.set_title(titles[i])
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('vis_logits_tsne.png', dpi=300)
    print("✓ Visualization saved to vis_logits_tsne.png")
    plt.show()

def load_ckpt(model, path, device):
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # 处理 module. 前缀
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd)
    return model

def main():
    device = torch.device(f"cuda:{ARGS['gpu']}")
    set_seed(42)
    
    # 加载 Label Map
    with open(os.path.join(ARGS['data_dir'], "label_mapping.json"), 'r') as f:
        label_map = json.load(f)
        id_to_name = {str(v): k for k, v in label_map.items()}
        num_classes = len(label_map)

    # 数据集
    dataset = TrafficDataset(
        os.path.join(ARGS['data_dir'], "test_data.h5"), 
        augmentation=False, 
        max_packets=10, 
        max_bytes=300, 
        global_selection_path=ARGS['global_selector']
    )
    loader = DataLoader(dataset, batch_size=ARGS['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # 模型
    model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)

    # 1. Pre
    load_ckpt(model, ARGS['ckpt_pre'], device)
    l_pre, p_pre = get_logits_data(model, loader, device)

    # 2. Post
    load_ckpt(model, ARGS['ckpt_post'], device)
    l_post, p_post = get_logits_data(model, loader, device)

    # 绘图
    plot_logits_tsne(l_pre, p_pre, l_post, p_post, id_to_name)

if __name__ == '__main__':
    main()