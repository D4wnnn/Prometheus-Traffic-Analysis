"""
File: step1_extract_data.py
功能：模型推理 -> t-SNE降维 -> 保存数据为 .npz
"""
import torch
import numpy as np
from sklearn.manifold import TSNE
import json
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入你的项目依赖
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
    'tsne_samples': 800,   # 采样数
    'target_classes': [11, 19, 8], # 目标类别
    'save_path': 'vis_boundary_data.npz' # 保存路径
}
# ===========================================

def get_emb_and_pred(model, loader, device):
    """获取 Embedding 和 预测结果"""
    model.eval()
    embeddings, predictions, true_labels = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['label'].to(device)
            logits, feats = model(batch_data, return_feats=True)
            preds = logits.argmax(dim=1)
            
            embeddings.append(feats.cpu().numpy())
            predictions.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            
    return np.concatenate(embeddings), np.concatenate(predictions), np.concatenate(true_labels)

def process_and_save(emb_pre, pred_pre, label_pre, 
                     emb_post, pred_post, label_post, 
                     id_to_name):
    
    print("Processing data for visualization...")
    
    # 1. 筛选数据
    mask_pre = np.isin(label_pre, ARGS['target_classes'])
    mask_post = np.isin(label_post, ARGS['target_classes'])
    
    # 2. 采样函数
    def sample(e, p, l, n=ARGS['tsne_samples']):
        if len(e) > n:
            idx = np.random.choice(len(e), n, replace=False)
            return e[idx], p[idx], l[idx]
        return e, p, l

    e_pre, p_pre, l_pre = sample(emb_pre[mask_pre], pred_pre[mask_pre], label_pre[mask_pre])
    e_post, p_post, l_post = sample(emb_post[mask_post], pred_post[mask_post], label_post[mask_post])
    
    # 3. 运行 t-SNE (这是最耗时的步骤，必须在这里完成)
    print(f"Running t-SNE on {len(e_pre) + len(e_post)} samples...")
    combined_emb = np.vstack([e_pre, e_post])
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    combined_2d = tsne.fit_transform(combined_emb)
    
    x_pre = combined_2d[:len(e_pre)]
    x_post = combined_2d[len(e_pre):]
    
    # 4. 保存所有绘图所需数据
    print(f"Saving data to {ARGS['save_path']}...")
    np.savez(
        ARGS['save_path'],
        # Pre data
        x_pre=x_pre, p_pre=p_pre, l_pre=l_pre,
        # Post data
        x_post=x_post, p_post=p_post, l_post=l_post,
        # Metadata
        target_classes=np.array(ARGS['target_classes']),
        id_to_name=id_to_name # 字典会被自动pickle
    )
    print("Done! You can now run the plotting script.")

def load_ckpt(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd)

def main():
    device = torch.device(f"cuda:{ARGS['gpu']}")
    set_seed(42)
    
    # 加载标签映射
    with open(os.path.join(ARGS['data_dir'], "label_mapping.json"), 'r') as f:
        label_map = json.load(f)
        id_to_name = {int(v): k for k, v in label_map.items()} # 确保key是int
        num_classes = len(label_map)

    # 加载数据
    dataset = TrafficDataset(
        os.path.join(ARGS['data_dir'], "test_data.h5"), 
        augmentation=False, max_packets=10, max_bytes=300, 
        global_selection_path=ARGS['global_selector']
    )
    loader = DataLoader(dataset, batch_size=ARGS['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
    model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)

    # 推理 Pre
    print(">>> Inferencing Pre-Model...")
    load_ckpt(model, ARGS['ckpt_pre'], device)
    e_pre, p_pre, l_pre = get_emb_and_pred(model, loader, device)

    # 推理 Post
    print(">>> Inferencing Post-Model...")
    load_ckpt(model, ARGS['ckpt_post'], device)
    e_post, p_post, l_post = get_emb_and_pred(model, loader, device)

    # 处理并保存
    process_and_save(e_pre, p_pre, l_pre, e_post, p_post, l_post, id_to_name)

if __name__ == '__main__':
    main()