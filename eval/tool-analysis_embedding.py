import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
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
    'data_dir': '../1-data_preprocess/chacha20_poly1305_h5',        # 数据集路径
    'ckpt_pre': '../outputs/light_model_best/aes_128_gcm_global_per_packet_5x3x50_distilled/light_model_global.pth',   # 在线微调前的权重路径
    'ckpt_post': '../system_online_h5_format/test4-embedding_save/checkpoints/aes_128_gcm2chacha20_poly1305_e5_drift_results_fix/light_stage10_epoch5.pth', # 在线微调后的权重路径
    'global_selector': '../outputs/global_selectors/aes_128_gcm_per_packet_5x3x50.npz', # 全局 Selector 路径
    'gpu': 0,
    'batch_size': 128,
    'tsne_samples': 1000, # 每个类别用于画图的最大样本数
    'top_k': 3            # 可视化提升最大的前K个类别
}
# ===========================================

def get_inference_data(model, loader, device):
    """推理并获取 Logits, Labels 和 Embeddings"""
    model.eval()
    all_preds = []
    all_labels = []
    all_embs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing"):
            # 移至 GPU
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            labels = batch['label'].to(device)
            
            # 调用修改后的 forward，获取特征
            logits, feats = model(batch_data, return_feats=True)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_embs.append(feats.cpu().numpy())
            
    return (np.concatenate(all_preds), 
            np.concatenate(all_labels), 
            np.concatenate(all_embs))

def plot_tsne(pre_data, post_data, target_class_ids, id_to_name):
    """绘制对比 t-SNE 图"""
    _, labels_pre, embs_pre = pre_data
    _, labels_post, embs_post = post_data
    
    # 筛选目标类别的样本
    mask_pre = np.isin(labels_pre, target_class_ids)
    mask_post = np.isin(labels_post, target_class_ids)
    
    # 为了保持对比公平性，我们只取两者共有的样本索引（假设数据集顺序未变）
    # 或者简单地分别筛选
    sel_embs_pre = embs_pre[mask_pre]
    sel_labels_pre = labels_pre[mask_pre]
    
    sel_embs_post = embs_post[mask_post]
    sel_labels_post = labels_post[mask_post]

    # 下采样以加快 t-SNE 速度
    if len(sel_embs_pre) > ARGS['tsne_samples']:
        idx = np.random.choice(len(sel_embs_pre), ARGS['tsne_samples'], replace=False)
        sel_embs_pre = sel_embs_pre[idx]
        sel_labels_pre = sel_labels_pre[idx]
        
    if len(sel_embs_post) > ARGS['tsne_samples']:
        idx = np.random.choice(len(sel_embs_post), ARGS['tsne_samples'], replace=False)
        sel_embs_post = sel_embs_post[idx]
        sel_labels_post = sel_labels_post[idx]

    print(f"Running t-SNE (Samples: Pre={len(sel_embs_pre)}, Post={len(sel_embs_post)})...")
    
    # 分别计算 t-SNE (也可以合并计算保持空间一致，但在分布差异巨大时分别计算能看清各自的簇结构)
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    
    # 合并计算
    combined_embs = np.vstack([sel_embs_pre, sel_embs_post])
    combined_2d = tsne.fit_transform(combined_embs)
    
    tsne_pre = combined_2d[:len(sel_embs_pre)]
    tsne_post = combined_2d[len(sel_embs_pre):]

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 获取颜色列表
    unique_ids = sorted(list(set(target_class_ids)))
    palette = sns.color_palette("bright", len(unique_ids))
    color_map = {uid: palette[i] for i, uid in enumerate(unique_ids)}
    
    # 子图1: Pre-Finetune
    for uid in unique_ids:
        idx = sel_labels_pre == uid
        axes[0].scatter(tsne_pre[idx, 0], tsne_pre[idx, 1], 
                        label=id_to_name[str(uid)], s=20, alpha=0.6, c=[color_map[uid]])
    axes[0].set_title("Before Online Finetune")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 子图2: Post-Finetune
    for uid in unique_ids:
        idx = sel_labels_post == uid
        axes[1].scatter(tsne_post[idx, 0], tsne_post[idx, 1], 
                        label=id_to_name[str(uid)], s=20, alpha=0.6, c=[color_map[uid]])
    axes[1].set_title("After Online Finetune")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = 'embedding_improvement.png'
    plt.savefig(save_path, dpi=300)
    print(f"✓ Visualization saved to {save_path}")
    plt.show()

def main():
    device = torch.device(f"cuda:{ARGS['gpu']}")
    set_seed(42)
    
    # 1. 加载 Label Mapping
    map_path = os.path.join(ARGS['data_dir'], "label_mapping.json")
    with open(map_path, 'r') as f:
        label_map = json.load(f)
        num_classes = len(label_map)
        id_to_name = {str(v): k for k, v in label_map.items()} # 反转映射
        
    print(f"Classes: {num_classes}")

    # 2. 数据集
    test_h5 = os.path.join(ARGS['data_dir'], "test_data.h5")
    # 保持和 eval_only_light 一样的参数
        
    dataset = TrafficDataset(test_h5, augmentation=False, max_packets=10, max_bytes=300,global_selection_path=ARGS['global_selector'])
    loader = DataLoader(dataset, batch_size=ARGS['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 3. 初始化模型
    model = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(device)

    # 4. 获取 Pre-Finetune 结果
    print(f"\nLOADING PRE-CHECKPOINT: {ARGS['ckpt_pre']}")
    # model.load_state_dict(torch.load(ARGS['ckpt_pre'], map_location=device))
    # pre_data = get_inference_data(model, loader, device) # (preds, labels, embs)

    checkpoint = torch.load(ARGS['ckpt_pre'], map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    pre_data = get_inference_data(model, loader, device) # (preds, labels, embs)



    # 5. 获取 Post-Finetune 结果
    print(f"\nLOADING POST-CHECKPOINT: {ARGS['ckpt_post']}")
    # model.load_state_dict(torch.load(ARGS['ckpt_post'], map_location=device))
    # post_data = get_inference_data(model, loader, device)

    checkpoint = torch.load(ARGS['ckpt_post'], map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    post_data = get_inference_data(model, loader, device) # (preds, labels, embs)


    # 6. 计算 F1 提升
    print("\nAnalyzing Improvements...")
    preds_pre, labels, _ = pre_data
    preds_post, _, _ = post_data
    
    rep_pre = classification_report(labels, preds_pre, output_dict=True, zero_division=0)
    rep_post = classification_report(labels, preds_post, output_dict=True, zero_division=0)
    
    improvements = []
    for cid in rep_pre.keys():
        if not cid.isdigit(): continue
        
        f1_pre = rep_pre[cid]['f1-score']
        f1_post = rep_post[cid]['f1-score']
        delta = f1_post - f1_pre
        support = rep_pre[cid]['support']
        
        if support > 50: # 只看样本量足够的
            improvements.append((cid, delta, f1_pre, f1_post, support))
            
    # 按提升幅度排序
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {ARGS['top_k']} Improved Classes:")
    print(f"{'ID':<5} {'Name':<20} {'F1-Pre':<8} {'F1-Post':<8} {'Delta':<8}")
    
    top_class_ids = []
    for item in improvements[:ARGS['top_k']]:
        cid, delta, pre, post, _ = item
        print(f"{cid:<5} {id_to_name[cid]:<20} {pre:.4f}   {post:.4f}   +{delta:.4f}")
        top_class_ids.append(int(cid))
        
    # 7. 可视化
    if top_class_ids:
        plot_tsne(pre_data, post_data, top_class_ids, id_to_name)
    else:
        print("No significant classes found.")

if __name__ == '__main__':
    main()