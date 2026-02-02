import torch
import torch.nn as nn
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EncryptedTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed, load_config_from_folder, load_config_from_file

# ==========================================
# 核心扰动逻辑
# ==========================================
class Perturber:
    def __init__(self, perturb_type, noise_ratio, device):
        self.type = perturb_type
        self.ratio = noise_ratio
        self.device = device
    
    def apply(self, batch_data):
        """
        batch_data: dict containing 'bytes_nlp', 'seq_nlp', 'tcp_data'
        """
        if self.ratio <= 0 or self.type == 'none':
            return batch_data

        # 1. Byte Masking: 随机将字节替换为随机噪声或0
        # 针对 bytes_nlp (LongTensor)
        if self.type == 'byte_mask':
            inputs = batch_data['bytes_nlp']
            # 生成掩码 [Batch, Seq, Bytes]
            mask = torch.rand_like(inputs.float()) < self.ratio
            # 生成噪声 (0-255)
            noise = torch.randint(0, 256, inputs.shape).to(self.device)
            # 应用扰动 (保持 Pad 0 不变通常更合理，但作为强攻击也可以全扰动)
            # 这里我们假设非0部分才是有效内容
            is_content = inputs != 0
            final_mask = mask & is_content
            batch_data['bytes_nlp'] = torch.where(final_mask, noise, inputs)

        # 2. Time Jitter: 对序列特征添加高斯噪声
        # 针对 seq_nlp (FloatTensor) -> 通常包含 [IAT, Direction, etc.]
        elif self.type == 'time_jitter':
            inputs = batch_data['seq_nlp'].clone()
            
            # 提取 IAT 特征 (第3列，索引2)
            iat_feat = inputs[:, :, 2]
            
            # 生成标准高斯噪声 N(0, 1) 并缩放
            noise = torch.randn_like(iat_feat) * self.ratio
            
            # 应用乘性扰动
            inputs[:, :, 2] = iat_feat * (1 + noise)
            
            batch_data['seq_nlp'] = inputs

        # 3. Packet Dropout: 随机丢弃整个包
        # 同时影响 bytes_nlp, seq_nlp, tcp_data
        elif self.type == 'packet_drop':
            # 维度通常是 [Batch, Max_Packets, ...]
            B, L = batch_data['bytes_nlp'].shape[0], batch_data['bytes_nlp'].shape[1]
            
            # 生成包级别的掩码 [Batch, L]
            keep_prob = 1.0 - self.ratio
            # 伯努利分布采样，1代表保留，0代表丢弃
            mask = torch.bernoulli(torch.full((B, L), keep_prob)).to(self.device)
            
            # 扩展掩码维度以应用到各个特征
            # bytes: [B, L, D_byte]
            mask_bytes = mask.unsqueeze(-1).expand_as(batch_data['bytes_nlp'])
            # seq: [B, L, D_seq]
            mask_seq = mask.unsqueeze(-1).expand_as(batch_data['seq_nlp'])
            
            # 执行丢弃 (置0)
            batch_data['bytes_nlp'] = batch_data['bytes_nlp'] * mask_bytes.long()
            batch_data['seq_nlp'] = batch_data['seq_nlp'] * mask_seq
            if 'tcp_data' in batch_data:
                 mask_tcp = mask.unsqueeze(-1).expand_as(batch_data['tcp_data'])
                 batch_data['tcp_data'] = batch_data['tcp_data'] * mask_tcp
                 
        # 4. [NEW] Length Jitter: 模拟参考代码的序列扰动 (随机替换)
        # 参考 FS-Net 的 perturb_sequence_discrete 逻辑
        elif self.type == 'length_jitter':
            # Clone 防止原地修改报错
            inputs = batch_data['seq_nlp'].clone()
            
            # --- 只提取第 0 维: pkt_len ---
            len_feat = inputs[:, :, 0]
            
            # 生成掩码 [Batch, Seq]
            mask = torch.rand_like(len_feat) < self.ratio
            
            # 生成噪声: 模拟包长 0-1500
            # 注意：如果模型输入是归一化后的数据，请将 1501 改为 1.0 或相应范围
            noise = torch.randint(0, 1501, len_feat.shape).float().to(self.device)
            
            # 保护 Padding (假设 pkt_len 为 0 表示 Pad)
            is_content = len_feat != 0
            final_mask = mask & is_content
            
            # 只替换被 Mask 的部分的 pkt_len
            inputs[:, :, 0] = torch.where(final_mask, noise, len_feat)
            
            # 更新回 batch_data
            batch_data['seq_nlp'] = inputs
        elif self.type == 'direction_flip':
            inputs = batch_data['seq_nlp'].clone()
            
            # --- 锁定 Index 1 (Direction) ---
            dir_feat = inputs[:, :, 1]
            
            # 生成掩码
            mask = torch.rand_like(dir_feat) < self.ratio
            
            # 执行取反 (翻转方向)
            # Padding (0) 取反后仍为 0，不受影响
            inputs[:, :, 1] = torch.where(mask, -dir_feat, dir_feat)
            
            batch_data['seq_nlp'] = inputs
        return batch_data

# ==========================================
# 评估主程序
# ==========================================
def evaluate_robustness(args):
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}')

    # --- 1. Load Config & Class Names ---
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    if isinstance(label_map, dict):
        class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    else:
        class_names = label_map
    num_classes = len(class_names)

    # Load Model Config
    if args.config_path:
        model_config, data_max_bytes, data_max_packets = load_config_from_file(args.config_path)
    elif args.pretrain_path:
         model_config, data_max_bytes, data_max_packets = load_config_from_folder(args.pretrain_path)
    else:
        raise ValueError("Need config_path or pretrain_path")

    # --- 2. Load Data ---
    test_h5 = os.path.join(args.data_dir, "test_data.h5")
    test_dataset = TrafficDataset(test_h5, augmentation=False, max_bytes=data_max_bytes, max_packets=data_max_packets)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- 3. Load Model ---
    # Handle Ablations
    disabled_head_indices = []
    if args.disable_heads:
        disabled_head_indices = [int(x.strip()) for x in args.disable_heads.split(',') if x.strip()]
    model_config['disabled_head_indices'] = disabled_head_indices

    model = EncryptedTrafficClassifier(num_classes=num_classes, **model_config).to(device)
    
    checkpoint = torch.load(args.heavy_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # Remove 'module.' prefix if saved with DDP
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # --- 4. Setup Perturber ---
    perturber = Perturber(args.perturb_type, args.noise_ratio, device)
    
    print(f"\n{'='*60}")
    print(f"Robustness Evaluation: {args.perturb_type.upper()} @ {args.noise_ratio}")
    print(f"{'='*60}")
    print(f"Data: {args.data_dir}")
    print(f"Model: {args.heavy_path}")
    print(f"{'='*60}\n")

    # --- 5. Inference Loop ---
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Eval ({args.perturb_type}={args.noise_ratio})'):
            # Prepare Data
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            batch_data = perturber.apply(batch_data)
            # --------------------------

            # Forward
            logits = model(batch_data)
            _, preds = logits.max(1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"\nResult >> Noise: {args.noise_ratio} | Type: {args.perturb_type} | Accuracy: {accuracy:.2f}%")

    # --- 6. Save Single Result ---
    # 我们只保存关键指标，由Shell脚本汇总
    result = {
        "perturb_type": args.perturb_type,
        "noise_ratio": args.noise_ratio,
        "accuracy": accuracy,
        "dataset": os.path.basename(args.data_dir)
    }
    
    # Append to a JSONL file (JSON Lines) for easy plotting later
    print(f"Saving result to {args.save_result}\n")
    # os.makedirs(os.path.dirname(args.save_result), exist_ok=True)
    # with open(args.save_result, 'a') as f:
    #     f.write(json.dumps(result) + "\n")
    output_dir = os.path.dirname(args.save_result)
    if output_dir:  # <--- 关键修改：只有当路径包含目录时（不是空字符串）才创建文件夹
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.save_result, 'a') as f:
        f.write(json.dumps(result) + "\n")

if __name__ == '__main__':
    from torch.utils.data import DataLoader # Re-import locally if needed

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--heavy_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--pretrain_path', type=str, default=None)
    
    # Attack Args
    parser.add_argument('--perturb_type', type=str, default='none',
                        help='Type of perturbation attack')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Intensity of noise (0.0 - 1.0)')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_result', type=str, default='robustness_results.jsonl')
    parser.add_argument('--disable_heads', type=str, default='')

    args = parser.parse_args()
    evaluate_robustness(args)