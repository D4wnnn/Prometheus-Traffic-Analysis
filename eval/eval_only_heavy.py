"""
Heavy-Only Baseline 评估脚本
只使用 Heavy Model 进行推理，测试各项指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix 
from tqdm import tqdm

sys.path.append('..')
from models import EncryptedTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed, load_config_from_folder


def evaluate_heavy_only(args):
    """Heavy-Only评估主函数"""
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}')
    
    # ------------------ ### 修改开始：获取类别名称 ------------------
    # 加载数据并解析 label_mapping 以获取类别名称列表
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # 假设 label_map 是 {"Chat": 0, "Email": 1} 这样的字典
    # 我们需要将其转换为列表 ["Chat", "Email"]，且顺序必须对应索引 0, 1...
    if isinstance(label_map, dict):
        # 按 value (索引) 排序，然后取 key (名称)
        class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    elif isinstance(label_map, list):
        class_names = label_map
    else:
        raise ValueError("Unknown format for label_mapping.json")
    if args.pretrain_path:
        model_config, data_max_bytes, data_max_packets = load_config_from_folder(args.pretrain_path)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")
    # ------------------ ### 修改结束 ------------------
    
    test_h5 = os.path.join(args.data_dir, "test_data.h5")
    test_dataset = TrafficDataset(test_h5, augmentation=False, max_bytes=data_max_bytes, max_packets=data_max_packets)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    print(f"{'='*60}")
    print(f"Heavy-Only Baseline Evaluation")
    print(f"{'='*60}")
    print(f"Dataset: {args.data_dir}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 加载Heavy Model
    # model = EncryptedTrafficClassifier(
    #     num_classes=num_classes,
    #     d_byte=128, d_packet=260, byte_layers=12, packet_layers=6, num_heads=5,max_packets=5,max_bytes=150,
    #     use_stats=False, use_adaptive_gating=True
    # ).to(device)
    
    # model = EncryptedTrafficClassifier(
    #     num_classes=num_classes,
    #     d_byte=128, d_packet=260, byte_layers=3, packet_layers=4, num_heads=5,max_packets=100,max_bytes=300,
    #     use_stats=False, use_adaptive_gating=True
    # ).to(device)
    # model_config['disabled_head_indices'] = [4]
    # ========== 处理消融实验参数 ==========
    disabled_head_indices = []
    if args.disable_heads:
        # 将字符串 "1,2,4" 转换为列表 [1, 2, 4]
        try:
            disabled_head_indices = [int(x.strip()) for x in args.disable_heads.split(',') if x.strip()]
            print(f"\n{'!'*40}")
            print(f" Ablation Study Active: Disabling Bias for Heads: {disabled_head_indices}")
            print(f"{'!'*40}\n")
        except ValueError:
            raise ValueError("Error parsing --disable_heads. Ensure format is like '1,2,4'")
    # 将消融参数加入 model_config 字典
    model_config['disabled_head_indices'] = disabled_head_indices    
    
    model = EncryptedTrafficClassifier(
        num_classes=num_classes,
        **model_config
    ).to(device)
    checkpoint = torch.load(args.heavy_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    print("✓ Heavy Model loaded successfully\n")
    
    # 统计变量
    all_preds = []
    all_labels = []
    batch_latencies = []
    total_samples = 0
    
    print("Starting inference...\n")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Heavy-Only Inference'):
            batch_start = time.time()
            
            # 准备数据
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            # 推理
            logits = model(batch_data)
            _, preds = logits.max(1)
            
            # 记录结果
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # 记录延迟
            batch_end = time.time()
            batch_latencies.append(batch_end - batch_start)
            total_samples += labels.size(0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 聚合预测结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision *= 100
    recall *= 100
    f1 *= 100
    
    throughput = total_samples / total_time
    
    # 延迟统计
    latencies_ms = np.array(batch_latencies) * 1000
    latency_p50 = np.percentile(latencies_ms, 50)
    latency_p95 = np.percentile(latencies_ms, 95)
    latency_p99 = np.percentile(latencies_ms, 99)
    latency_mean = np.mean(latencies_ms)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Heavy-Only Evaluation Results")
    print(f"{'='*60}")
    print(f"Total Time:        {total_time:.2f}s")
    print(f"Total Samples:     {total_samples}")
    print(f"Throughput:        {throughput:.2f} packets/s")
    print(f"{'-'*60}")
    print(f"Accuracy:          {accuracy:.2f}%")
    print(f"Precision:         {precision:.2f}%")
    print(f"Recall:            {recall:.2f}%")
    print(f"F1 Score:          {f1:.2f}%")
    print(f"{'-'*60}")
    
    # ------------------ ### 修改开始：计算并打印混淆矩阵 ------------------
    print("Detailed Confusion Matrix Analysis:")
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 创建 DataFrame
    # 确保 class_names 的长度和 cm 的维度一致
    if len(class_names) == cm.shape[0]:
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        print("\n[Confusion Matrix Table]")
        print("Rows = True Labels (Ground Truth)")
        print("Columns = Predicted Labels")
        print("-" * 20)
        
        # # 尝试使用 markdown 格式输出，方便 LLM 读取
        # try:
        #     print(df_cm.to_markdown())
        # except ImportError:
        #     # 如果没有安装 tabulate 库，退回到 string 格式
        #     print(df_cm.to_string())
            
        # 保存 csv，方便后续绘图或写论文
        csv_filename = args.save_result.replace('.json', '_cm.csv')
        df_cm.to_csv(csv_filename)
        print(f"\n✓ Confusion matrix saved to {csv_filename}")
    else:
        print(f"Warning: Class names count ({len(class_names)}) does not match confusion matrix shape ({cm.shape[0]}). Skipping detailed print.")
        print("Raw CM:\n", cm)
    # ------------------ ### 修改结束 ------------------

    print(f"{'-'*60}")
    print(f"Latency (per batch):")
    print(f"  Mean:            {latency_mean:.2f} ms")
    print(f"  P50:             {latency_p50:.2f} ms")
    print(f"  P95:             {latency_p95:.2f} ms")
    print(f"  P99:             {latency_p99:.2f} ms")
    print(f"{'='*60}\n")
    
    # 保存结果
    results = {
        'mode': 'Heavy-Only',
        'config': {
            'model_path': args.heavy_path,
            'batch_size': args.batch_size,
            'data_dir': args.data_dir,
        },
        'metrics': {
            'total_time': total_time,
            'total_samples': total_samples,
            'throughput': throughput,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'heavy_calls': total_samples,  # 所有样本都用Heavy
            'heavy_call_rate': 100.0,
            'latency_mean': float(latency_mean),
            'latency_p50': float(latency_p50),
            'latency_p95': float(latency_p95),
            'latency_p99': float(latency_p99),
        },
        'latencies': latencies_ms.tolist(),  # 每个batch的延迟
        # 可选：也将混淆矩阵数据存入 json
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    with open(args.save_result, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {args.save_result}\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--heavy_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_result', type=str, default='results_heavy_only.json')
    parser.add_argument('--pretrain_path', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--disable_heads', type=str, default='',help='要禁用偏置的头索引，用逗号分隔。例如 "1,3" 表示禁用时间(1)和方向(3)。可选: 1,2,3,4')
    
    args = parser.parse_args()
    evaluate_heavy_only(args)