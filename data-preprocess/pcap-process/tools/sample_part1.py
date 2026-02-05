import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(input_dir, output_dir, split_ratios=(0.8, 0.1, 0.1), seed=42, clear_output=False, max_samples=None):
    """
    按照比例随机划分数据集（Copy模式）
    :param clear_output: 是否清空输出目录
    :param max_samples: 每个类别允许的最大样本数 (int)，超过则随机采样
    """
    # 1. 设置随机种子，保证可复现性
    random.seed(seed)
    
    # 2. 检查并归一化比例
    total_ratio = sum(split_ratios)
    train_r, val_r, test_r = [x / total_ratio for x in split_ratios]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: 输入目录 {input_dir} 不存在")
        return

    # --- 新增逻辑: 清空输出目录 ---
    if clear_output:
        if output_path.exists():
            print(f"Warning: 检测到 --clear 参数，正在删除输出目录: {output_dir}")
            try:
                shutil.rmtree(output_path)
                print("输出目录已清空。")
            except Exception as e:
                print(f"Error: 清空目录失败 - {e}")
                return
    # ---------------------------

    # 3. 遍历一级目录（Labels）
    # 获取所有子文件夹，排除隐藏文件
    categories = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"{'='*20} 开始处理 {'='*20}")
    print(f"检测到类别数量: {len(categories)}")
    print(f"划分比例 (Train/Val/Test): {train_r:.2f} / {val_r:.2f} / {test_r:.2f}")
    if max_samples:
        print(f"单类最大样本限制: {max_samples}")
    
    for category in categories:
        label_name = category.name
        
        # 获取该类别下所有文件 (过滤 .pcap 或 .pcapng)
        files = [f for f in category.glob('*') if f.is_file() and f.suffix in ['.pcap', '.pcapng']]
        
        # 如果没有pcap文件，尝试获取所有文件
        if not files:
             files = [f for f in category.glob('*') if f.is_file()]
        
        file_count_original = len(files)
        if file_count_original == 0:
            print(f"警告: 类别 {label_name} 为空，跳过。")
            continue
            
        # 打乱顺序 (这一步非常重要，既用于随机划分，也用于随机下采样)
        random.shuffle(files)
        
        # --- 新增逻辑: 最大样本限制 ---
        if max_samples is not None and file_count_original > max_samples:
            files = files[:max_samples] # 截取前 max_samples 个
            print(f"类别 [{label_name:<15}] -> 原始数量: {file_count_original} | 超过上限，随机采样至: {max_samples}")
        
        file_count = len(files)
        # ---------------------------
        
        # 计算切分点
        train_end = int(file_count * train_r)
        val_end = train_end + int(file_count * val_r)
        
        # 切片
        splits = {
            'train': files[:train_end],
            'val':   files[train_end:val_end],
            'test':  files[val_end:]
        }
        
        # 执行复制
        for split_name, split_files in splits.items():
            # 目标目录: output/train/label_name/
            dest_dir = output_path / split_name / label_name
            os.makedirs(dest_dir, exist_ok=True)
            
            for file in split_files:
                try:
                    # copy2 保留文件元数据（时间戳等）
                    shutil.copy2(file, dest_dir / file.name)
                except Exception as e:
                    print(f"复制失败 {file.name}: {e}")
        
        # 打印当前类别的统计信息（如果有采样，打印的是采样后的分布）
        print(f"类别 [{label_name:<15}] -> 处理后总数: {file_count:<5} "
              f"| Train: {len(splits['train']):<5} "
              f"| Val: {len(splits['val']):<5} "
              f"| Test: {len(splits['test']):<5}")

    print(f"{'='*20} 处理完成 {'='*20}")
    print(f"数据集已保存至: {output_dir}")

if __name__ == "__main__":
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="按比例随机划分PCAP数据集")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="原始数据集根目录路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出目录路径")
    parser.add_argument("--ratios", "-r", type=float, nargs=3, default=[0.75, 0.15, 0.15], 
                        help="划分比例 (默认: 0.75 0.15 0.15)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="随机种子 (默认: 42)")
    
    # --- 新增参数 ---
    parser.add_argument("--clear", action="store_true", help="是否在运行前清空输出目录")
    parser.add_argument("--max", type=int, default=None, help="每个类别采样的最大样本数 (默认不限制)")
    # ----------------

    args = parser.parse_args()
    
    split_dataset(
        input_dir=args.input, 
        output_dir=args.output, 
        split_ratios=args.ratios, 
        seed=args.seed,
        clear_output=args.clear,
        max_samples=args.max
    )