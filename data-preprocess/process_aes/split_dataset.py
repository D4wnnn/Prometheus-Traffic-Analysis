import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio, seed=42):
    # 1. 校验比例和
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5:
        print("错误：训练、验证、测试比例之和必须等于 1.0")
        return

    # 设置随机种子，保证可复现性
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 检查输入目录是否存在
    if not input_path.exists():
        print(f"错误：输入目录 {input_dir} 不存在")
        return

    # 获取所有一级目录（Label）
    labels = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not labels:
        print("未找到Label目录，请检查输入目录结构。")
        return

    print(f"发现 {len(labels)} 个类别 (Label)，准备开始处理...")

    # 定义集合名称
    splits = ['train', 'val', 'test']

    # 遍历每个 Label 进行切分
    for label_dir in labels:
        label_name = label_dir.name
        # 获取该 Label 下所有的 pcap 文件
        # 如果后缀不是 .pcap，可以修改 glob 的参数，例如 '*.pcapng' 或 '*.*'
        files = list(label_dir.glob('*.pcap'))
        
        # 随机打乱文件列表
        random.shuffle(files)
        
        total_files = len(files)
        if total_files == 0:
            print(f"跳过空目录: {label_name}")
            continue

        # 计算切分点
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # 切分数据
        split_files = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        print(f"正在处理 Label: {label_name} (总数: {total_files}) -> "
              f"Train: {len(split_files['train'])}, "
              f"Val: {len(split_files['val'])}, "
              f"Test: {len(split_files['test'])}")

        # 执行复制操作
        for split_name, file_list in split_files.items():
            # 目标目录结构: output/train/label_name/
            target_dir = output_path / split_name / label_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in file_list:
                # 复制文件
                # shutil.copy2 会保留文件的元数据（如时间戳），如果不需要可以使用 shutil.copy
                shutil.copy2(file_path, target_dir / file_path.name)

    print("\n--- 数据集切分完成 ---")
    print(f"输出目录: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机切分流量分类数据集 (Pcap)")
    
    # 添加参数
    parser.add_argument('--input', type=str, required=True, help='原始数据集的一级目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出数据集的根目录路径')
    parser.add_argument('--train', type=float, default=0.7, help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='验证集比例 (默认: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='测试集比例 (默认: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')

    args = parser.parse_args()

    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )