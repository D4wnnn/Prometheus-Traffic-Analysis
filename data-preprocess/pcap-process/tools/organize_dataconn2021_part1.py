import os
import shutil
from tqdm import tqdm  # 如果没有安装tqdm，可以将下面的 tqdm() 去掉，直接用 range

def organize_dataset():
    # ================= 配置路径 =================
    # 基础路径 (根据你提供的上下文设置)
    base_path = "path/to/your/dataset"  # 请修改为实际路径
    
    # 原始 pcap 文件所在的目录
    source_dir = os.path.join(base_path, "real_data")
    
    # 标签文件路径
    label_file_path = os.path.join(base_path, "part1_label.txt")
    
    # 输出目录 (整理后的数据存放位置)
    output_dir = os.path.join(base_path, "organized_real_data")
    # ===========================================

    # 1. 读取标签文件
    print(f"正在读取标签文件: {label_file_path} ...")
    file_label_map = {}
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = parts[1]
                    file_label_map[filename] = label
    except FileNotFoundError:
        print(f"错误: 找不到标签文件 {label_file_path}")
        return

    print(f"共找到 {len(file_label_map)} 个样本映射关系。")

    # 2. 开始整理
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    success_count = 0
    missing_count = 0

    # 使用 tqdm 显示进度条
    print("开始复制并整理文件...")
    for filename, label in tqdm(file_label_map.items()):
        src_path = os.path.join(source_dir, filename)
        
        # 目标文件夹路径 (一级目录是 label)
        dst_folder = os.path.join(output_dir, label)
        dst_path = os.path.join(dst_folder, filename)

        # 检查源文件是否存在
        if os.path.exists(src_path):
            # 如果目标 label 文件夹不存在，则创建
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            
            # 复制文件 (使用 copy2 保留文件元数据)
            shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            # print(f"警告: 源文件 {filename} 未在 {source_dir} 中找到")
            missing_count += 1

    print("-" * 30)
    print("整理完成！")
    print(f"成功整理: {success_count} 个文件")
    if missing_count > 0:
        print(f"缺失文件: {missing_count} 个 (在标签列表中但目录下不存在)")
    print(f"新数据集保存在: {output_dir}")

if __name__ == "__main__":
    organize_dataset()