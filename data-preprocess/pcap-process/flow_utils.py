import os
import shutil
from pathlib import Path
from datetime import datetime
import logging
import struct
import random

logger = logging.getLogger(__name__)

class FlowInfo:
    """流信息类"""
    def __init__(self, filepath, start_time, packet_count=0, size=0):
        self.filepath = Path(filepath)
        self.start_time = start_time
        self.packet_count = packet_count
        self.size = size
        self.filename = self.filepath.name
    
    def __repr__(self):
        return f"FlowInfo({self.filename}, {self.start_time})"
    
    def __lt__(self, other):
        """用于排序"""
        return self.start_time < other.start_time

def get_pcap_timestamp_direct(pcap_path):
    """直接从PCAP文件读取第一个包的时间戳（不依赖pyshark）"""
    try:
        with open(pcap_path, 'rb') as f:
            # 读取PCAP文件头（24字节）
            pcap_header = f.read(24)
            if len(pcap_header) < 24:
                return None
            
            # 检查魔数以确定字节序
            magic = struct.unpack('I', pcap_header[:4])[0]
            if magic == 0xa1b2c3d4:
                # 标准字节序
                endian = '<'
            elif magic == 0xd4c3b2a1:
                # 反向字节序
                endian = '>'
            else:
                logger.warning(f"无效的PCAP文件: {pcap_path}")
                return None
            
            # 读取第一个包头（16字节）
            packet_header = f.read(16)
            if len(packet_header) < 16:
                return None
            
            # 解析时间戳
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', packet_header)
            
            # 转换为datetime对象
            timestamp = ts_sec + ts_usec / 1000000.0
            return datetime.fromtimestamp(timestamp)
            
    except Exception as e:
        logger.error(f"读取PCAP时间戳失败 {pcap_path}: {e}")
        return None

def get_flow_start_time_pyshark(pcap_path):
    """使用pyshark获取pcap文件的第一个包的时间戳（备用方法）"""
    try:
        import pyshark
        
        # 确保使用绝对路径
        pcap_path = Path(pcap_path).resolve()
        
        # 检查文件是否存在和大小
        if not pcap_path.exists():
            logger.error(f"文件不存在: {pcap_path}")
            return None
            
        file_size = pcap_path.stat().st_size
        if file_size == 0:
            logger.warning(f"空文件: {pcap_path}")
            return None
        
        logger.debug(f"尝试读取文件: {pcap_path} (大小: {file_size} bytes)")
        
        # 使用pyshark读取
        # 注意：使用display_filter而不是only_summaries
        cap = pyshark.FileCapture(
            str(pcap_path),
            keep_packets=False,
            use_json=True,
            include_raw=False
        )
        
        # 读取第一个包
        first_packet = None
        packet_count = 0
        for packet in cap:
            packet_count += 1
            if packet_count == 1:
                first_packet = packet
                break
        
        cap.close()
        
        if first_packet:
            # 尝试不同的时间戳属性
            timestamp = None
            
            # 方法1: sniff_timestamp
            if hasattr(first_packet, 'sniff_timestamp'):
                timestamp = float(first_packet.sniff_timestamp)
            # 方法2: frame_info
            elif hasattr(first_packet, 'frame_info'):
                if hasattr(first_packet.frame_info, 'time_epoch'):
                    timestamp = float(first_packet.frame_info.time_epoch)
            # 方法3: sniff_time
            elif hasattr(first_packet, 'sniff_time'):
                timestamp = first_packet.sniff_time.timestamp()
            
            if timestamp:
                return datetime.fromtimestamp(timestamp)
            else:
                logger.warning(f"无法从包中提取时间戳: {pcap_path}")
                return None
        else:
            logger.warning(f"文件中没有数据包: {pcap_path}")
            return None
            
    except Exception as e:
        logger.error(f"pyshark读取失败 {pcap_path}: {e}")
        return None

def get_flow_start_time(pcap_path):
    """获取pcap文件的第一个包的时间戳（综合方法）"""
    # 首先尝试直接读取（更快更可靠）
    timestamp = get_pcap_timestamp_direct(pcap_path)
    
    # 如果失败，尝试使用pyshark
    if timestamp is None:
        logger.debug(f"直接读取失败，尝试使用pyshark: {pcap_path}")
        timestamp = get_flow_start_time_pyshark(pcap_path)
    
    return timestamp

def get_pcap_packet_count_direct(pcap_path):
    """直接从PCAP文件读取包数量（不依赖pyshark）"""
    try:
        with open(pcap_path, 'rb') as f:
            # 读取PCAP文件头（24字节）
            pcap_header = f.read(24)
            if len(pcap_header) < 24:
                return 0
            
            # 检查魔数以确定字节序
            magic = struct.unpack('I', pcap_header[:4])[0]
            if magic == 0xa1b2c3d4:
                # 标准字节序
                endian = '<'
            elif magic == 0xd4c3b2a1:
                # 反向字节序
                endian = '>'
            else:
                logger.warning(f"无效的PCAP文件: {pcap_path}")
                return 0
            
            packet_count = 0
            # 逐个读取数据包
            while True:
                # 读取包头（16字节）
                packet_header = f.read(16)
                if len(packet_header) < 16:
                    break
                
                # 解析包头获取包长度
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', packet_header)
                
                # 跳过包数据
                f.seek(incl_len, 1)  # 从当前位置向前跳过incl_len字节
                
                packet_count += 1
            
            return packet_count
            
    except Exception as e:
        logger.error(f"读取PCAP包数量失败 {pcap_path}: {e}")
        return 0

def get_flow_info(pcap_path):
    """获取流的详细信息"""
    pcap_path = Path(pcap_path)
    
    # 检查文件是否存在
    if not pcap_path.exists():
        logger.error(f"文件不存在: {pcap_path}")
        return None
    
    # 获取文件大小
    size = pcap_path.stat().st_size
    
    # 跳过空文件
    if size == 0:
        logger.warning(f"跳过空文件: {pcap_path}")
        return None
    
    # 获取时间戳
    start_time = get_flow_start_time(pcap_path)
    if start_time is None:
        # 如果无法获取时间戳，使用文件修改时间作为备选
        logger.warning(f"无法获取时间戳，使用文件修改时间: {pcap_path}")
        start_time = datetime.fromtimestamp(pcap_path.stat().st_mtime)
    
    # 获取包数量
    packet_count = get_pcap_packet_count_direct(pcap_path)
    return FlowInfo(pcap_path, start_time, packet_count, size)

def collect_flows(flow_dir):
    """收集目录下所有流文件并获取信息"""
    flow_dir = Path(flow_dir)
    flows = []
    
    # 获取所有pcap文件
    pcap_patterns = ['*.pcap', '*.cap', '*.pcapng']
    pcap_files = []
    for pattern in pcap_patterns:
        pcap_files.extend(flow_dir.glob(pattern))
    
    if not pcap_files:
        logger.warning(f"目录 {flow_dir} 中没有找到PCAP文件")
        return flows
    
    logger.info(f"正在收集 {len(pcap_files)} 个流的信息...")
    
    valid_count = 0
    skip_count = 0
    error_count = 0
    
    for i, pcap_file in enumerate(pcap_files, 1):
        if i % 10 == 0:
            logger.debug(f"处理进度: {i}/{len(pcap_files)}")
        
        try:
            flow_info = get_flow_info(pcap_file)
            if flow_info:
                # 可以根据配置过滤太小或太大的流
                from config import MIN_FLOW_PACKETS, MAX_FLOW_SIZE
                if flow_info.size > MAX_FLOW_SIZE:
                    logger.debug(f"跳过过大的流: {flow_info.filename} ({flow_info.size} bytes)")
                    skip_count += 1
                    continue
                if flow_info.packet_count < MIN_FLOW_PACKETS:  # 跳过太小的文件
                    logger.info(f"跳过过小的流: {flow_info.filename} ({flow_info.size} bytes)")
                    # print(f"跳过过小的流: {flow_info.filename} ({flow_info.size} bytes)")
                    skip_count += 1
                    continue
                    
                flows.append(flow_info)
                valid_count += 1
            else:
                error_count += 1
                logger.warning(f"无法处理流文件: {pcap_file}")
        except Exception as e:
            error_count += 1
            logger.error(f"处理流文件出错 {pcap_file}: {e}")
    
    logger.info(f"流收集完成: 有效={valid_count}, 跳过={skip_count}, 错误={error_count}")
    return flows

def split_flows_by_time(flows, train_ratio=0.7, val_ratio=0.15):
    """按时间顺序切分流为训练集、验证集和测试集
    
    Args:
        flows: 流列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        train_flows, val_flows, test_flows: 三个流列表
    """
    if not flows:
        return [], [], []
    
    # 按开始时间排序
    sorted_flows = sorted(flows, key=lambda x: x.start_time)
    
    total_count = len(sorted_flows)
    
    # 计算切分点
    train_end_idx = int(total_count * train_ratio)
    val_end_idx = int(total_count * (train_ratio + val_ratio))
    
    # 确保每个集合至少有1个样本（如果总数>=3）
    if total_count >= 3:
        if train_end_idx == 0:
            train_end_idx = 1
        if val_end_idx <= train_end_idx:
            val_end_idx = train_end_idx + 1
        if val_end_idx >= total_count:
            val_end_idx = total_count - 1
    
    train_flows = sorted_flows[:train_end_idx]
    val_flows = sorted_flows[train_end_idx:val_end_idx]
    test_flows = sorted_flows[val_end_idx:]
    
    logger.info(f"按时间切分结果: 训练集 {len(train_flows)} 个流, 验证集 {len(val_flows)} 个流, 测试集 {len(test_flows)} 个流")
    
    # 输出时间范围信息
    if train_flows:
        logger.info(f"训练集时间范围: {train_flows[0].start_time} - {train_flows[-1].start_time}")
    if val_flows:
        logger.info(f"验证集时间范围: {val_flows[0].start_time} - {val_flows[-1].start_time}")
    if test_flows:
        logger.info(f"测试集时间范围: {test_flows[0].start_time} - {test_flows[-1].start_time}")
    
    return train_flows, val_flows, test_flows

def split_flows_randomly(flows, train_ratio=0.7, val_ratio=0.15, seed=None):
    if not flows:
        return [], [], []
    
    flows_copy = flows.copy()
    
    if seed is not None:
        random.seed(seed)
    
    random.shuffle(flows_copy)
    
    total_count = len(flows_copy)
    
    # 计算切分点
    train_end_idx = int(total_count * train_ratio)
    val_end_idx = int(total_count * (train_ratio + val_ratio))
    
    # --- 修改开始：更智能的边界检查 ---
    if total_count > 0:
        # 1. 只有当 train_ratio > 0 且计算结果为 0 时，才强行给 train 留一个
        if train_ratio > 0 and train_end_idx == 0:
            train_end_idx = 1
            
        # 2. 只有当 val_ratio > 0 且计算结果挤压了 val 时，才强行给 val 留一个
        # 注意：要防止越界
        if val_ratio > 0 and val_end_idx <= train_end_idx:
            val_end_idx = min(train_end_idx + 1, total_count)
            
        # 3. 关键修正：只有当本来就打算留测试集（比例之和明显小于1）时，才强行保留 test
        # 使用 0.999 避免浮点数精度问题
        if (train_ratio + val_ratio) < 0.999 and val_end_idx >= total_count:
            val_end_idx = total_count - 1
        
        # 4. 如果不需要测试集，允许 val_end_idx 等于 total_count
        elif val_end_idx > total_count:
            val_end_idx = total_count
    # --- 修改结束 ---

    train_flows = flows_copy[:train_end_idx]
    val_flows = flows_copy[train_end_idx:val_end_idx]
    test_flows = flows_copy[val_end_idx:]
    
    logger.info(f"随机切分结果: 训练集 {len(train_flows)} 个流, 验证集 {len(val_flows)} 个流, 测试集 {len(test_flows)} 个流")
    
    return train_flows, val_flows, test_flows

def save_flows_to_directory(flows, target_dir):
    """将流文件复制到目标目录"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for flow_info in flows:
        try:
            # 使用绝对路径
            source_path = flow_info.filepath.resolve()
            target_path = target_dir / flow_info.filename
            
            # 如果目标文件已存在，可以选择跳过或覆盖
            if target_path.exists():
                logger.debug(f"目标文件已存在，覆盖: {target_path}")
            
            shutil.copy2(source_path, target_path)
            success_count += 1
        except Exception as e:
            logger.error(f"复制文件 {flow_info.filename} 失败: {e}")
    
    logger.info(f"成功保存 {success_count}/{len(flows)} 个流到 {target_dir}")
    return success_count

def save_flows_to_directory_with_prefix(flows, target_dir, prefix=""):
    """将流文件复制到目标目录，并添加前缀到文件名"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, flow_info in enumerate(flows, 1):
        try:
            # 使用绝对路径
            source_path = flow_info.filepath.resolve()
            
            # 构造新文件名：prefix_originalname
            if prefix:
                # 保留原始文件名的后缀部分（如 TCP_xxx.pcap）
                original_name = flow_info.filename
                # 提取文件扩展名
                suffix = ''.join(source_path.suffixes)  # 获取所有后缀，如 .pcap
                # 去掉扩展名的原始名称
                name_without_suffix = original_name[:-len(suffix)] if suffix else original_name
                # 构造新名称
                new_filename = f"{prefix}_{name_without_suffix}{suffix}"
            else:
                new_filename = flow_info.filename
            
            target_path = target_dir / new_filename
            
            # 如果文件名冲突，添加序号
            if target_path.exists():
                base_name = target_path.stem
                extension = ''.join(target_path.suffixes)
                counter = 1
                while target_path.exists():
                    new_name = f"{base_name}_{counter}{extension}"
                    target_path = target_dir / new_name
                    counter += 1
                logger.debug(f"文件名冲突，使用新名称: {target_path.name}")
            
            shutil.copy2(source_path, target_path)
            success_count += 1
            
            if i % 100 == 0:
                logger.debug(f"已保存 {i}/{len(flows)} 个流文件")
                
        except Exception as e:
            logger.error(f"复制文件 {flow_info.filename} 失败: {e}")
    
    logger.info(f"成功保存 {success_count}/{len(flows)} 个流到 {target_dir}")
    return success_count