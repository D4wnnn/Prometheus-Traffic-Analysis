import logging
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

def setup_logger(name, log_file=None):
    """设置日志器"""
    # 1. 获取根 logger (不传参数，或者传空字符串)
    # 这样配置会对所有 import logging 的模块生效
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    
    # 为了避免重复打印（如果多次调用setup），先清空已有的handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

    # 格式化器
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    if log_file:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 返回具体名字的 logger 用于 main.py 调用，但这其实不重要了，因为根 logger 已经配好了
    return logging.getLogger(name)

def clean_directory(dir_path):
    """清空目录内容"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


protocols = {
    'network_management_protocols': ['icmp', 'icmpv6', 'dhcp', 'dhcpv6', 'igmp', 'snmp', 'arp', 'cops'],
    'nat_protocols': ['nat-pmp', 'rsip'],
    'route_management_protocols': ['db-lsp', 'db-lsp-disc', 'pathport', 'stp', 'bfd_echo', 'bgp', 'ecmp'],
    'service_management_protocols': ['ssdp', 'lldp', 'srvloc', 'ipxsap', 'opa', 'cbsp'],
    'link-local_protocols': ['llmnr', 'nbns', 'mdns', 'lsd'],
    'link_management_protocols': ['llc'],
    'distributed_protocols': ['thrift', 'dcerpc', 'rmi'],
    'real_time_protocols': ['rtcp', 'stun'],
    'remote_access_protocols': ['vnc', 'x11', 'msnms'],
    'network_time_protocols': ['ntp'],
    'security_protocols': ['ocsp', 'pkix-cert', 'egd', 'chargen', 'tpm', 'knet'],
    'industrial_protocols': ['r-goose', 'dcp-pft', 'dcp-af', 'nxp_802154_sniffer', 'enip', 'c1222', 'ax4000'],
    'file_protocols': ['lanman', 'bjnp', 'spoolss', 'ndps', 'laplink', 'bzr', 'cvspserver'],
    'quake_protocols': ['quake', 'quake2', 'quake3', 'quakeworld'],
    'iot_management_protocols': ['bat.vis', 'tplink-smarthome', 'coap','mqtt'],
    'mobile_protocols': ['gsm_ipa'],
    'database_protocols': ['tds']
}
def filter_pcap():
    rule = ""
    for type in protocols:
        for protocol in protocols[type]:
            rule += f"not {protocol} and "
    rule = rule[:-5]
    
    return rule




# clean_protocols = '"not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp and frame.len > 80"'
        # "not (arp or dhcp or ntp or icmp or dns) and (tcp or udp) and frame.len > 80",
# ISCX-VPN "tcp or udp"
# USTC-TFC "tcp or udp"



def run_tshark(input_pcap, output_pcap, filter_rule="tcp or udp"):
    """使用 tshark 过滤 pcap 文件，去除 icmp 和 dns 数据包"""
    command = [
        "tshark",
        "-r",
        str(input_pcap),
        "-Y",
        # "not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp and frame.len > 80",
        # "not (arp or dhcp or ntp or icmp or dns) and (tcp or udp) and frame.len > 80",
        filter_rule,
        "-w",
        str(output_pcap),
        "-F",
        "pcap",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"tshark failed: {e.stderr}")
        return False


def run_splitcap(input_pcap, output_dir):
    """使用 splitcap.exe 进行流切割"""
    command = ["mono", "SplitCap.exe", "-r", str(input_pcap), "-o", str(output_dir), "-s", "session", '-p','1000']
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"splitcap failed: {e.stderr}")
        return False


def get_pcap_files(directory):
    """递归获取目录下所有pcap文件"""
    pcap_files = []
    for ext in ["*.pcap", "*.pcapng", "*.cap"]:
        pcap_files.extend(Path(directory).rglob(ext))
    return pcap_files
