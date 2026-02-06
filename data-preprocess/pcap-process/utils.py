import logging
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

def setup_logger(name, log_file=None):
    """Set up the logger"""
    # 1. Get root logger
    # This configuration will apply to all modules importing logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicate log entries
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    if log_file:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Return a named logger for main.py, though root is already configured
    return logging.getLogger(name)

def clean_directory(dir_path):
    """Clear directory content"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def run_tshark(input_pcap, output_pcap, filter_rule="tcp or udp"):
    """Use tshark to filter pcap files (e.g., removing ICMP and DNS)"""
    command = [
        "tshark",
        "-r",
        str(input_pcap),
        "-Y",
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
    """Use splitcap.exe (via Mono) for session-based flow splitting"""
    command = ["mono", "SplitCap.exe", "-r", str(input_pcap), "-o", str(output_dir), "-s", "session", '-p','1000']
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"splitcap failed: {e.stderr}")
        return False


def get_pcap_files(directory):
    """Recursively retrieve all PCAP files in the directory"""
    pcap_files = []
    for ext in ["*.pcap", "*.pcapng", "*.cap"]:
        pcap_files.extend(Path(directory).rglob(ext))
    return pcap_files