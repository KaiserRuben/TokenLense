"""
System information collection utilities for benchmarking.

This module provides functions to collect and format detailed system information
including CPU, GPU, memory, and PyTorch configurations.
"""

import os
import platform
import socket
import subprocess
import logging
from typing import Dict, Any

import torch

from src.benchmark.schema import SystemInfo

logger = logging.getLogger(__name__)

def collect_system_info() -> SystemInfo:
    """
    Collect system information including hardware and software details.

    Returns:
        SystemInfo: Object containing collected system information
    """
    info = SystemInfo()

    # Basic system info
    info.hostname = socket.gethostname()
    info.platform = platform.system()
    info.platform_version = platform.version()
    info.processor = platform.processor()

    # CPU info
    try:
        if info.platform == "Darwin":  # macOS
            # Get CPU model name
            cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            info.cpu_model = cpu_model

            # Get CPU cores
            cpu_cores = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
            info.cpu_cores = cpu_cores

            # Get RAM
            mem_total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            info.memory_total_gb = mem_total / (1024**3)  # Convert to GB

        elif info.platform == "Linux":
            # CPU model from /proc/cpuinfo
            cpu_info = subprocess.check_output(["cat", "/proc/cpuinfo"]).decode()
            model_name_line = [line for line in cpu_info.split('\n') if "model name" in line]
            if model_name_line:
                info.cpu_model = model_name_line[0].split(':')[1].strip()

            # CPU cores
            info.cpu_cores = os.cpu_count() or 0

            # Get RAM
            mem_info = subprocess.check_output(["cat", "/proc/meminfo"]).decode()
            mem_line = [line for line in mem_info.split('\n') if "MemTotal" in line]
            if mem_line:
                mem_kb = int(mem_line[0].split(':')[1].strip().split()[0])
                info.memory_total_gb = mem_kb / (1024**2)  # Convert KB to GB

        elif info.platform == "Windows":
            # On Windows, use wmic for CPU info
            cpu_model = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode()
            cpu_model_lines = cpu_model.strip().split('\n')
            if len(cpu_model_lines) > 1:
                info.cpu_model = cpu_model_lines[1].strip()

            # CPU cores
            info.cpu_cores = os.cpu_count() or 0

            # Get RAM
            mem_info = subprocess.check_output(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"]).decode()
            mem_lines = mem_info.strip().split('\n')
            if len(mem_lines) > 1:
                mem_bytes = int(mem_lines[1].strip())
                info.memory_total_gb = mem_bytes / (1024**3)  # Convert bytes to GB
    except Exception as e:
        logger.warning(f"Failed to collect detailed CPU information: {e}")
        info.cpu_model = platform.processor()
        info.cpu_cores = os.cpu_count() or 0

    # GPU information
    try:
        # CUDA
        info.torch_cuda_available = torch.cuda.is_available()
        if info.torch_cuda_available:
            info.cuda_version = torch.version.cuda or "Unknown"
            gpu_count = torch.cuda.device_count()
            gpu_models = []
            for i in range(gpu_count):
                gpu_models.append(torch.cuda.get_device_name(i))
            info.gpu_info = ", ".join(gpu_models)

        # MPS (Apple Silicon)
        info.torch_mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if info.torch_mps_available and not info.gpu_info:
            if info.platform == "Darwin":
                # Try to get Apple Silicon model
                try:
                    chip_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                    if "Apple" in chip_info:
                        info.gpu_info = f"Apple MPS ({chip_info})"
                    else:
                        info.gpu_info = "Apple MPS"
                except:
                    info.gpu_info = "Apple MPS"
    except Exception as e:
        logger.warning(f"Failed to collect GPU information: {e}")

    # PyTorch version
    info.torch_version = torch.__version__

    return info

def format_system_info(system_info: SystemInfo) -> str:
    """
    Format system information for display in summary report.

    Args:
        system_info: SystemInfo object

    Returns:
        Formatted string with system information
    """
    lines = [
        f"System Information:",
        f"  Platform: {system_info.platform} ({system_info.platform_version})",
        f"  CPU: {system_info.cpu_model or system_info.processor}",
        f"  CPU Cores: {system_info.cpu_cores}",
        f"  Memory: {system_info.memory_total_gb:.2f} GB",
    ]

    # GPU information
    if system_info.torch_cuda_available:
        lines.append(f"  GPU: {system_info.gpu_info} (CUDA {system_info.cuda_version})")
    elif system_info.torch_mps_available:
        lines.append(f"  GPU: {system_info.gpu_info} (Apple MPS)")
    else:
        lines.append(f"  GPU: None")

    # PyTorch version
    lines.append(f"  PyTorch: {system_info.torch_version}")

    # Device availability
    devices = []
    if system_info.torch_cuda_available:
        devices.append("CUDA")
    if system_info.torch_mps_available:
        devices.append("MPS")
    if not devices:
        devices.append("CPU only")

    lines.append(f"  Available Devices: {', '.join(devices)}")

    return "\n".join(lines)