#!/usr/bin/env python3
# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Isa Wazirzada

# ───────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
try:
    import setproctitle
    SETPROCTITLE_AVAILABLE = True
except ImportError:
    SETPROCTITLE_AVAILABLE = False
# Auto-detect and set ROCM_PATH if not already configured
if "ROCM_PATH" not in os.environ:
    # Try common ROCm installation paths
    rocm_paths = [
        "/opt/rocm",  # Standard symlink
        "/opt/rocm-6.1.0",
        "/opt/rocm-6.0.0",
        "/usr/lib64/rocm",
    ]
    # Also check for versioned installs
    try:
        import glob
        versioned = sorted(glob.glob("/opt/rocm-*"), reverse=True)
        rocm_paths = ["/opt/rocm"] + versioned + rocm_paths
    except Exception:
        pass
    
    for path in rocm_paths:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "lib", "libamd_smi.so")):
            os.environ["ROCM_PATH"] = path
            break
    else:
        # Fallback
        os.environ["ROCM_PATH"] = "/opt/rocm"

# ───────────────────────────────────────────────────────────────────────
# PRE-TORCH CPU THREADING SETUP  ───────────────────────────────────────
# Set OpenMP environment BEFORE importing torch for optimal CPU performance.
# These must be set before the OpenMP runtime initializes.
def _get_physical_core_count_early() -> int:
    """Quick physical core detection before torch import."""
    try:
        # Linux: count unique (package, core) pairs
        cores_seen = set()
        cpu_dir = "/sys/devices/system/cpu"
        if os.path.isdir(cpu_dir):
            for entry in os.listdir(cpu_dir):
                if entry.startswith("cpu") and entry[3:].isdigit():
                    topo = os.path.join(cpu_dir, entry, "topology", "core_id")
                    pkg = os.path.join(cpu_dir, entry, "topology", "physical_package_id")
                    if os.path.exists(topo) and os.path.exists(pkg):
                        with open(topo) as f:
                            core_id = f.read().strip()
                        with open(pkg) as f:
                            package_id = f.read().strip()
                        cores_seen.add((package_id, core_id))
            if cores_seen:
                return len(cores_seen)
    except Exception:
        pass
    # macOS
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                    capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        except Exception:
            pass
    # Fallback: assume 2-way SMT
    return max(1, (os.cpu_count() or 1) // 2)

# Set OMP environment early (before torch import) for CPU-only systems
# These are no-ops if GPU is later detected
if "OMP_NUM_THREADS" not in os.environ:
    _early_cores = _get_physical_core_count_early()
    os.environ["OMP_NUM_THREADS"] = str(_early_cores)
    os.environ["OMP_PROC_BIND"] = "spread"
    os.environ["OMP_PLACES"] = "cores"

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import multiprocessing
import threading
import torch

# ───────────────────────────────────────────────────────────────────────
# ASCII BANNER  ─────────────────────────────────────────────────────────
TORCH_HAMMER_BANNER = r'''
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║      🔥  ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗  🔥                   ║
║          ╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║                       ║
║     ░░░░    ██║   ██║   ██║██████╔╝██║     ███████║    ░░░░               ║
║    ░░░░░    ██║   ██║   ██║██╔══██╗██║     ██╔══██║   ░░░░░               ║
║     ░░░░    ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║    ░░░░               ║
║             ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝                       ║
║                                                                           ║
║         ██╗  ██╗ █████╗ ███╗   ███╗███╗   ███╗███████╗██████╗  🔨         ║
║         ██║  ██║██╔══██╗████╗ ████║████╗ ████║██╔════╝██╔══██╗            ║
║         ███████║███████║██╔████╔██║██╔████╔██║█████╗  ██████╔╝            ║
║         ██╔══██║██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══╝  ██╔══██╗   ⚡       ║
║         ██║  ██║██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║███████╗██║  ██║            ║
║         ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝            ║
║                                                                           ║
║            ▄▄▄▄▄▄▄▄▄▄▄   Forged with PyTorch   ▄▄▄▄▄▄▄▄▄▄▄                ║
║           ═══════════════════════════════════════════════════             ║
║                      GPU/CPU/APU Micro-Benchmark Suite                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
'''

# Easter egg quotes for --forge mode
FORGE_QUOTES = [
    "Strike while the GPU is hot! 🔥",
    "Forging tensors in the fires of compute...",
    "The hammer falls, the benchmarks rise.",
    "In the forge of silicon, performance is born.",
    "Every GPU has its breaking point.",
    "Tempering your hardware with mathematical fury.",
    "The anvil of compute awaits your workload.",
    "From raw silicon to refined performance.",
    "Heat, pressure, tensors. The recipe for truth.",
    "Your GPU called. It wants a challenge.",
    "Benchmarking: because 'it feels fast' isn't a metric.",
    "May your thermals be cool and your FLOPS be high.",
    "The forge remembers every tensor it has shaped.",
    "PyTorch + Hammer = Truth about your hardware.",
]

# ANSI color codes for animated banner
ANSI_COLORS = {
    'red': '\033[91m',
    'orange': '\033[38;5;208m',
    'yellow': '\033[93m',
    'white': '\033[97m',
    'reset': '\033[0m',
    'bold': '\033[1m',
    'dim': '\033[2m',
}

def print_banner():
    """Print the Torch Hammer ASCII banner."""
    print(TORCH_HAMMER_BANNER)

def print_forge_banner():
    """Easter egg: Animated banner with fire effect and random quote."""
    import sys
    
    # Check if terminal supports colors
    is_tty = sys.stdout.isatty()
    
    # Fire gradient frames for animation
    fire_frames = [
        ['red', 'orange', 'yellow', 'white'],
        ['orange', 'yellow', 'white', 'yellow'],
        ['yellow', 'white', 'yellow', 'orange'],
        ['white', 'yellow', 'orange', 'red'],
        ['yellow', 'orange', 'red', 'orange'],
        ['orange', 'red', 'orange', 'yellow'],
    ]
    
    banner_lines = TORCH_HAMMER_BANNER.strip().split('\n')
    
    if is_tty:
        # Animated version
        try:
            for frame_idx in range(12):  # 2 full cycles
                colors = fire_frames[frame_idx % len(fire_frames)]
                sys.stdout.write('\033[H\033[J')  # Clear screen
                
                for i, line in enumerate(banner_lines):
                    # Color based on position (top = hottest)
                    color_idx = min(i // 6, len(colors) - 1)
                    color = ANSI_COLORS.get(colors[color_idx], '')
                    reset = ANSI_COLORS['reset']
                    
                    # Make fire emojis and special chars extra bright
                    if is_tty:
                        line = line.replace('🔥', f"{ANSI_COLORS['bold']}🔥{reset}{color}")
                        line = line.replace('🔨', f"{ANSI_COLORS['bold']}🔨{reset}{color}")
                        line = line.replace('⚡', f"{ANSI_COLORS['yellow']}⚡{reset}{color}")
                    
                    print(f"{color}{line}{reset}")
                
                time.sleep(0.15)
            
            # Final static frame with quote
            sys.stdout.write('\033[H\033[J')  # Clear one more time
            color = ANSI_COLORS['orange']
            reset = ANSI_COLORS['reset']
            bold = ANSI_COLORS['bold']
            
            for line in banner_lines:
                print(f"{color}{line}{reset}")
            
            # Random quote
            quote = random.choice(FORGE_QUOTES)
            print()
            print(f"{bold}  ⚒️  {quote}{reset}")
            print()
            
        except (KeyboardInterrupt, Exception):
            # Fallback to static if animation fails
            print(TORCH_HAMMER_BANNER)
            quote = random.choice(FORGE_QUOTES)
            print(f"\n  ⚒️  {quote}\n")
    else:
        # Non-TTY: static banner with quote
        print(TORCH_HAMMER_BANNER)
        quote = random.choice(FORGE_QUOTES)
        print(f"\n  ⚒️  {quote}\n")

# ───────────────────────────────────────────────────────────────────────
# HARDWARE BASELINES  ─────────────────────────────────────────
# Optional: Hardware performance baselines for validation
# By default, validation is DISABLED. To enable:
#   1. Use --baseline-file to load custom baselines from JSON/YAML
#   2. See baselines/ directory for example files
# This empty dict ensures validation is opt-in, not mandatory
HARDWARE_BASELINES = {}

def load_hardware_baselines(filepath: str) -> Dict[str, Any]:
    """Load hardware baselines from external JSON or YAML file."""
    try:
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            if not YAML_AVAILABLE:
                print(f"Warning: Cannot load {filepath} - PyYAML not installed")
                return {}
            import yaml
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        elif filepath.endswith('.json'):
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Unknown baseline file format: {filepath}")
            return {}
    except Exception as e:
        print(f"Warning: Could not load baseline file {filepath}: {e}")
        return {}


def validate_performance(model_name: str, benchmark: str, dtype: str, measured_value: float, 
                        unit: str = "gflops", tf32_mode: bool = False, 
                        logger = None,
                        efficiency_warn_pct: float = 70.0, 
                        baselines: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate measured performance against hardware baselines for ALL benchmarks.
    
    Supports two baseline formats:
    
    1. NEW (target-based):
       benchmarks:
         batched_gemm:
           float32:
             target_gflops: 49000.0
             min_efficiency: 90.0
    
    2. OLD (theoretical peak):
       fp32_tflops: 66.9
       memory_bandwidth_gbps: 4800.0
    
    NEW format takes precedence. Falls back to OLD for backward compatibility.
    
    Returns validation result with warnings if performance is unexpectedly low.
    Optional validation - gracefully skips if baselines not available.
    """
    result = {
        'valid': True,
        'baseline': None,
        'expected': None,
        'efficiency': None,
        'warning': None
    }
    
    # Use provided baselines or fall back to built-in
    baseline_db = baselines if baselines is not None else HARDWARE_BASELINES
    
    # Skip validation if no baselines available
    if not baseline_db:
        return result
    
    # Find matching baseline
    baseline = None
    for known_model, specs in baseline_db.items():
        if known_model in model_name:
            baseline = specs
            break
    
    if not baseline:
        return result  # No baseline available for this hardware
    
    result['baseline'] = baseline
    
    # Normalize benchmark and dtype names for lookup
    benchmark_normalized = benchmark.lower().replace(' ', '_').replace('-', '_')
    dtype_normalized = dtype.replace('torch.', '')
    
    # TRY NEW FORMAT FIRST: benchmarks.<name>.<dtype>.target_*
    benchmarks_section = baseline.get('benchmarks', {})
    if benchmarks_section and benchmark_normalized in benchmarks_section:
        bench_config = benchmarks_section[benchmark_normalized]
        
        # Handle TF32 mode (special case - uses float32 tensor but different target)
        dtype_lookup = 'tf32' if tf32_mode else dtype_normalized
        
        if dtype_lookup in bench_config:
            dtype_config = bench_config[dtype_lookup]
            
            # Get target value based on unit
            target = None
            if unit.lower() in ['gflop/s', 'gflops']:
                target = dtype_config.get('target_gflops')
            elif unit.lower() in ['gb/s', 'gbps']:
                target = dtype_config.get('target_gbps')
            elif unit.lower() in ['mlup/s', 'mlups']:
                target = dtype_config.get('target_mlups')
            elif unit.lower() in ['img/s', 'images/s']:
                target = dtype_config.get('target_imgps')
            
            if target:
                result['expected'] = target
                efficiency = (measured_value / target) * 100.0
                result['efficiency'] = efficiency
                
                # Get benchmark-specific threshold or fall back to default
                threshold = dtype_config.get('min_efficiency', efficiency_warn_pct)
                
                # Validate
                if efficiency < threshold:
                    result['valid'] = False
                    result['warning'] = f"[WARN] Performance below target: {efficiency:.1f}% of expected {target:.1f} {unit} (threshold: {threshold:.0f}%)"
                    if logger:
                        logger.warning(f"Performance below target: {measured_value:.1f} {unit} vs {target:.1f} {unit} expected ({efficiency:.1f}%)")
                elif efficiency >= 100.0:
                    if logger:
                        logger.info(f"✅ Excellent performance: {efficiency:.1f}% of target ({target:.1f} {unit})")
                elif efficiency >= threshold:
                    if logger:
                        logger.info(f"✅ Good performance: {efficiency:.1f}% of target ({target:.1f} {unit})")
                
                return result  # NEW format handled successfully
    
    # FALLBACK TO OLD FORMAT: fp32_tflops, memory_bandwidth_gbps
    benchmark_upper = benchmark.upper()
    
    # Determine expected performance based on benchmark and dtype
    
    # COMPUTE-BOUND TESTS: Validate against TFLOPS
    if benchmark_upper in ['BATCHED GEMM', 'GEMM', 'CONVOLUTION', 'EINSUM ATTENTION', '3D FFT']:
        if tf32_mode and 'tf32_tflops' in baseline:
            expected_tflops = baseline['tf32_tflops']
            dtype_key = 'TF32'
        elif dtype in ['float32', 'torch.float32']:
            expected_tflops = baseline.get('fp32_tflops', None)
            dtype_key = 'FP32'
        elif dtype in ['float64', 'torch.float64']:
            expected_tflops = baseline.get('fp64_tflops', None)
            dtype_key = 'FP64'
        elif dtype in ['float16', 'torch.float16', 'bfloat16', 'torch.bfloat16']:
            expected_tflops = baseline.get('fp16_tflops', None)
            dtype_key = 'FP16'
        elif dtype in ['complex64', 'torch.complex64']:
            # Complex64 uses 2x float32 ops, similar throughput to FP32
            expected_tflops = baseline.get('fp32_tflops', None)
            dtype_key = 'Complex64'
        elif dtype in ['complex128', 'torch.complex128']:
            # Complex128 uses 2x float64 ops
            expected_tflops = baseline.get('fp64_tflops', None)
            dtype_key = 'Complex128'
        else:
            return result
        
        if expected_tflops:
            result['expected'] = expected_tflops
            measured_tflops = measured_value / 1000.0  # Convert GFLOPS to TFLOPS
            efficiency = (measured_tflops / expected_tflops) * 100.0
            result['efficiency'] = efficiency
            
            # Different efficiency thresholds for different benchmarks
            if benchmark_upper in ['CONVOLUTION', '3D FFT']:
                # Conv/FFT can be slightly lower due to memory access patterns
                threshold = efficiency_warn_pct * 0.85  # 85% of threshold (e.g., 60% instead of 70%)
            else:
                threshold = efficiency_warn_pct
            
            # Warn if efficiency is below threshold
            if efficiency < threshold:
                result['valid'] = False
                result['warning'] = f"[WARN] Low efficiency: {efficiency:.1f}% of peak {dtype_key} ({expected_tflops:.1f} TFLOPS)"
                if logger:
                    logger.warning(f"Performance below expected: {measured_tflops:.1f} TFLOPS vs {expected_tflops:.1f} TFLOPS peak ({efficiency:.1f}%)")
            elif efficiency > 95.0:
                if logger:
                    logger.info(f"Excellent performance: {efficiency:.1f}% of peak {dtype_key}")
    
    # MEMORY-BOUND TESTS: Validate against memory bandwidth (GB/s)
    elif benchmark_upper in ['MEMORY TRAFFIC', 'HEAT EQUATION', 'SCHRÖDINGER EQUATION']:
        expected_bandwidth = baseline.get('memory_bandwidth_gbps', None)
        
        if expected_bandwidth:
            result['expected'] = expected_bandwidth
            # measured_value is already in GB/s for these benchmarks
            efficiency = (measured_value / expected_bandwidth) * 100.0
            result['efficiency'] = efficiency
            
            # Memory-bound codes typically achieve 60-90% of peak BW
            mem_threshold = efficiency_warn_pct * 0.7  # Lower threshold for memory tests (e.g., 50% instead of 70%)
            
            if efficiency < mem_threshold:
                result['valid'] = False
                result['warning'] = f"[WARN] Low memory efficiency: {efficiency:.1f}% of peak BW ({expected_bandwidth:.1f} GB/s)"
                if logger:
                    logger.warning(f"Memory bandwidth below expected: {measured_value:.1f} GB/s vs {expected_bandwidth:.1f} GB/s peak ({efficiency:.1f}%)")
            elif efficiency > 85.0:
                if logger:
                    logger.info(f"Excellent memory bandwidth: {efficiency:.1f}% of peak")
    
    return result

# ───────────────────────────────────────────────────────────────────────
# 1.  TELEMETRY  ────────────────────────────────────────────────────────
class TelemetryBase:
    """Abstract base; subclasses fill `supported` and `read()`"""

    supported: List[str] = ["vendor", "model", "device_id"]

    def read(self) -> Dict[str, Any]:
        raise NotImplementedError

    def schema(self) -> List[str]:
        return self.supported
    
    def get_stats(self, skip_first_n=0) -> Dict[str, Any]:
        """Return min/max statistics collected during benchmark."""
        return {}
    
    def reset_stats(self):
        """Clear accumulated statistics to start fresh."""
        pass

    def shutdown(self):
        pass


# ── NVIDIA ‑ NVML ─────────────────────────────────────────────────────
class NVMLTelemetry(TelemetryBase):
    supported = [
        "hostname", "vendor", "model", "device_id", "serial",
        "sm_util", "mem_bw_util", "mem_util",
        "gpu_clock", "mem_clock", "vbst_sync",
        "power_W", "temp_gpu_C", "temp_hbm_C",
        "mem_used_MB", "mem_total_MB", "mem_free_MB",
        "hw_slowdown", "sw_slowdown", "power_limit", "throttled",
    ]
    def __init__(self, index):
        import socket
        try:
            import pynvml as nv
            nv.nvmlInit()
        except ImportError:
            print("The pynvml package is missing, exiting...")
            sys.exit(1)
        # Nice trick from
        # https://stackoverflow.com/questions/30185706/what-is-the-correct-way-of-getting-a-base-or-short-hostname
        self.hostname = socket.gethostname().split('.', 1)[0]
        self.nv = nv
        self.h = nv.nvmlDeviceGetHandleByIndex(index)
        self.idx = index
        self._model = nv.nvmlDeviceGetName(self.h)
        self.hbm_temp_available = False
        if hasattr(self.nv, "NVML_TEMPERATURE_SENSOR_MEMORY"):
            self.hbm_temp_available = True
        # Min/max/mean tracking for telemetry
        self.readings = {
            'temp_gpu_C': [], 'temp_hbm_C': [], 'power_W': [],
            'gpu_clock': [], 'sm_util': [], 'mem_bw_util': [],
            'mem_used_MB': [], 'vbst_sync': []
        }
        # Track if throttling occurred at any point
        self.throttle_detected = False
        self.hw_slowdown_count = 0
        self.sw_slowdown_count = 0
        self.power_limit_count = 0
        self.max_temp_hbm = 0  # Initialize HBM temp tracking
        # Track max power limit and ECC errors
        try:
            self.power_limit_W = nv.nvmlDeviceGetPowerManagementLimit(self.h) / 1000.0
        except Exception:
            self.power_limit_W = None  # Not supported on this GPU
        self.ecc_errors_initial = self._get_ecc_errors()
        self.ecc_errors_current = 0

    def read(self):
        nv, h = self.nv, self.h
        gpu_clock = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
        power_W = nv.nvmlDeviceGetPowerUsage(h) / 1e3
        temp_gpu_C = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        util_rates = nv.nvmlDeviceGetUtilizationRates(h)  # Call once, reuse
        
        # Read VBST (VBOOST sync status)
        vbst_sync = "N/A"
        try:
            vbst_sync = nv.nvmlDeviceGetSyncBoostParts(h)
        except Exception:
            pass  # VBST not available on all GPU models
        
        d = {
            "vendor": "NVIDIA",
            "model": self._model,
            "device_id": self.idx,
            "hostname": self.hostname,
            "serial": nv.nvmlDeviceGetSerial(h),
            "sm_util": util_rates.gpu,
            "mem_bw_util": util_rates.memory,
            "gpu_clock": gpu_clock,
            "mem_clock": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            "vbst_sync": vbst_sync,
            "power_W": power_W,
            "temp_gpu_C": temp_gpu_C,
        }
        
        # HBM temperature - always include field for CSV alignment
        if self.hbm_temp_available:
            try:
                temp_hbm = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_SENSOR_MEMORY)
                d["temp_hbm_C"] = temp_hbm
                self.max_temp_hbm = max(self.max_temp_hbm, temp_hbm)
            except Exception:
                d["temp_hbm_C"] = "N/A"  # Sensor query failed
        else:
            d["temp_hbm_C"] = "N/A"
        
        # Memory usage tracking
        try:
            mem_info = nv.nvmlDeviceGetMemoryInfo(h)
            d["mem_used_MB"] = mem_info.used / 1024**2
            d["mem_total_MB"] = mem_info.total / 1024**2
            d["mem_free_MB"] = mem_info.free / 1024**2
        except Exception as e:
            logging.debug(f"GPU {self.idx} memory read failed: {e}")
            d["mem_used_MB"] = "N/A"
            d["mem_total_MB"] = "N/A"
            d["mem_free_MB"] = "N/A"
        
        # Collect readings for statistics
        self.readings['temp_gpu_C'].append(temp_gpu_C)
        self.readings['power_W'].append(power_W)
        self.readings['gpu_clock'].append(gpu_clock)
        self.readings['sm_util'].append(d['sm_util'])
        self.readings['mem_bw_util'].append(d['mem_bw_util'])
        if d['temp_hbm_C'] != 'N/A':
            self.readings['temp_hbm_C'].append(d['temp_hbm_C'])
        if d['mem_used_MB'] != 'N/A':
            self.readings['mem_used_MB'].append(d['mem_used_MB'])
        if vbst_sync != 'N/A':
            self.readings['vbst_sync'].append(vbst_sync)
        
        # Check for throttling - full reasons except SW Power Cap
        try:
            reasons = nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            hw_slowdown = bool(reasons & nv.nvmlClocksThrottleReasonHwSlowdown)
            sw_slowdown = bool(reasons & nv.nvmlClocksThrottleReasonSwThermalSlowdown)
            power_limit = bool(reasons & nv.nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
            d["hw_slowdown"] = 1 if hw_slowdown else 0
            d["sw_slowdown"] = 1 if sw_slowdown else 0
            d["power_limit"] = 1 if power_limit else 0
            d["throttled"] = 1 if (hw_slowdown or sw_slowdown or power_limit) else 0
            
            # Track throttling events
            if hw_slowdown or sw_slowdown or power_limit:
                self.throttle_detected = True
                if hw_slowdown:
                    self.hw_slowdown_count += 1
                if sw_slowdown:
                    self.sw_slowdown_count += 1
                if power_limit:
                    self.power_limit_count += 1
        except Exception as e:
            logging.debug(f"GPU {self.idx} throttle query failed: {e}")
            d["hw_slowdown"] = 0
            d["sw_slowdown"] = 0
            d["power_limit"] = 0
            d["throttled"] = 0

        return d

    def get_stats(self, skip_first_n=0):
        """Return min/mean/max statistics collected during benchmark.
        
        Args:
            skip_first_n: Skip first N readings when calculating mean (to avoid warmup skew)
        """
        stats = {}
        for key, vals in self.readings.items():
            if vals:
                # Use all values for min/max, skip first N for mean
                stats[f"{key}_min"] = min(vals)
                vals_for_mean = vals[skip_first_n:] if len(vals) > skip_first_n else vals
                stats[f"{key}_mean"] = statistics.mean(vals_for_mean)
                stats[f"{key}_max"] = max(vals)
        return stats
    
    def reset_stats(self):
        """Clear accumulated statistics to start fresh."""
        for key in self.readings:
            self.readings[key] = []
        self.throttle_detected = False
        self.hw_slowdown_count = 0
        self.sw_slowdown_count = 0
        self.power_limit_count = 0
        self.ecc_errors_initial = self._get_ecc_errors()
    
    def _get_ecc_errors(self):
        """Get total ECC errors (single-bit + double-bit, volatile + aggregate)."""
        try:
            nv, h = self.nv, self.h
            sbe_volatile = nv.nvmlDeviceGetTotalEccErrors(h, nv.NVML_MEMORY_ERROR_TYPE_CORRECTED, nv.NVML_VOLATILE_ECC)
            dbe_volatile = nv.nvmlDeviceGetTotalEccErrors(h, nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nv.NVML_VOLATILE_ECC)
            return sbe_volatile + dbe_volatile
        except Exception:
            return 0
    
    def check_thermal_warnings(self, log=None, temp_warn=90.0, temp_critical=95.0, power_warn_pct=98.0) -> Dict[str, Any]:
        """Check for thermal or power warnings. Returns dict with warnings.
        
        Args:
            log: Logger instance for warnings
            temp_warn: Temperature warning threshold in Celsius
            temp_critical: Temperature critical threshold in Celsius  
            power_warn_pct: Power limit warning threshold in percent
        """
        warnings = {'thermal': [], 'power': [], 'ecc': []}
        
        # Check HBM temperature limits (NVIDIA)
        if 'temp_hbm_C_max' in self.get_stats():
            hbm_max = self.get_stats()['temp_hbm_C_max']
            if hbm_max >= temp_critical:
                msg = f"[CRIT] HBM temperature reached {hbm_max}°C (limit: {temp_critical}°C)"
                warnings['thermal'].append(msg)
                if log:
                    log.error(msg)
            elif hbm_max >= temp_warn:
                msg = f"[WARN] HBM temperature reached {hbm_max}°C (approaching {temp_critical}°C limit)"
                warnings['thermal'].append(msg)
                if log:
                    log.warning(msg)
        
        # Check GPU temperature (fallback for non-HBM or other telemetry)
        if 'temp_gpu_C_max' in self.get_stats():
            gpu_max = self.get_stats()['temp_gpu_C_max']
            if gpu_max >= temp_critical:
                msg = f"[CRIT] GPU temperature reached {gpu_max}°C (limit: {temp_critical}°C)"
                warnings['thermal'].append(msg)
                if log:
                    log.error(msg)
            elif gpu_max >= temp_warn:
                msg = f"[WARN] GPU temperature reached {gpu_max}°C (approaching {temp_critical}°C limit)"
                warnings['thermal'].append(msg)
                if log:
                    log.warning(msg)
        
        # Check power limit proximity
        if self.power_limit_W and 'power_W_max' in self.get_stats():
            power_max = self.get_stats()['power_W_max']
            power_pct = (power_max / self.power_limit_W) * 100
            if power_pct >= power_warn_pct:
                msg = f"[WARN] Power limit reached: {power_max:.0f}W / {self.power_limit_W:.0f}W ({power_pct:.1f}%)"
                warnings['power'].append(msg)
                if log:
                    log.warning(msg)
        
        # Check for new ECC errors (NVIDIA)
        current_ecc = self._get_ecc_errors()
        new_ecc = current_ecc - self.ecc_errors_initial
        if new_ecc > 0:
            msg = f"[WARN] Detected {new_ecc} new ECC error(s) during benchmark"
            warnings['ecc'].append(msg)
            if log:
                log.warning(msg)
        self.ecc_errors_current = new_ecc
        
        return warnings
    
    def shutdown(self):
        self.nv.nvmlShutdown()


# ── AMD ROCm (rocm-smi) ───────────────────────────────────────────────
class RocmTelemetry(TelemetryBase):
    GPU_FIELDS = [
        "vendor", "model", "device_id", "serial", "sm_util", "mem_util",
        "gpu_clock", "mem_clock", "power_W", "temp_gpu_C",
        "mem_used_MB", "mem_total_MB", "mem_free_MB", "throttled"
    ]
    CPU_FIELDS = [
        "vendor", "model", "device_id", "serial", "cpu_power_W", "soc_temp_C"
    ]

    def __init__(self, index):
        import socket
        import amdsmi
        from amdsmi.amdsmi_exception import AmdSmiException
        from amdsmi import AmdSmiInitFlags
        os.environ["AMDSMI_GPU_METRICS_CACHE_MS"] = str(200)
        self.hostname = socket.gethostname().split('.', 1)[0]
        self.amdsmi = amdsmi  # Store module reference for use in read()
        self.idx = index  # Store PyTorch device index for correct GPU labeling
        amdsmi.amdsmi_init()
        self.handles = amdsmi.amdsmi_get_processor_handles()
        self.handle = self.handles[index]
        board_info = amdsmi.amdsmi_get_gpu_board_info(self.handle)
        # Try asic_info if board_info is empty
        try:
            asic_info = amdsmi.amdsmi_get_gpu_asic_info(self.handle)
            self._model = (board_info.get("product_name") or
                           board_info.get("model_number") or
                           asic_info.get("market_name") or
                           asic_info.get("asic_serial"))
        except Exception:
            asic_info = {}
            self._model = (board_info.get("product_name") or
                           board_info.get("model_number"))
        
        # Get serial number - try multiple sources
        # Helper to check if value is valid (not empty, not 'N/A')
        def valid_serial(val):
            return val and val != 'N/A' and val.strip()
        
        self._serial = None
        # Try board_info first
        for key in ["product_serial", "serial_number", "serial", "fru_id"]:
            if valid_serial(board_info.get(key)):
                self._serial = board_info[key]
                break
        # Try asic_info (asic_serial is usually the best source on MI300A)
        if not self._serial:
            for key in ["asic_serial", "serial"]:
                if valid_serial(asic_info.get(key)):
                    self._serial = asic_info[key]
                    break
        # Try UUID as fallback
        if not self._serial:
            try:
                uuid = amdsmi.amdsmi_get_gpu_device_uuid(self.handle)
                if valid_serial(uuid):
                    self._serial = uuid
            except Exception:
                pass
        # Try VBIOS serial
        if not self._serial:
            try:
                vbios_info = amdsmi.amdsmi_get_gpu_vbios_info(self.handle)
                for key in ["serial", "part_number"]:
                    if valid_serial(vbios_info.get(key)):
                        self._serial = vbios_info[key]
                        break
            except Exception:
                pass
        if not self._serial:
            self._serial = "N/A"
        self._ptype = amdsmi.amdsmi_get_processor_type(self.handle)
        self._is_gpu = isinstance(self._ptype, dict) and self._ptype.get('processor_type') == 'AMD_GPU'
        
        # If processor_type detection failed, try to detect GPU by calling a GPU-specific API
        if not self._is_gpu:
            try:
                # Try to get GPU metrics - this only works on actual GPUs
                test_metrics = amdsmi.amdsmi_get_gpu_metrics_info(self.handle)
                if test_metrics and isinstance(test_metrics, dict):
                    self._is_gpu = True
                    logging.debug(f"AMD device {index}: Detected as GPU via metrics API")
            except Exception:
                pass  # Not a GPU or API not available
        
        logging.debug(f"AMD device {index}: ptype={self._ptype}, is_gpu={self._is_gpu}, model={self._model}")
        self.supported = self.GPU_FIELDS if self._is_gpu else self.CPU_FIELDS
        # Telemetry tracking
        self.readings = {
            'temp_gpu_C': [], 'power_W': [], 'gpu_clock': [],
            'sm_util': [], 'mem_util': [], 'mem_used_MB': []
        }
        # Track if throttling occurred at any point
        self.throttle_detected = False
        self.throttle_count = 0
        self.power_throttle_count = 0
        self.thermal_throttle_count = 0
        # Track if we've warned about unsupported CPU/APU metrics (warn once, not every read)
        self._cpu_power_warned = False
        self._cpu_temp_warned = False
        
        # Try to get power limit for AMD GPU
        if self._is_gpu:
            try:
                power_cap = amdsmi.amdsmi_get_gpu_power_cap(self.handle)
                # power_cap returns dict like {'power_cap': <watts>}
                self.power_limit_W = power_cap.get('power_cap', None)
                if self.power_limit_W:
                    self.power_limit_W = self.power_limit_W  # Already in watts
            except Exception as e:
                logging.debug(f"AMD GPU {index} power cap query failed: {e}")
                self.power_limit_W = None
        else:
            self.power_limit_W = None

    def read(self):
        if self._is_gpu:  # GPU
            # Use gpu_metrics_info for comprehensive telemetry in one call
            try:
                metrics = self.amdsmi.amdsmi_get_gpu_metrics_info(self.handle)
            except Exception as e:
                logging.warning(f"AMD GPU metrics read failed: {e}")
                metrics = {}
            
            # GPU Utilization (from metrics)
            util = metrics.get('average_gfx_activity', 'N/A')
            
            # Memory Utilization (from metrics)
            mem_util = metrics.get('average_umc_activity', 'N/A')
            
            # GPU Clock (from metrics, already in MHz)
            gpu_clk = metrics.get('current_gfxclk', metrics.get('average_gfxclk_frequency', 'N/A'))
            if gpu_clk == 'N/A' or gpu_clk == 0:
                gpu_clk = "N/A"
            
            # Memory Clock (from metrics, already in MHz)
            mem_clk = metrics.get('current_uclk', metrics.get('average_uclk_frequency', 'N/A'))
            if mem_clk == 'N/A' or mem_clk == 0:
                mem_clk = "N/A"
            
            # Power — try current first (MI300), then average (MI250).
            # Keys may exist with value 'N/A' or 0; treat both as missing.
            power = metrics.get('current_socket_power')
            if power in (None, 'N/A', 0):
                power = metrics.get('average_socket_power')
            if power in (None, 'N/A', 0):
                power = "N/A"
            
            # Temperature (from metrics, try multiple sources)
            temp = metrics.get('temperature_edge', 'N/A')
            if temp == 'N/A' or temp is None:
                temp = metrics.get('temperature_hotspot', 'N/A')
            if temp == 'N/A' or temp is None:
                temp = metrics.get('temperature_mem', 'N/A')
            
            # Memory usage (still need separate API calls)
            try:
                mem_total = self.amdsmi.amdsmi_get_gpu_memory_total(self.handle, self.amdsmi.AmdSmiMemoryType.VRAM)
                mem_used = self.amdsmi.amdsmi_get_gpu_memory_usage(self.handle, self.amdsmi.AmdSmiMemoryType.VRAM)
                mem_total_MB = mem_total / 1024**2
                mem_used_MB = mem_used / 1024**2
                mem_free_MB = mem_total_MB - mem_used_MB
            except Exception as e:
                logging.warning(f"AMD memory usage read failed: {e}")
                mem_used_MB = mem_total_MB = mem_free_MB = "N/A"
            
            # AMD throttle detection (from metrics dict we already fetched)
            # throttle_status is a bitmask - decode different throttle reasons
            throttled = 0
            throttle_reason = ""
            try:
                throttle_status = metrics.get('throttle_status', 0)
                if throttle_status != 0:
                    throttled = 1
                    self.throttle_detected = True
                    self.throttle_count += 1
                    
                    # Decode throttle_status bitmask (AMD ROCm SMI definitions)
                    # These are common throttle reason bits - actual values may vary by ASIC
                    reasons = []
                    if throttle_status & 0x1:  # Power cap
                        reasons.append("POWER")
                        self.power_throttle_count = getattr(self, 'power_throttle_count', 0) + 1
                    if throttle_status & 0x2:  # Thermal limit
                        reasons.append("THERMAL")
                        self.thermal_throttle_count = getattr(self, 'thermal_throttle_count', 0) + 1
                    if throttle_status & 0x4:  # Current limit
                        reasons.append("CURRENT")
                    if not reasons:
                        reasons.append(f"0x{throttle_status:x}")  # Unknown bits, show hex
                    
                    throttle_reason = ",".join(reasons)
                    logging.debug(f"AMD GPU throttle: status=0x{throttle_status:x} ({throttle_reason})")
            except Exception as e:
                logging.debug(f"AMD GPU throttle check failed: {e}")
                throttled = 0
            
            # Collect readings for statistics
            if util != 'N/A':
                self.readings['sm_util'].append(util)
            if mem_util != 'N/A':
                self.readings['mem_util'].append(mem_util)
            if gpu_clk != 'N/A':
                self.readings['gpu_clock'].append(gpu_clk)
            if power != 'N/A':
                self.readings['power_W'].append(power)
            if temp != 'N/A':
                self.readings['temp_gpu_C'].append(temp)
            if mem_used_MB != 'N/A':
                self.readings['mem_used_MB'].append(mem_used_MB)
            
            return {
                "vendor": "AMD",
                "model": self._model,
                "device_id": self.idx,
                "hostname": self.hostname,
                "serial": self._serial,
                "sm_util": util,
                "mem_util": mem_util,
                "gpu_clock": gpu_clk,
                "mem_clock": mem_clk,
                "power_W": power,
                "temp_gpu_C": temp,
                "mem_used_MB": mem_used_MB,
                "mem_total_MB": mem_total_MB,
                "mem_free_MB": mem_free_MB,
                "throttled": throttled,
            }
        else:  # CPU/APU
            try:
                power = self.amdsmi.amdsmi_get_cpu_socket_power(self.handle) / 1000.0
            except Exception as e:
                if not self._cpu_power_warned:
                    logging.debug(f"AMD CPU/APU power not available (HSMP not supported): {e}")
                    self._cpu_power_warned = True
                power = "N/A"
            try:
                temp = self.amdsmi.amdsmi_get_cpu_socket_temperature(self.handle)
            except Exception as e:
                if not self._cpu_temp_warned:
                    logging.debug(f"AMD CPU/APU temp not available (HSMP not supported): {e}")
                    self._cpu_temp_warned = True
                temp = "N/A"
            return {
                "vendor": "AMD",
                "model": self._model,
                "device_id": self.idx,
                "hostname": self.hostname,
                "serial": self._serial,
                "cpu_power_W": power,
                "soc_temp_C": temp,
            }

    def get_stats(self, skip_first_n=0):
        """Return min/mean/max statistics collected during benchmark.
        
        Args:
            skip_first_n: Skip first N readings when calculating mean (to avoid warmup skew)
        """
        stats = {}
        for key, vals in self.readings.items():
            if vals:
                stats[f"{key}_min"] = min(vals)
                vals_for_mean = vals[skip_first_n:] if len(vals) > skip_first_n else vals
                stats[f"{key}_mean"] = statistics.mean(vals_for_mean)
                stats[f"{key}_max"] = max(vals)
        return stats
    
    def reset_stats(self):
        """Clear accumulated statistics to start fresh."""
        for key in self.readings:
            self.readings[key] = []
        self.throttle_detected = False
        self.throttle_count = 0
    
    def check_thermal_warnings(self, log=None, temp_warn=90.0, temp_critical=95.0, power_warn_pct=98.0) -> Dict[str, Any]:
        """Check for thermal or power warnings on AMD GPU/APU.
        
        Args:
            log: Logger instance for warnings
            temp_warn: Temperature warning threshold in Celsius
            temp_critical: Temperature critical threshold in Celsius  
            power_warn_pct: Power limit warning threshold in percent
        """
        warnings = {'thermal': [], 'power': [], 'throttle': []}
        
        # Check GPU temperature (for GPUs only)
        if self._is_gpu and 'temp_gpu_C_max' in self.get_stats():
            gpu_max = self.get_stats()['temp_gpu_C_max']
            if gpu_max >= temp_critical:
                msg = f"[CRIT] GPU temperature reached {gpu_max}°C (limit: {temp_critical}°C)"
                warnings['thermal'].append(msg)
                if log:
                    log.error(msg)
            elif gpu_max >= temp_warn:
                msg = f"[WARN] GPU temperature reached {gpu_max}°C (approaching {temp_critical}°C limit)"
                warnings['thermal'].append(msg)
                if log:
                    log.warning(msg)
        
        # Check power limit proximity (for GPUs with power cap info)
        if self.power_limit_W and 'power_W_max' in self.get_stats():
            power_max = self.get_stats()['power_W_max']
            power_pct = (power_max / self.power_limit_W) * 100
            if power_pct >= power_warn_pct:
                msg = f"[WARN] Power limit reached: {power_max:.0f}W / {self.power_limit_W:.0f}W ({power_pct:.1f}%)"
                warnings['power'].append(msg)
                if log:
                    log.warning(msg)
        
        # Report THERMAL throttling only (power cap throttling is expected/normal)
        if self.thermal_throttle_count > 0:
            msg = f"[WARN] THERMAL throttling detected in {self.thermal_throttle_count} telemetry reading(s)"
            warnings['throttle'].append(msg)
            if log:
                log.warning(msg)
        
        return warnings
    
    def shutdown(self):
        try:
            self.amdsmi.amdsmi_shut_down()
        except Exception:
            pass



# ── Intel dGPU (Placeholder) ─────────────────────────────────────────
class IntelTelemetry(TelemetryBase):
    def __init__(self, index):
        import socket
        self.idx = index
        self.hostname = socket.gethostname().split('.', 1)[0]

    def read(self):
        return {
            "vendor": "Intel",
            "model": "Intel_GPU",
            "device_id": self.idx,
            "hostname": self.hostname,
        }


# ── CPU fallback ───────────────────────────────────────────────────────
class CpuTelemetry(TelemetryBase):
    def __init__(self, index):
        import socket
        self.idx = index
        self._model = platform.machine()
        self.hostname = socket.gethostname().split('.', 1)[0]

    def read(self):
        return {
            "vendor": platform.processor() or "CPU",
            "model": self._model,
            "device_id": self.idx,
            "hostname": self.hostname,
        }


# ── Factory selecting best available backend ───────────────────────────
def make_telemetry(index: int, device: torch.device) -> TelemetryBase:
    if shutil.which("nvidia-smi") and not shutil.which("rocm-smi"):
        try:
            import pynvml
            return NVMLTelemetry(index)
        except Exception as e:
            print(f"Warning: Could not initialize Nvidia telemetry {e}")
    if shutil.which("rocm-smi"):
        try:
            return RocmTelemetry(index)
        except Exception as e:
            print(f"Warning: Could not load AMD telemetry: {e}")
            pass
    if device.type == "xpu":
        return IntelTelemetry(index)
    return CpuTelemetry(index)


# ── Background Telemetry Thread ────────────────────────────────────────
# Constants for telemetry thread configuration
TELEMETRY_IDLE_INTERVAL_MS = 500  # Sampling interval when benchmark not active

# ── Memory Management Constants ────────────────────────────────────────
# Conservative memory thresholds to avoid OOM during stress testing
MEMORY_TOTAL_CAPACITY_RATIO = 0.60  # Use at most 60% of total GPU memory
MEMORY_FREE_USAGE_RATIO = 0.90  # Use at most 90% of currently free memory
MEMORY_STRESS_TEST_RATIO = 0.50  # Use 50% of available for stress test sizing
MEMORY_SPARSE_OPS_RATIO = 0.60  # Extra margin for sparse ops workspace
MEMORY_DEFAULT_MPS_MB = 8000  # Conservative fallback for Apple Metal (8GB)
MEMORY_DEFAULT_CPU_MB = 16000  # Fallback for CPU mode (16GB)
MEMORY_DEFAULT_FALLBACK_MB = 8000  # Ultimate fallback (8GB)

class TelemetryThread:
    """Background thread for continuous telemetry monitoring with zero GPU interrupts."""
    
    def __init__(self, telemetry: TelemetryBase, device: torch.device, sample_interval_ms: float = 100):
        self.telemetry = telemetry
        self.device = device
        self.sample_interval_ms = sample_interval_ms
        self.idle_interval_ms = TELEMETRY_IDLE_INTERVAL_MS
        
        # Thread control
        self.running = False
        self.active = False  # Benchmark running flag
        self.thread = None
        
        # Thread-safe data structures
        self.lock = threading.Lock()
        self.latest_reading = {}
        self.all_samples = []  # All telemetry samples with timestamps
        
        # Per-iteration tracking
        self.current_iteration = None
        self.iteration_start_time = None
        self.iteration_samples = {}  # iteration_num -> [samples]
    
    def start(self):
        """Start background telemetry thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background telemetry thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def set_active(self, active: bool):
        """Set benchmark active state (affects sampling rate)."""
        with self.lock:
            self.active = active
    
    def mark_iteration_start(self, iteration_num: int):
        """Mark start of benchmark iteration for time-windowed telemetry."""
        with self.lock:
            self.current_iteration = iteration_num
            self.iteration_start_time = time.perf_counter()
            if iteration_num not in self.iteration_samples:
                self.iteration_samples[iteration_num] = []
    
    def mark_iteration_end(self, iteration_num: int):
        """Mark end of benchmark iteration."""
        with self.lock:
            self.current_iteration = None
            self.iteration_start_time = None
    
    def get_latest(self) -> Dict[str, Any]:
        """Get most recent telemetry reading (non-blocking)."""
        with self.lock:
            return self.latest_reading.copy() if self.latest_reading else {}
    
    def get_iteration_telemetry(self, iteration_num: int) -> Optional[Dict[str, Any]]:
        """Get aggregated telemetry for specific iteration (min/mean/max)."""
        with self.lock:
            samples = self.iteration_samples.get(iteration_num, [])
            if not samples:
                return self.latest_reading.copy() if self.latest_reading else {}
            
            # Aggregate numeric fields across all samples in iteration window
            aggregated = samples[0].copy()  # Start with first sample
            numeric_fields = {k: [] for k in samples[0].keys() 
                            if isinstance(samples[0][k], (int, float)) and k not in ('device_id',)}
            
            for sample in samples:
                for field in numeric_fields:
                    if field in sample and isinstance(sample[field], (int, float)):
                        numeric_fields[field].append(sample[field])
            
            # Use mean values for the iteration
            for field, values in numeric_fields.items():
                if values:
                    aggregated[field] = int(statistics.mean(values)) if isinstance(values[0], int) else statistics.mean(values)
            
            return aggregated
    
    def _poll_loop(self):
        """Background polling loop with adaptive sampling."""
        # Pin telemetry thread to dedicated CPU (last in affinity set) to avoid contention
        import os
        if hasattr(os, 'sched_getaffinity'):
            try:
                current_affinity = os.sched_getaffinity(0)
                if len(current_affinity) > 1:
                    # Reserve highest-numbered CPU for telemetry, rest for main thread
                    telemetry_cpu = max(current_affinity)
                    os.sched_setaffinity(0, {telemetry_cpu})
                    # Note: Can't easily log here since this runs in background thread
                    # but this ensures telemetry doesn't compete with benchmark compute
            except (AttributeError, OSError):
                # Not available on this platform (macOS, Windows) or permission denied
                pass
        
        while self.running:
            try:
                # Read telemetry (this is the ONLY place tel.read() is called during benchmarks)
                reading = self.telemetry.read()
                timestamp = time.perf_counter()
                
                # Thread-safe update
                with self.lock:
                    self.latest_reading = reading
                    self.all_samples.append((timestamp, reading))
                    
                    # Associate with current iteration if active
                    if self.current_iteration is not None:
                        self.iteration_samples[self.current_iteration].append(reading)
                
                # Adaptive sleep interval
                interval = self.sample_interval_ms if self.active else self.idle_interval_ms
                time.sleep(interval / 1000.0)
                
            except Exception as e:
                # Don't crash thread on telemetry errors
                time.sleep(0.5)
                continue


# ───────────────────────────────────────────────────────────────────────
# 2.  VERBOSE PRINTER  ─────────────────────────────────────────────────
class VerbosePrinter:
    def __init__(self, logger, schema: List[str], gpu_id: int):
        self.log = logger
        self.schema = ["vendor", "model"] + [k for k in schema if k not in ("vendor", "model")]
        self.gpu_id = gpu_id
        self.last_header_keys = None
        self.repeat_num = 1  # Current repeat number

    def emit(self, i, test, dtype, metric_name, metric_val, tel_data: Dict[str, Any]):
        """Emit verbose output with pre-fetched telemetry data (no blocking read)."""
        _i = i + 1
        row = {
            "repeat": self.repeat_num,
            "iter": _i,
            "test": test,
            "dtype": dtype,
            metric_name: f"{metric_val:.2f}",
        }
        for k in self.schema:
            row[k] = tel_data.get(k, "")
        
        # Print header if keys changed (different metric name or first time)
        current_keys = tuple(row.keys())
        if current_keys != self.last_header_keys:
            self.log.info(", ".join(row.keys()))
            self.last_header_keys = current_keys
        
        # Format values with reasonable precision for floats
        def format_value(v):
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)
        
        self.log.info(", ".join(format_value(v) for v in row.values()))


# ───────────────────────────────────────────────────────────────────────
# 3.  TIMER  ───────────────────────────────────────────────────────────
class Timer:
    """CUDA events for GPU, perf_counter for CPU."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        if self.cuda:
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.cuda:
            self.s.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.cuda:
            self.e.record()
            self.e.synchronize()
            # Explicit sync for ROCm/HIP - events don't imply host sync like CUDA
            if torch.version.hip:
                torch.cuda.synchronize()
            self.elapsed = self.s.elapsed_time(self.e) / 1e3
        else:
            self.elapsed = time.perf_counter() - self.t0


# ───────────────────────────────────────────────────────────────────────
# 4.  SMALL HELPERS  ───────────────────────────────────────────────────
def gflops(flops: float, secs: float) -> float:
    return flops / secs / 1e9


def _format_telemetry_compact(tel_data: Dict[str, Any]) -> str:
    """Format telemetry data in compact human-readable form."""
    parts = []
    
    # Device info - smart label based on vendor
    device_id = tel_data.get('device_id', '?')
    vendor = tel_data.get('vendor', '')
    model = tel_data.get('model', '')
    
    # Determine appropriate label
    if vendor in ('NVIDIA Corporation', 'Advanced Micro Devices, Inc. [AMD/ATI]', 'NVIDIA', 'AMD'):
        parts.append(f"GPU{device_id}")
    elif vendor in ('CPU', 'GenuineIntel', 'AuthenticAMD') or \
         'arm' in model.lower() or 'x86' in model.lower() or \
         model in ('arm64', 'aarch64', 'x86_64', 'i386', 'i686'):
        # CPU or Apple Silicon (MPS) - show model name
        parts.append(model if model else 'CPU')
    elif vendor == '' and model:
        # Empty vendor but we have a model - likely CPU
        parts.append(model)
    else:
        # Unknown device type
        parts.append(f"Device{device_id}")
    
    # Utilization
    if 'sm_util' in tel_data and tel_data['sm_util'] != 'N/A':
        parts.append(f"SM:{tel_data['sm_util']}%")
    if 'mem_bw_util' in tel_data and tel_data['mem_bw_util'] != 'N/A':
        parts.append(f"MemBW:{tel_data['mem_bw_util']}%")
    
    # Temperature
    if 'temp_gpu_C' in tel_data and tel_data['temp_gpu_C'] != 'N/A':
        parts.append(f"Temp:{tel_data['temp_gpu_C']}°C")
    
    # Power - handle both numeric and string values safely
    if 'power_W' in tel_data and tel_data['power_W'] != 'N/A':
        try:
            parts.append(f"Power:{float(tel_data['power_W']):.0f}W")
        except (ValueError, TypeError):
            pass  # Skip if not convertible to float
    
    # Clock
    if 'gpu_clock' in tel_data and tel_data['gpu_clock'] != 'N/A':
        parts.append(f"Clock:{tel_data['gpu_clock']}MHz")
    
    # Memory usage
    if 'mem_used_MB' in tel_data and tel_data['mem_used_MB'] != 'N/A':
        used = tel_data['mem_used_MB'] / 1024  # GB
        if 'mem_total_MB' in tel_data and tel_data['mem_total_MB'] != 'N/A':
            total = tel_data['mem_total_MB'] / 1024  # GB
            pct = (tel_data['mem_used_MB'] / tel_data['mem_total_MB']) * 100
            parts.append(f"Mem:{used:.1f}/{total:.1f}GB({pct:.0f}%)")
        else:
            parts.append(f"Mem:{used:.1f}GB")
    
    return " | ".join(parts)


# ───────────────────────────────────────────────────────────────────────
# 4a′. COMPACT CSV HELPERS  ────────────────────────────────────────────

def _compact_csv_columns(verbose: bool = False) -> list:
    """Return ordered column names for compact CSV output.

    Base columns (14):
        hostname, gpu, gpu_model, serial, benchmark, dtype, iterations,
        runtime_s, min, mean, max, unit, power_avg_w, temp_max_c

    With --verbose (19): appends sm_util_mean, mem_bw_util_mean,
        gpu_clock_mean, mem_used_gb_mean, throttled
    """
    cols = [
        "hostname", "gpu", "gpu_model", "serial",
        "benchmark", "dtype", "iterations", "runtime_s",
        "min", "mean", "max", "unit",
        "power_avg_w", "temp_max_c",
    ]
    if verbose:
        cols += [
            "sm_util_mean", "mem_bw_util_mean",
            "gpu_clock_mean", "mem_used_gb_mean", "throttled",
        ]
    return cols


def _emit_compact_csv(row: dict, verbose: bool = False,
                      header: bool = False, file=None) -> None:
    """Print a single compact CSV row (and optional header) to *file*.

    *file* defaults to ``sys.stdout`` so that CSV data always goes to
    stdout even when the logging framework is redirected to stderr or a
    log file.
    """
    import csv, io
    out = file or sys.stdout
    cols = _compact_csv_columns(verbose)
    if header:
        print(",".join(cols), file=out, flush=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writerow({c: row.get(c, "") for c in cols})
    print(buf.getvalue().rstrip("\r\n"), file=out, flush=True)


def _log_summary(name, vals, unit, logger, tel, device, params=None, baselines=None, verbose=False, skip_telemetry=10, tel_thread=None, runtime_s=None):
    s = dict(min=min(vals), mean=statistics.mean(vals), max=max(vals))
    tel_data = tel.read()
    tel_stats = tel.get_stats(skip_first_n=skip_telemetry)
    gpu_id = tel_data.get('device_id', '?')
    
    # Get device-appropriate label (GPU0, MPS, or CPU)
    dev_label = device_label(device, gpu_id)
    
    # Build header with benchmark name and performance
    perf_str = f"{s['min']:.2f} / {s['mean']:.2f} / {s['max']:.2f} {unit}"
    
    # Collect per-iteration telemetry with performance values and timestamps
    iteration_telemetry = []
    if tel_thread:
        with tel_thread.lock:
            # Get all iterations that have telemetry samples
            for iter_num in sorted(tel_thread.iteration_samples.keys()):
                samples = tel_thread.iteration_samples[iter_num]
                if samples and iter_num < len(vals):
                    # Get corresponding performance value and timestamp
                    perf_value = vals[iter_num]
                    # Use timestamp from first sample in iteration window
                    timestamp = None
                    for ts, reading in tel_thread.all_samples:
                        if reading == samples[0]:
                            timestamp = ts
                            break
                    
                    iteration_telemetry.append({
                        'iteration': iter_num,
                        'timestamp': timestamp,
                        'performance': perf_value,
                        'unit': unit,
                        'telemetry': samples[0]  # First sample in iteration window
                    })
    
    # Return performance summary for multi-GPU aggregation
    perf_summary = {
        'name': name,
        'min': s['min'],
        'mean': s['mean'],
        'max': s['max'],
        'unit': unit,
        'iterations': len(vals),
        'runtime_s': round(runtime_s, 3) if runtime_s is not None else None,
        'params': params or {},  # Include test parameters
        'telemetry': tel_stats or {},  # Include telemetry stats for this benchmark
        'iteration_telemetry': iteration_telemetry  # Per-iteration telemetry with perf + timestamp
    }
    
    # Throttle warning - check if throttling occurred during entire run
    throttle_msg = ""
    if hasattr(tel, 'throttle_detected') and tel.throttle_detected:
        if hasattr(tel, 'hw_slowdown_count'):  # NVIDIA
            reasons = []
            if tel.hw_slowdown_count > 0:
                reasons.append(f"HW_THERMAL({tel.hw_slowdown_count}x)")
            if tel.sw_slowdown_count > 0:
                reasons.append(f"SW_THERMAL({tel.sw_slowdown_count}x)")
            if tel.power_limit_count > 0:
                reasons.append(f"HW_POWER_BRAKE({tel.power_limit_count}x)")
            throttle_msg = f" [THROTTLED: {', '.join(reasons)}]"
        else:  # AMD
            throttle_msg = f" [THROTTLED ({tel.throttle_count}x readings)]"
    
    # Skip detailed logging in verbose mode - just return summary data
    # In verbose mode, user sees CSV rows during run and table at end
    if verbose:
        # Still do baseline validation to populate perf_summary
        if params and tel_data:
            model_name = tel_data.get('model', '')
            dtype = params.get('dtype', '')
            tf32_mode = params.get('tf32', False)
            
            validation = validate_performance(model_name, name, dtype, s['mean'], unit, tf32_mode, logger, baselines=baselines)
            if validation['warning']:
                perf_summary['efficiency_pct'] = validation['efficiency']
                perf_summary['expected_tflops'] = validation['expected']
            elif validation['efficiency']:
                perf_summary['efficiency_pct'] = validation['efficiency']
        
        # Calculate power efficiency for summary
        if tel_stats and 'power_W_mean' in tel_stats:
            avg_power = tel_stats['power_W_mean']
            if unit == "GFLOP/s" and avg_power > 0:
                perf_summary['gflops_per_watt'] = s['mean'] / avg_power
            elif unit == "GB/s" and avg_power > 0:
                perf_summary['gbps_per_watt'] = s['mean'] / avg_power
        
        # Include throttle statistics in summary for verbose mode
        if hasattr(tel, 'throttle_detected') and tel.throttle_detected:
            perf_summary['throttled'] = True
            if hasattr(tel, 'hw_slowdown_count'):  # NVIDIA
                perf_summary['hw_thermal_count'] = tel.hw_slowdown_count
                perf_summary['sw_thermal_count'] = tel.sw_slowdown_count
                perf_summary['power_limit_count'] = tel.power_limit_count
            else:  # AMD
                perf_summary['throttle_count'] = tel.throttle_count
        
        return perf_summary
    
    # Non-verbose mode: log detailed summary blocks
    logger.info(f"{'='*80}")
    logger.info(f"[{dev_label} {name}] Performance: {perf_str} (min/mean/max){throttle_msg}")
    
    # Power efficiency metrics
    if tel_stats and 'power_W_mean' in tel_stats:
        avg_power = tel_stats['power_W_mean']
        if unit == "GFLOP/s" and avg_power > 0:
            gflops_per_watt = s['mean'] / avg_power
            perf_summary['gflops_per_watt'] = gflops_per_watt
            logger.info(f"[{dev_label} {name}] Power Efficiency: {gflops_per_watt:.2f} GFLOP/s/W (avg)")
        elif unit == "GB/s" and avg_power > 0:
            gbps_per_watt = s['mean'] / avg_power
            perf_summary['gbps_per_watt'] = gbps_per_watt
            logger.info(f"[{dev_label} {name}] Power Efficiency: {gbps_per_watt:.3f} GB/s/W (avg)")
    
    # Baseline validation for known hardware
    if params and tel_data:
        model_name = tel_data.get('model', '')
        dtype = params.get('dtype', '')
        tf32_mode = params.get('tf32', False)
        
        validation = validate_performance(model_name, name, dtype, s['mean'], unit, tf32_mode, logger, baselines=baselines)
        if validation['warning']:
            logger.warning(f"[{dev_label} {name}] {validation['warning']}")
            perf_summary['efficiency_pct'] = validation['efficiency']
            perf_summary['expected_tflops'] = validation['expected']
        elif validation['efficiency']:
            perf_summary['efficiency_pct'] = validation['efficiency']
            logger.info(f"[{dev_label} {name}] Hardware Efficiency: {validation['efficiency']:.1f}% of theoretical peak")
    
    # Log comprehensive telemetry statistics
    if tel_stats:
        logger.info(f"[{dev_label} {name}] Telemetry Statistics (min / mean / max):")
        
        # Group related metrics
        if 'sm_util_min' in tel_stats:
            logger.info(f"[{dev_label} {name}]   SM Utilization:  {tel_stats['sm_util_min']:.0f}% / {tel_stats['sm_util_mean']:.0f}% / {tel_stats['sm_util_max']:.0f}%")
        if 'mem_bw_util_min' in tel_stats:
            logger.info(f"[{dev_label} {name}]   Mem BW Util:     {tel_stats['mem_bw_util_min']:.0f}% / {tel_stats['mem_bw_util_mean']:.0f}% / {tel_stats['mem_bw_util_max']:.0f}%")
        if 'mem_util_min' in tel_stats:  # AMD
            logger.info(f"[{dev_label} {name}]   Mem Utilization: {tel_stats['mem_util_min']:.0f}% / {tel_stats['mem_util_mean']:.0f}% / {tel_stats['mem_util_max']:.0f}%")
        
        if 'temp_gpu_C_min' in tel_stats:
            logger.info(f"[{dev_label} {name}]   GPU Temp:        {tel_stats['temp_gpu_C_min']:.0f}°C / {tel_stats['temp_gpu_C_mean']:.0f}°C / {tel_stats['temp_gpu_C_max']:.0f}°C")
        if 'temp_hbm_C_min' in tel_stats:
            logger.info(f"[{dev_label} {name}]   HBM Temp:        {tel_stats['temp_hbm_C_min']:.0f}°C / {tel_stats['temp_hbm_C_mean']:.0f}°C / {tel_stats['temp_hbm_C_max']:.0f}°C")
        
        if 'power_W_min' in tel_stats:
            logger.info(f"[{dev_label} {name}]   Power:           {tel_stats['power_W_min']:.0f}W / {tel_stats['power_W_mean']:.0f}W / {tel_stats['power_W_max']:.0f}W")
        
        if 'gpu_clock_min' in tel_stats:
            logger.info(f"[{dev_label} {name}]   GPU Clock:       {tel_stats['gpu_clock_min']:.0f}MHz / {tel_stats['gpu_clock_mean']:.0f}MHz / {tel_stats['gpu_clock_max']:.0f}MHz")
        
        if 'mem_used_MB_min' in tel_stats:
            min_gb = tel_stats['mem_used_MB_min'] / 1024
            mean_gb = tel_stats['mem_used_MB_mean'] / 1024
            max_gb = tel_stats['mem_used_MB_max'] / 1024
            logger.info(f"[{dev_label} {name}]   Memory Used:     {min_gb:.1f}GB / {mean_gb:.1f}GB / {max_gb:.1f}GB")
    
    logger.info(f"{'='*80}")
    return perf_summary


# ───────────────────────────────────────────────────────────────────────
# 4b. DEVICE LABELING HELPER  ──────────────────────────────────────────
def device_label(device: torch.device, index: Any = 0) -> str:
    """Get human-readable device label for logging.
    
    Args:
        device: PyTorch device object
        index: GPU index (int) or fallback value (str like '?'). Non-int values default to 0.
    
    Returns:
        "GPU0", "GPU1", etc. for CUDA
        "MPS" for Apple Metal
        "CPU" for CPU-only mode
    """
    if device.type == "cuda":
        # Prefer device.index (set by torch.device('cuda:N')) over telemetry-derived index
        # This avoids race conditions with background telemetry thread and
        # incorrect device_id from AMD amdsmi handle ordering in multi-process runs
        if device.index is not None:
            return f"GPU{device.index}"
        idx = index if isinstance(index, int) else 0
        return f"GPU{idx}"
    elif device.type == "mps":
        return "MPS"
    else:
        return "CPU"


# ───────────────────────────────────────────────────────────────────────
# 4c. NUMA & CPU AFFINITY HELPERS  ─────────────────────────────────────
def get_gpu_numa_node(gpu_index: int) -> int:
    """Get CPU NUMA node for a GPU (NVIDIA and AMD).
    
    For NVIDIA: Uses nvidia-smi topo to get CPU NUMA affinity.
    For AMD: Uses rocm-smi --showtoponuma or sysfs fallback.
    """
    # Try NVIDIA first
    if shutil.which("nvidia-smi"):
        try:
            # Parse nvidia-smi topo -m for accurate CPU NUMA affinity
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, check=True
            )
            # Parse output: look for GPU{index} row and extract NUMA Affinity column
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith(f'GPU{gpu_index}'):
                    # Format: GPU0  X  NV6  NV6  NV6  0-71  0  4
                    # Columns: GPU, connections..., CPU Affinity, NUMA Affinity, GPU NUMA ID
                    parts = line.split()
                    # NUMA Affinity is second-to-last column
                    if len(parts) >= 2:
                        try:
                            numa_affinity = int(parts[-2])
                            return numa_affinity if numa_affinity >= 0 else 0
                        except ValueError:
                            pass
        except Exception:
            pass
    
    # Try AMD rocm-smi
    if shutil.which("rocm-smi"):
        try:
            # rocm-smi --showtoponuma shows NUMA node for each GPU
            result = subprocess.run(
                ["rocm-smi", "--showtoponuma"],
                capture_output=True, text=True, check=True
            )
            # Parse output - look for GPU[index] and its NUMA node
            # Format varies, look for patterns like "GPU[0]" or "GPU 0" followed by NUMA info
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if f'GPU[{gpu_index}]' in line or f'GPU {gpu_index}' in line:
                    # Look for NUMA node number in this or following lines
                    for check_line in [line] + lines[i+1:i+3]:
                        # Look for patterns like "NUMA node: 0" or just a number
                        import re
                        numa_match = re.search(r'NUMA[^0-9]*(\d+)', check_line, re.IGNORECASE)
                        if numa_match:
                            return int(numa_match.group(1))
                        # Also check for just "Node: X" pattern
                        node_match = re.search(r'Node[^0-9]*(\d+)', check_line, re.IGNORECASE)
                        if node_match:
                            return int(node_match.group(1))
        except Exception:
            pass
    
    # Fallback: try sysfs for any GPU type
    # Try render node first (more reliable for AMD)
    for path_pattern in [
        f"/sys/class/drm/renderD{128 + gpu_index}/device/numa_node",
        f"/sys/class/drm/card{gpu_index}/device/numa_node",
    ]:
        try:
            if os.path.exists(path_pattern):
                with open(path_pattern) as f:
                    numa_node = int(f.read().strip())
                    return numa_node if numa_node >= 0 else 0
        except Exception:
            pass
    
    return 0  # Default to NUMA node 0


def get_numa_cpus(numa_node: int) -> List[int]:
    """Get CPU cores for a NUMA node."""
    try:
        cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        if os.path.exists(cpulist_path):
            with open(cpulist_path) as f:
                cpulist = f.read().strip()
                # Parse ranges like "0-15,32-47"
                cpus = []
                for part in cpulist.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        cpus.extend(range(start, end + 1))
                    else:
                        cpus.append(int(part))
                return cpus
    except Exception as e:
        logging.warning(f"Could not read NUMA CPU list: {e}")
    return []


def format_cpu_list(cpus: List[int]) -> str:
    """Format CPU list as compact ranges (e.g., '0-15,32-47' instead of full list)."""
    if not cpus:
        return "[]"
    
    cpus_sorted = sorted(cpus)
    ranges = []
    start = cpus_sorted[0]
    end = cpus_sorted[0]
    
    for cpu in cpus_sorted[1:]:
        if cpu == end + 1:
            end = cpu
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = cpu
    
    # Add final range
    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    return ",".join(ranges)


def set_cpu_affinity(cpus: List[int], quiet: bool = False) -> bool:
    """Set CPU affinity for current process.
    
    Args:
        cpus: List of CPU cores to pin to
        quiet: If True, suppress logging (caller will log instead)
    """
    try:
        import os
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, cpus)
            if not quiet:
                logging.info(f"CPU affinity: pinned to {format_cpu_list(cpus)}")
            return True
    except Exception as e:
        logging.warning(f"Could not set CPU affinity: {e}")
    return False


def get_gpu_numa_mapping(gpu_indices: List[int]) -> Dict[int, List[int]]:
    """Get mapping of NUMA nodes to GPUs for intelligent CPU distribution.
    
    Returns:
        Dict mapping NUMA node ID to list of GPU indices on that node
    """
    numa_to_gpus = {}
    for gpu_idx in gpu_indices:
        numa_node = get_gpu_numa_node(gpu_idx)
        if numa_node not in numa_to_gpus:
            numa_to_gpus[numa_node] = []
        numa_to_gpus[numa_node].append(gpu_idx)
    return numa_to_gpus


def distribute_cpus_for_gpus(gpu_indices: List[int]) -> Dict[int, List[int]]:
    """Assign CPUs per GPU from its NUMA domain.
    
    Each GPU gets:
      - One dedicated CPU for benchmark (starting from 2nd core, skipping core 0)
      - Last CPU in its NUMA domain for telemetry
    
    For 1:1 GPU:NUMA mapping (e.g., GH200), each GPU has its own telemetry CPU.
    For multi-GPU per NUMA, GPUs on the same NUMA share the telemetry CPU.
    
    Example: GH200 (1 GPU per NUMA, CPUs 0-17 per domain)
      NUMA0: GPU 0 → [1, 17]   (benchmark=1, telemetry=17)
      NUMA1: GPU 1 → [19, 35]  (benchmark=19, telemetry=35)
      NUMA2: GPU 2 → [37, 53]  (benchmark=37, telemetry=53)
      NUMA3: GPU 3 → [55, 71]  (benchmark=55, telemetry=71)
    
    Example: 4 GPUs on NUMA0 (CPUs 0-31)
      GPU 0 → [1, 31]   (benchmark=1, telemetry=31 shared)
      GPU 1 → [2, 31]   (benchmark=2, telemetry=31 shared)
      GPU 2 → [3, 31]
      GPU 3 → [4, 31]
    
    Returns:
        Dict mapping GPU index to list of CPU cores [benchmark_cpu, telemetry_cpu]
    """
    numa_to_gpus = get_gpu_numa_mapping(gpu_indices)
    gpu_to_cpus = {}
    
    for numa_node, gpus_on_node in numa_to_gpus.items():
        all_cpus = get_numa_cpus(numa_node)
        if not all_cpus:
            continue
        
        num_gpus = len(gpus_on_node)
        telemetry_cpu = all_cpus[-1]  # Last CPU in NUMA domain for telemetry
        
        # Reserve: core 0 (interrupts) + last core (telemetry)
        # Benchmark CPUs start from core 1
        if len(all_cpus) > 2:
            benchmark_cpus = all_cpus[1:-1]  # Skip first and last
        elif len(all_cpus) == 2:
            benchmark_cpus = [all_cpus[0]]  # Use first, last is telemetry
        else:
            benchmark_cpus = all_cpus  # Single core, must share
        
        # Assign one benchmark CPU per GPU
        for i, gpu_idx in enumerate(sorted(gpus_on_node)):
            if i < len(benchmark_cpus):
                gpu_to_cpus[gpu_idx] = [benchmark_cpus[i], telemetry_cpu]
            else:
                # More GPUs than benchmark CPUs - wrap around
                gpu_to_cpus[gpu_idx] = [benchmark_cpus[i % len(benchmark_cpus)], telemetry_cpu]
    
    return gpu_to_cpus


def parse_cpu_gpu_map(map_str: str) -> Dict[int, List[int]]:
    """Parse CPU-GPU mapping string like '0:0-15,1:16-31,2:32-47'."""
    mapping = {}
    for entry in map_str.split(','):
        if ':' not in entry:
            continue
        gpu_part, cpu_part = entry.split(':', 1)
        gpu_id = int(gpu_part.strip())
        
        cpus = []
        for cpu_range in cpu_part.split(','):
            cpu_range = cpu_range.strip()
            if '-' in cpu_range:
                start, end = map(int, cpu_range.split('-'))
                cpus.extend(range(start, end + 1))
            else:
                cpus.append(int(cpu_range))
        mapping[gpu_id] = cpus
    return mapping


def parse_cpu_list(cpu_list_str: str) -> List[int]:
    """Parse CPU list string like '0-23,48-71' into list of CPU IDs.
    
    Args:
        cpu_list_str: CPU specification (e.g., '0-23,48-71' or 'all')
    
    Returns:
        List of CPU core IDs
    """
    if cpu_list_str.lower() == 'all':
        return list(range(os.cpu_count() or 1))
    
    cpus = []
    for part in cpu_list_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def get_physical_core_count() -> int:
    """Get number of physical CPU cores (excluding hyperthreads).
    
    Uses /sys/devices/system/cpu topology on Linux, falls back to os.cpu_count()/2.
    """
    # Linux: count unique core_id values across all CPUs
    try:
        cores_seen = set()
        cpu_dir = "/sys/devices/system/cpu"
        if os.path.isdir(cpu_dir):
            for entry in os.listdir(cpu_dir):
                if entry.startswith("cpu") and entry[3:].isdigit():
                    topology_path = os.path.join(cpu_dir, entry, "topology", "core_id")
                    package_path = os.path.join(cpu_dir, entry, "topology", "physical_package_id")
                    if os.path.exists(topology_path) and os.path.exists(package_path):
                        with open(topology_path) as f:
                            core_id = f.read().strip()
                        with open(package_path) as f:
                            package_id = f.read().strip()
                        # Unique physical core = (package_id, core_id) tuple
                        cores_seen.add((package_id, core_id))
            if cores_seen:
                return len(cores_seen)
    except Exception:
        pass
    
    # macOS: use sysctl
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except Exception:
            pass
    
    # Fallback: assume 2-way SMT (hyperthreading)
    logical_cpus = os.cpu_count() or 1
    return max(1, logical_cpus // 2)


def setup_cpu_threading(device: torch.device, args, log) -> None:
    """Configure optimal CPU threading for CPU-only mode.
    
    Sets OMP_NUM_THREADS and thread binding for maximum CPU utilization.
    Only applies when running in CPU-only mode (no GPU/MPS).
    
    Args:
        device: PyTorch device (should be cpu)
        args: Parsed arguments (may have cpu_list override)
        log: Logger instance
    """
    if device.type != "cpu":
        return  # Only for CPU-only mode
    
    # Check for user override via --cpu-list
    if hasattr(args, 'cpu_list') and args.cpu_list:
        cpus = parse_cpu_list(args.cpu_list)
        num_threads = len(cpus)
        log.info(f"CPU threading: using {num_threads} cores from --cpu-list ({format_cpu_list(cpus)})")
        
        # Set thread count
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        
        # Set CPU affinity if available
        if hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, cpus)
                os.environ["GOMP_CPU_AFFINITY"] = format_cpu_list(cpus).replace(",", " ")
            except Exception as e:
                log.warning(f"Could not set CPU affinity: {e}")
        return
    
    # Default: use all physical cores (no hyperthreads)
    physical_cores = get_physical_core_count()
    logical_cores = os.cpu_count() or 1
    
    log.info(f"CPU threading: {physical_cores} physical cores detected ({logical_cores} logical)")
    
    # Set PyTorch thread count (works at runtime)
    torch.set_num_threads(physical_cores)
    
    # Note: OMP_NUM_THREADS, OMP_PROC_BIND, OMP_PLACES were set early (before torch import)
    # Verify they match our detected core count
    omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")
    log.info(f"CPU threading: torch.set_num_threads({physical_cores}), OMP_NUM_THREADS={omp_threads}")


def get_available_memory_mb(device: torch.device) -> float:
    """Get available GPU memory in MB (portable across CUDA/ROCm/MPS)."""
    if device.type == "cuda":
        try:
            # Clear PyTorch cache first to get accurate reading
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info(device.index)
            free_mb = free_mem / 1024**2
            total_mb = total_mem / 1024**2
            
            # Use whichever is smaller: configured ratio of total OR free memory
            # This handles cases where memory is already allocated
            conservative_total = total_mb * MEMORY_TOTAL_CAPACITY_RATIO
            return min(conservative_total, free_mb * MEMORY_FREE_USAGE_RATIO)
        except Exception:
            # Fallback: total memory with conservative margin
            try:
                total = torch.cuda.get_device_properties(device.index).total_memory / 1024**2
                return total * MEMORY_TOTAL_CAPACITY_RATIO
            except Exception:
                return MEMORY_DEFAULT_FALLBACK_MB
    elif device.type == "mps":
        # Apple Metal - no direct query, use conservative estimate
        return MEMORY_DEFAULT_MPS_MB
    else:
        return MEMORY_DEFAULT_CPU_MB


def calculate_stress_params(benchmark: str, precision: str, available_mb: float, log, args=None) -> Dict[str, int]:
    """Calculate maximum stress test parameters that fit in memory.
    
    If args is provided, uses CLI-specified values where available (e.g., density).
    Only auto-calculates size parameters to fit memory.
    """
    dtype_bytes = {
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
        "complex64": 8, "complex128": 16
    }
    bytes_per_elem = dtype_bytes.get(precision, 4)
    
    # Use configured ratio of available memory for safety
    # (PyTorch overhead, workspace buffers, gradients, etc.)
    usable_mb = available_mb * MEMORY_STRESS_TEST_RATIO
    usable_bytes = usable_mb * 1024**2
    
    params = {}
    
    if benchmark == "gemm":
        # GEMM: 2 input matrices + 1 output = 3 matrices total
        # Memory = B * (M*K + K*N + M*N) * bytes
        # For square matrices: B * 3*M^2 * bytes
        batch = 64  # Smaller batch to reduce memory pressure
        total_elems = usable_bytes / bytes_per_elem / 3  # 3 matrices
        side = int((total_elems / batch) ** 0.5)
        side = max(512, min(side, 16384))  # Clamp to reasonable range
        params = {"m": side, "n": side, "k": side, "batch_count_gemm": batch}
        est_mb = batch * 3 * side * side * bytes_per_elem / 1024**2
        log.info(f"[Stress Test] GEMM: {batch}x{side}x{side}x{side} (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "convolution":
        # Conv: input (B*Ci*H*W) + output (B*Co*H*W) + weights (Co*Ci*K*K)
        # Assume Co=4*Ci, optimize spatial size
        batch = 128
        ci, co, kernel = 64, 256, 3
        # Total ≈ B*Ci*H*W + B*Co*H*W + Co*Ci*K*K
        # ≈ B*H*W*(Ci+Co) dominant
        spatial_elems = usable_bytes / bytes_per_elem / (batch * (ci + co) + co * ci * kernel * kernel)
        side = int(spatial_elems ** 0.5)
        side = max(128, min(side, 4096))
        params = {"batch_count_convolution": batch, "in_channels": ci, 
                 "out_channels": co, "height": side, "width": side, "kernel_size": kernel}
        est_mb = (batch * ci * side * side + batch * co * side * side) * bytes_per_elem / 1024**2
        log.info(f"[Stress Test] Conv: {batch}x{ci}→{co}x{side}x{side} (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "fft":
        # FFT: 1 input + 1 output (complex) = 2x memory
        batch = 64  # Smaller batch
        total_elems = usable_bytes / bytes_per_elem / 2  # Input + output
        cube_side = int((total_elems / batch) ** (1/3))
        cube_side = max(128, min(cube_side, 1024))
        params = {"batch_count_fft": batch, "nx": cube_side, "ny": cube_side, "nz": cube_side}
        est_mb = batch * 2 * cube_side**3 * bytes_per_elem / 1024**2
        log.info(f"[Stress Test] FFT: {batch}x{cube_side}³ (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "einsum":
        # Einsum: 2 inputs (B*H*Q*D each), output (B*H*Q*Q)
        # Total ≈ 2*B*H*Q*D + B*H*Q*Q
        batch, heads = 64, 32  # Smaller batch
        d_model = 128  # Fixed
        # Solve for Q: 2*B*H*Q*D + B*H*Q*Q ≈ usable_bytes / bytes_per_elem
        # Dominated by B*H*Q*Q for large Q
        q_squared = usable_bytes / bytes_per_elem / (batch * heads) - 2 * d_model
        seq_len = int(q_squared ** 0.5)
        seq_len = max(128, min(seq_len, 8192))
        params = {"batch_count_einsum": batch, "heads": heads, "seq_len": seq_len, "d_model": d_model}
        est_mb = (2 * batch * heads * seq_len * d_model + batch * heads * seq_len * seq_len) * bytes_per_elem / 1024**2
        log.info(f"[Stress Test] Einsum: {batch}x{heads}x{seq_len}x{d_model} (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "memory":
        # Memory: 1 vector (N elements) + index vector (int64 = 8 bytes)
        total_elems = usable_bytes / (bytes_per_elem + 8)  # Data + indices
        size = int(total_elems)
        size = max(1_000_000, min(size, 1_000_000_000))
        params = {"memory_size": size}
        est_mb = size * (bytes_per_elem + 8) / 1024**2
        log.info(f"[Stress Test] Memory: {size} elements (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "heat":
        # Heat: 2 grids (N*N each for current and next)
        total_elems = usable_bytes / bytes_per_elem / 2
        grid_side = int(total_elems ** 0.5)
        grid_side = max(512, min(grid_side, 16384))
        params = {"heat_grid_size": grid_side}
        est_mb = 2 * grid_side * grid_side * bytes_per_elem / 1024**2
        log.info(f"[Stress Test] Heat: {grid_side}x{grid_side} grid (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "schrodinger":
        # Schrödinger: 1D vector (N elements, complex128 = 16 bytes fixed)
        total_elems = usable_bytes / 16  # Force complex128
        size = int(total_elems)
        size = max(10_000, min(size, 100_000_000))
        params = {"schrodinger_grid_size": size}
        est_mb = size * 16 / 1024**2
        log.info(f"[Stress Test] Schrödinger: {size} grid (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "atomic":
        # Atomic: 1 target array (N) + 1 index array (int64) + 1 source array (N_updates)
        # Memory is dominated by index array for large update counts
        # Use fixed target size, scale updates based on memory
        target_size = 1_000_000  # 1M elements fixed (small to maximize contention)
        # Index array is int64 (8 bytes), source array is dtype
        # Total = target_size*bytes + num_updates*8 + num_updates*bytes
        max_updates = usable_bytes / (8 + bytes_per_elem)  # Per update: index + value
        num_updates = int(max_updates)
        num_updates = max(1_000_000, min(num_updates, 100_000_000))  # 1M to 100M updates
        params = {"atomic_target_size": target_size, "atomic_num_updates": num_updates}
        est_mb = (target_size * bytes_per_elem + num_updates * (8 + bytes_per_elem)) / 1024**2
        log.info(f"[Stress Test] Atomic: {num_updates} updates to {target_size} target (est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    elif benchmark == "sparse":
        # Sparse MM: sparse (M*K, nnz elements) * dense (K*N) = output (M*N)
        # Memory = nnz*bytes + nnz*2*int64 (COO indices) + K*N*bytes (dense B) + M*N*bytes (output C)
        # For density d: nnz = M*K*d
        # Use CLI-provided density if available, otherwise default to 10%
        density = getattr(args, 'sparse_density', 0.10) if args else 0.10
        # Optimize for square matrices with given density
        # Total for M=N=K=side: side^2*d*(bytes+16) + side^2*bytes (B) + side^2*bytes (C)
        #                     = side^2 * (d*(bytes+16) + 2*bytes)
        # PyTorch sparse ops need extra workspace, apply additional safety margin
        conservative_bytes = usable_bytes * MEMORY_SPARSE_OPS_RATIO
        side_sq = conservative_bytes / (density * (bytes_per_elem + 16) + 2 * bytes_per_elem)
        side = int(side_sq ** 0.5)
        side = max(1024, min(side, 8192))  # Cap at 8192 to avoid extreme memory usage
        nnz = int(side * side * density)
        params = {"sparse_m": side, "sparse_n": side, "sparse_k": side, "sparse_density": density}
        est_mb = (nnz * (bytes_per_elem + 16) + 2 * side * side * bytes_per_elem) / 1024**2
        log.info(f"[Stress Test] Sparse MM: {side}x{side} @ {density*100:.1f}% density (~{nnz} nnz, est {est_mb:.0f}MB of {available_mb:.0f}MB available)")
    
    return params


# ───────────────────────────────────────────────────────────────────────
# 4c. ITERATION CONTROL HELPERS ────────────────────────────────────────
def should_continue_iterations(args, iteration: int, vals: List[float], start_time: float) -> bool:
    """
    Determine if benchmark should continue based on duration or iteration limits.
    
    Returns: True if should continue, False if stopping criteria met
    """
    # Check minimum iterations first
    if iteration < args.min_iterations:
        return True
    
    # Check maximum iterations
    if args.max_iterations and iteration >= args.max_iterations:
        logging.debug(f"Stopping: reached max_iterations ({args.max_iterations})")
        return False
    
    # Check duration limit
    if args.duration:
        elapsed = time.perf_counter() - start_time
        if elapsed >= args.duration:
            logging.debug(f"Stopping: duration {elapsed:.1f}s >= {args.duration}s")
            return False
    
    # Default: continue
    return True


# ───────────────────────────────────────────────────────────────────────
# 5.  BENCHMARKS  ───────────────────────────────────────────────────────
def batched_gemm_test(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    try:
        dtype = getattr(torch, a.precision_gemm)
        dtype_str = str(dtype).split(".")[-1]
        if a.batched_gemm_TF32_mode and dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            dtype = torch.float32
            dtype_str = "TF32"

        B, M, N, K = a.batch_count_gemm, a.m, a.n, a.k
        dev_lbl = device_label(dev, gpu_id)
        log.info(f"[{dev_lbl} GEMM] Allocating tensors ({B}x{M}x{K} + {B}x{K}x{N})...")
        A, Bm = (
            torch.rand(B, M, K, device=dev, dtype=dtype),
            torch.rand(B, K, N, device=dev, dtype=dtype),
        )
        flops = 2 * B * M * N * K * (4 if dtype.is_complex else 1)

        log.info(f"[{dev_lbl} GEMM] Warmup ({a.warmup} iterations)...")
        for _ in range(a.warmup):
            torch.matmul(A, Bm)
        log.info(f"[{dev_lbl} GEMM] Starting timed iterations...")

        tel_thread.set_active(True)  # Fast sampling during benchmark
        vals = []
        start_time = time.perf_counter()
        i = 0
        max_iters = a.max_iterations if a.max_iterations else a.inner_loop_batched_gemm
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                torch.matmul(A, Bm)  # PURE GPU work - zero interrupts!
            tel_thread.mark_iteration_end(i)
            val = gflops(flops, t.elapsed)
            vals.append(val)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "gemm", dtype_str, "gflops", val, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Batched GEMM] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, a.inner_loop_batched_gemm // 4) == 0:
                log.info(f"[{dev_lbl} Batched GEMM] Progress: {i+1}/{a.inner_loop_batched_gemm} iterations")
            i += 1
        tel_thread.set_active(False)  # Back to slow sampling
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        params = {
            'dtype': dtype_str,
            'batch': B,
            'm': M,
            'n': N,
            'k': K,
            'tf32': a.batched_gemm_TF32_mode if dev.type == "cuda" else False
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Batched GEMM", vals, "GFLOP/s", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        dev_label_str = device_label(dev, gpu_id)
        log.error(f"[{dev_label_str} Batched GEMM] Failed: {e}")
        import traceback
        log.error(f"[{dev_label_str} Batched GEMM] Traceback: {traceback.format_exc()}")
        return None


def convolution_test(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    try:
        B, Ci, Co, H, W, K = (
            a.batch_count_convolution,
            a.in_channels,
            a.out_channels,
            a.height,
            a.width,
            a.kernel_size,
        )
        dtype = getattr(torch, a.precision_convolution)
        dtype_str = str(dtype).split(".")[-1]
        dev_lbl = device_label(dev, gpu_id)
        log.info(f"[{dev_lbl} Conv] Allocating tensors ({B}x{Ci}x{H}x{W})...")
        x = torch.rand(B, Ci, H, W, device=dev, dtype=dtype)
        conv = torch.nn.Conv2d(Ci, Co, K).to(dev, dtype=dtype)
        log.info(f"[{dev_lbl} Conv] Warmup ({a.warmup} iterations)...")
        for _ in range(a.warmup):
            conv(x)
        log.info(f"[{dev_lbl} Conv] Starting timed iterations...")

        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        if a.duration:
            max_iters = a.max_iterations if a.max_iterations else float('inf')
        else:
            max_iters = a.max_iterations if a.max_iterations else a.inner_loop_convolution
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                conv(x)
            tel_thread.mark_iteration_end(i)
            # t.elapsed is in ms, so (1000 * B) / ms = B / second = images/second
            val = 1000 * B / t.elapsed
            vals.append(val)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "conv", dtype_str, "img_s", val, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Convolution] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, a.inner_loop_convolution // 4) == 0:
                log.info(f"[{dev_lbl} Convolution] Progress: {i+1}/{a.inner_loop_convolution} iterations")
            i += 1
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        params = {
            'dtype': dtype_str,
            'batch': B,
            'in_channels': Ci,
            'out_channels': Co,
            'height': H,
            'width': W,
            'kernel': K
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Convolution", vals, "img/s", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        dev_label_str = device_label(dev, gpu_id)
        log.error(f"[{dev_label_str} Convolution] Failed: {e}")
        import traceback
        log.error(f"[{dev_label_str} Convolution] Traceback: {traceback.format_exc()}")
        return None


def fft_test(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    dev_lbl = device_label(dev, gpu_id)
    try:
        B, NX, NY, NZ = a.batch_count_fft, a.nx, a.ny, a.nz
        dtype = getattr(torch, a.precision_fft)
        dtype_str = str(dtype).split(".")[-1]

        log.info(f"[{dev_lbl} FFT] Allocating tensors ({B}x{NX}x{NY}x{NZ})...")
        x = torch.rand(B, NX, NY, NZ, device=dev, dtype=dtype)
        # FFT flops: 5*N*log2(N) per dimension - assumes power-of-2 for radix-2 FFT
        # Non-power-of-2 uses Bluestein/mixed-radix (formula is approximate)
        flops = 5 * B * NX * NY * NZ * (NX.bit_length() - 1)

        log.info(f"[{dev_lbl} FFT] Warmup ({a.warmup} iterations)...")
        for _ in range(a.warmup):
            torch.fft.fftn(x)
        log.info(f"[{dev_lbl} FFT] Starting timed iterations...")

        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        if a.duration:
            max_iters = a.max_iterations if a.max_iterations else float('inf')
        else:
            max_iters = a.max_iterations if a.max_iterations else a.inner_loop_fft
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                torch.fft.fftn(x)
            tel_thread.mark_iteration_end(i)
            val = gflops(flops, t.elapsed)
            vals.append(val)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "fft3d", dtype_str, "gflops", val, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} 3-D FFT] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, a.inner_loop_fft // 4) == 0:
                log.info(f"[{dev_lbl} 3-D FFT] Progress: {i+1}/{a.inner_loop_fft} iterations")
            i += 1
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        params = {
            'dtype': dtype_str,
            'batch': B,
            'nx': NX,
            'ny': NY,
            'nz': NZ
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("3D FFT", vals, "GFLOP/s", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        dev_label_str = device_label(dev, gpu_id)
        log.error(f"[{dev_label_str} 3-D FFT] Failed: {e}")
        import traceback
        log.error(f"[{dev_label_str} 3-D FFT] Traceback: {traceback.format_exc()}")
        return None


def einsum_test(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    try:
        B, H, Q, D = a.batch_count_einsum, a.heads, a.seq_len, a.d_model
        dtype = getattr(torch, a.precision_einsum)
        dtype_str = str(dtype).split(".")[-1]
        dev_lbl = device_label(dev, gpu_id)

        log.info(f"[{dev_lbl} Einsum] Allocating tensors ({B}x{H}x{Q}x{D})...")
        q = torch.rand(B, H, Q, D, device=dev, dtype=dtype)
        k = torch.rand(B, H, Q, D, device=dev, dtype=dtype)
        flops = 2 * B * H * Q * Q * D

        log.info(f"[{dev_lbl} Einsum] Warmup ({a.warmup} iterations)...")
        for _ in range(a.warmup):
            torch.einsum("bhqd,bhkd->bhqk", q, k)
        log.info(f"[{dev_lbl} Einsum] Starting timed iterations...")

        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        if a.duration:
            max_iters = a.max_iterations if a.max_iterations else float('inf')
        else:
            max_iters = a.max_iterations if a.max_iterations else a.inner_loop_einsum
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                torch.einsum("bhqd,bhkd->bhqk", q, k)
            tel_thread.mark_iteration_end(i)
            val = gflops(flops, t.elapsed)
            vals.append(val)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "einsum", dtype_str, "gflops", val, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Einsum] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, a.inner_loop_einsum // 4) == 0:
                log.info(f"[{dev_lbl} Einsum] Progress: {i+1}/{a.inner_loop_einsum} iterations")
            i += 1
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        params = {
            'dtype': dtype_str,
            'batch': B,
            'heads': H,
            'seq_len': Q,
            'head_dim': D
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Einsum Attention", vals, "GFLOP/s", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        dev_label_str = device_label(dev, gpu_id)
        log.error(f"[{dev_label_str} Einsum] Failed: {e}")
        import traceback
        log.error(f"[{dev_label_str} Einsum] Traceback: {traceback.format_exc()}")
        return None


def memory_traffic_test(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    dev_lbl = device_label(dev, gpu_id)
    try:
        N, ITR = a.memory_size, a.memory_iterations
        dtype = getattr(torch, a.precision_memory)
        dtype_str = str(dtype).split(".")[-1]

        log.info(f"[{dev_lbl} Memory] Allocating tensor ({N} elements)...")
        data = torch.rand(N, device=dev, dtype=dtype)
        log.info(f"[{dev_lbl} Memory] Starting {a.memory_pattern} pattern test...")
        
        # Define access pattern based on --memory-pattern
        pattern = a.memory_pattern
        if pattern == "random":
            # Random indexing - stresses memory system with unpredictable access
            def phase():
                idx = torch.randint(0, N, (N,), device=dev)
                data[idx] += data
            bytes_iter = data.element_size() * N * 2
        elif pattern == "streaming":
            # Sequential streaming - measures peak bandwidth with predictable access
            chunk_size = N // 100  # Process in chunks for better measurement
            def phase():
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    data[start:end] += 1.0
            bytes_iter = data.element_size() * N * 2  # Read + write
        elif pattern == "unit":
            # Unit-stride sequential - optimal for cache/coalescing
            def phase():
                data[:] = data[:] * 1.01 + 0.01
            bytes_iter = data.element_size() * N * 2
        else:
            log.error(f"Unknown memory pattern: {pattern}")
            return None

        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        if a.duration:
            max_iters = a.max_iterations if a.max_iterations else float('inf')
        else:
            max_iters = a.max_iterations if a.max_iterations else a.inner_loop_memory_traffic
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                for _ in range(ITR):
                    phase()
            tel_thread.mark_iteration_end(i)
            val = bytes_iter * ITR / t.elapsed / 1e9
            vals.append(val)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "mem", dtype_str, "gb_s", val, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Memory traffic] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, a.inner_loop_memory_traffic // 4) == 0:
                log.info(f"[{dev_lbl} Memory traffic] Progress: {i+1}/{a.inner_loop_memory_traffic} iterations")
            i += 1
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        params = {
            'dtype': dtype_str,
            'size': N,
            'pattern': pattern,
            'iterations': ITR
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Memory Traffic", vals, "GB/s", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        log.error(f"[{dev_lbl} Memory traffic] Failed: {e}")
        import traceback
        log.error(f"[{dev_lbl} Memory traffic] Traceback: {traceback.format_exc()}")
        return None


def laplacian_heat_equation(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    dev_lbl = device_label(dev, gpu_id)
    try:
        N, steps = a.heat_grid_size, a.heat_time_steps
        dtype = getattr(torch, a.precision_heat)
        dtype_str = str(dtype).split(".")[-1]

        log.info(f"[{dev_lbl} Heat] Allocating grid ({N}x{N})...")
        u = torch.rand(N, N, device=dev, dtype=dtype)
        coeff = a.alpha * a.delta_t
        log.info(f"[{dev_lbl} Heat] Starting simulation ({steps} time steps)...")

        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        max_iters = a.max_iterations if a.max_iterations else steps
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                u_new = u.clone()
                u_new[1:-1, 1:-1] = (
                    u[1:-1, 1:-1]
                    + coeff
                    * (
                        u[2:, 1:-1]
                        + u[:-2, 1:-1]
                        + u[1:-1, 2:]
                        + u[1:-1, :-2]
                        - 4 * u[1:-1, 1:-1]
                    )
                )
                u = u_new
            tel_thread.mark_iteration_end(i)
            mlups = (N - 2) ** 2 / t.elapsed / 1e6
            vals.append(mlups)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "heat", dtype_str, "mlups", mlups, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Heat Equation] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, steps // 4) == 0:
                log.info(f"[{dev_lbl} Heat Equation] Progress: {i+1}/{steps} time steps")
            i += 1
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        
        params = {
            'dtype': dtype_str,
            'grid_size': N,
            'time_steps': steps,
            'alpha': a.alpha,
            'delta_t': a.delta_t
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Heat Equation", vals, "MLUPS", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        log.error(f"[{dev_lbl} Heat Equation] Failed: {e}")
        import traceback
        log.error(f"[{dev_lbl} Heat Equation] Traceback: {traceback.format_exc()}")
        return None


def schrodinger_equation(a, dev, log, tel, tel_thread, prn):
    gpu_id = tel_thread.get_latest().get('device_id', '?')
    dev_lbl = device_label(dev, gpu_id)
    try:
        N, steps = a.schrodinger_grid_size, a.schrodinger_time_steps
        dx, dt, ħ, m = (
            a.schrodinger_delta_x,
            a.schrodinger_delta_t,
            a.schrodinger_hbar,
            a.schrodinger_mass,
        )
        dtype = getattr(torch, a.precision_schrodinger)
        dtype_str = str(dtype).split(".")[-1]

        log.info(f"[{dev_lbl} Schrödinger] Allocating grid ({N} points)...")
        x = torch.linspace(-10, 10, N, device=dev, dtype=dtype)
        ψ = torch.exp(-x**2) * torch.exp(1j * x)
        if a.schrodinger_potential == "harmonic":
            V = 0.5 * x**2
        else:
            V = torch.where(torch.abs(x) < 1, torch.tensor(0, dtype=dtype, device=dev), torch.tensor(10, dtype=dtype, device=dev))

        Ck, Cv = -1j * ħ / (2 * m * dx**2), -1j * dt / ħ
        log.info(f"[{dev_lbl} Schrödinger] Starting simulation ({steps} time steps)...")

        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        max_iters = a.max_iterations if a.max_iterations else steps
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                lap = (ψ.roll(-1, 0) - 2 * ψ + ψ.roll(1, 0)) / dx**2
                ψ += dt * (Ck * lap + Cv * V * ψ)
            tel_thread.mark_iteration_end(i)
            iter_per_sec = 1.0 / t.elapsed
            vals.append(iter_per_sec)
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "schrod", dtype_str, "iter/s", iter_per_sec, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Schrödinger Equation] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            elif not a.duration and (i + 1) % max(1, steps // 4) == 0:
                log.info(f"[{dev_lbl} Schrödinger Equation] Progress: {i+1}/{steps} time steps")
            i += 1
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        
        params = {
            'dtype': dtype_str,
            'grid_size': N,
            'time_steps': steps,
            'potential': a.schrodinger_potential
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Schrödinger Equation", vals, "iter/s", log, tel, dev, params, baselines, verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        log.error(f"[{dev_lbl} Schrödinger Equation] Failed: {e}")
        import traceback
        log.error(f"[{dev_lbl} Schrödinger Equation] Traceback: {traceback.format_exc()}")
        return None


def atomic_contention_test(a, dev, log, tel, tel_thread, prn):
    """Stress L2 cache atomic units with high-contention scatter_add operations.
    
    This test creates contention by having many threads repeatedly update 
    a small target array, stressing atomic operations and L2 cache.
    """
    gpu_id = tel.read().get('device_id', tel_thread.get_latest().get('device_id', '?'))
    dev_lbl = device_label(dev, gpu_id)
    try:
        N = a.atomic_target_size
        num_updates = a.atomic_num_updates
        dtype = getattr(torch, a.precision_atomic)
        dtype_str = str(dtype).split(".")[-1]
        
        # Check dtype support for scatter_add_
        # scatter_add_ supports: float16, bfloat16, float32, float64 (no complex, no int)
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        if dtype not in supported_dtypes:
            log.warning(f"[{dev_lbl} Atomic Contention] dtype {dtype_str} not supported for scatter_add_, skipping")
            return None
        
        # Test if dtype is actually available on this device
        try:
            test_tensor = torch.zeros(10, device=dev, dtype=dtype)
            test_idx = torch.zeros(10, device=dev, dtype=torch.long)
            test_src = torch.ones(10, device=dev, dtype=dtype)
            test_tensor.scatter_add_(0, test_idx, test_src)
            del test_tensor, test_idx, test_src
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except (RuntimeError, TypeError) as e:
            log.warning(f"[{dev_lbl} Atomic Contention] dtype {dtype_str} not supported on this device: {e}")
            return None
        
        # Estimate memory requirements before allocation
        bytes_per_elem = dtype.itemsize if hasattr(dtype, 'itemsize') else 4
        est_target_mb = N * bytes_per_elem / 1024**2
        est_indices_mb = num_updates * 8 / 1024**2  # int64 indices
        est_src_mb = num_updates * bytes_per_elem / 1024**2
        est_total_mb = est_target_mb + est_indices_mb + est_src_mb
        
        if dev.type == "cuda":
            try:
                free_mb = torch.cuda.mem_get_info(dev.index if dev.index is not None else 0)[0] / 1024**2
                if est_total_mb * 1.2 > free_mb:  # 20% safety margin
                    log.warning(f"[{dev_lbl} Atomic Contention] Estimated memory {est_total_mb:.0f}MB exceeds available {free_mb:.0f}MB, skipping")
                    return None
            except Exception as e:
                log.debug(f"[{dev_lbl} Atomic Contention] Could not check memory: {e}")
        
        # Target tensor (small to maximize contention)
        log.info(f"[{dev_lbl} Atomic] Allocating tensors ({N} target, {num_updates} updates)...")
        target = torch.zeros(N, device=dev, dtype=dtype)
        
        # Index tensor - many threads hitting same locations (high contention)
        # All indices point to a small range to maximize atomic contention
        contention_range = min(N, a.atomic_contention_range)  # Configurable contention range
        indices = torch.randint(0, contention_range, (num_updates,), device=dev)
        src = torch.ones(num_updates, device=dev, dtype=dtype)
        
        log.info(f"[{dev_lbl} Atomic] Starting: {num_updates} updates to {N} element target (contention range: {contention_range})")
        
        # Warmup
        log.info(f"[{dev_lbl} Atomic] Warmup ({a.warmup} iterations)...")
        for _ in range(a.warmup):
            target.zero_()
            target.scatter_add_(0, indices, src)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        log.info(f"[{dev_lbl} Atomic] Starting timed iterations...")
        
        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        max_iters = a.max_iterations if a.max_iterations else a.inner_loop_atomic
        
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            target.zero_()
            with Timer(dev) as t:
                target.scatter_add_(0, indices, src)
            tel_thread.mark_iteration_end(i)
            
            # Measure atomic ops per second (millions)
            mops = num_updates / t.elapsed / 1e6
            vals.append(mops)
            
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "atomic", dtype_str, "mops", mops, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Atomic Contention] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            i += 1
        
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        
        params = {
            'dtype': dtype_str,
            'target_size': N,
            'num_updates': num_updates,
            'contention_range': contention_range
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Atomic Contention", vals, "Mops/s", log, tel, dev, params, baselines, 
                           verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        log.error(f"[{dev_lbl} Atomic Contention] Failed: {e}")
        import traceback
        log.error(f"[{dev_lbl} Atomic Contention] Traceback: {traceback.format_exc()}")
        return None


def sparse_mm_test(a, dev, log, tel, tel_thread, prn):
    """Sparse matrix multiplication stress test (SpMM).
    
    Tests sparse tensor core performance with COO format sparse matrices.
    Important for GNN workloads and sparse transformer patterns.
    """
    gpu_id = tel.read().get('device_id', tel_thread.get_latest().get('device_id', '?'))
    dev_lbl = device_label(dev, gpu_id)
    try:
        M, N, K = a.sparse_m, a.sparse_n, a.sparse_k
        density = a.sparse_density
        dtype = getattr(torch, a.precision_sparse)
        dtype_str = str(dtype).split(".")[-1]
        
        # Calculate bytes per element
        dtype_bytes = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8}
        bytes_per_elem = dtype_bytes.get(dtype_str, 4)
        
        # Estimate memory requirements before allocating
        nnz = max(1, int(M * K * density))
        sparse_mem = nnz * (bytes_per_elem + 16)  # values + 2x int64 indices
        dense_mem = K * N * bytes_per_elem  # Dense B matrix
        output_mem = M * N * bytes_per_elem  # Output C matrix
        total_est_mb = (sparse_mem + dense_mem + output_mem) / 1024**2
        
        # Check available memory
        if dev.type == "cuda":
            try:
                free_mb = torch.cuda.mem_get_info(dev.index if dev.index is not None else 0)[0] / 1024**2
                if total_est_mb > free_mb * 0.90:  # 90% of free memory
                    log.error(f"[{dev_lbl} Sparse MM] Estimated memory {total_est_mb:.0f}MB exceeds available {free_mb:.0f}MB, skipping")
                    log.error(f"[{dev_lbl} Sparse MM] Try smaller sizes: --sparse-m/n/k or use --stress-test for auto-sizing")
                    return None
            except Exception as e:
                log.warning(f"[{dev_lbl} Sparse MM] Could not check memory: {e}")
        
        log.info(f"[{dev_lbl} Sparse MM] Estimated memory: {total_est_mb:.0f}MB (sparse:{sparse_mem/1024**2:.0f} + dense:{dense_mem/1024**2:.0f} + output:{output_mem/1024**2:.0f})")
        
        # Check dtype support for sparse operations
        # torch.sparse.mm supports: float32, float64 (limited support for float16/bfloat16)
        supported_dtypes = {torch.float32, torch.float64}
        # Try to extend support for half precision if available
        if dtype in {torch.float16, torch.bfloat16}:
            try:
                # Test if sparse ops work with this dtype
                test_indices = torch.tensor([[0, 1], [0, 1]], device=dev)
                test_values = torch.tensor([1.0, 1.0], device=dev, dtype=dtype)
                test_sparse = torch.sparse_coo_tensor(test_indices, test_values, (2, 2)).coalesce()
                test_dense = torch.ones(2, 2, device=dev, dtype=dtype)
                _ = torch.sparse.mm(test_sparse, test_dense)
                del test_sparse, test_dense, test_indices, test_values
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                supported_dtypes.add(dtype)  # It works!
            except (RuntimeError, TypeError):
                pass  # dtype not supported
        
        if dtype not in supported_dtypes:
            log.warning(f"[{dev_lbl} Sparse MM] dtype {dtype_str} not supported for sparse operations, skipping")
            return None
        
        log.info(f"[{dev_lbl} Sparse MM] Creating {M}x{K} sparse matrix with {density*100:.1f}% density ({nnz} non-zeros)")
        
        # Create sparse matrix in COO format
        log.info(f"[{dev_lbl} Sparse MM] Allocating index tensors...")
        row_indices = torch.randint(0, M, (nnz,), device=dev)
        col_indices = torch.randint(0, K, (nnz,), device=dev)
        indices = torch.stack([row_indices, col_indices])
        log.info(f"[{dev_lbl} Sparse MM] Allocating value tensor...")
        values = torch.rand(nnz, device=dev, dtype=dtype)
        
        log.info(f"[{dev_lbl} Sparse MM] Creating sparse COO tensor...")
        sparse_A = torch.sparse_coo_tensor(indices, values, (M, K))
        log.info(f"[{dev_lbl} Sparse MM] Coalescing sparse tensor (this may take a while)...")
        sparse_A = sparse_A.coalesce()
        
        # Dense matrix B
        log.info(f"[{dev_lbl} Sparse MM] Allocating dense B matrix ({K}x{N})...")
        dense_B = torch.rand(K, N, device=dev, dtype=dtype)
        
        # Actual nnz after coalesce (duplicates removed)
        actual_nnz = sparse_A._nnz()
        
        log.info(f"[{dev_lbl} Sparse MM] Starting: ({M}x{K}) sparse @ ({K}x{N}) dense, nnz={actual_nnz}")
        
        # Warmup
        log.info(f"[{dev_lbl} Sparse MM] Starting warmup ({a.warmup} iterations)...")
        for wi in range(a.warmup):
            log.info(f"[{dev_lbl} Sparse MM] Warmup iteration {wi+1}/{a.warmup}...")
            _ = torch.sparse.mm(sparse_A, dense_B)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        log.info(f"[{dev_lbl} Sparse MM] Warmup complete, starting timed iterations...")
        
        # Effective FLOPS: 2 * nnz * N (each non-zero contributes 2 ops per output column)
        flops = 2 * actual_nnz * N
        
        tel_thread.set_active(True)
        vals = []
        start_time = time.perf_counter()
        i = 0
        max_iters = a.max_iterations if a.max_iterations else a.inner_loop_sparse
        
        while i < max_iters and should_continue_iterations(a, i, vals, start_time):
            tel_thread.mark_iteration_start(i)
            with Timer(dev) as t:
                _ = torch.sparse.mm(sparse_A, dense_B)
            tel_thread.mark_iteration_end(i)
            
            val = gflops(flops, t.elapsed)
            vals.append(val)
            
            if a.verbose:
                tel_data = tel_thread.get_iteration_telemetry(i)
                prn.emit(i, "sparse_mm", dtype_str, "gflops", val, tel_data)
            elif a.duration and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                log.info(f"[{dev_lbl} Sparse MM] Progress: {i+1} iterations, {elapsed:.1f}s elapsed")
            i += 1
        
        tel_thread.set_active(False)
        
        # Check for thermal warnings
        if hasattr(tel, 'check_thermal_warnings'):
            tel.check_thermal_warnings(log, a.temp_warn_C, a.temp_critical_C, a.power_warn_pct)
        
        params = {
            'dtype': dtype_str,
            'm': M,
            'n': N,
            'k': K,
            'density': density,
            'nnz': actual_nnz
        }
        baselines = getattr(a, '_hardware_baselines', None)
        return _log_summary("Sparse MM", vals, "GFLOP/s", log, tel, dev, params, baselines,
                           verbose=a.verbose, skip_telemetry=a.skip_telemetry_first_n, tel_thread=tel_thread, runtime_s=time.perf_counter() - start_time)
    except Exception as e:
        log.error(f"[{dev_lbl} Sparse MM] Failed: {e}")
        import traceback
        log.error(f"[{dev_lbl} Sparse MM] Traceback: {traceback.format_exc()}")
        return None


# ───────────────────────────────────────────────────────────────────────
# 6.  CLI & LOGGING  ───────────────────────────────────────────────────
def load_config(config_path):
    """Load YAML configuration file."""
    if not YAML_AVAILABLE:
        print("Error: PyYAML is not installed. Install with: pip install pyyaml")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)


def apply_config_to_args(args, config):
    """Apply configuration file settings to args namespace. CLI args take precedence."""
    if not config:
        return args
    
    # Detect which args were explicitly set on CLI by checking sys.argv
    import sys
    cli_args_set = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            cli_args_set.add(arg.lstrip('-').replace('-', '_'))
    
    # Apply global settings if not overridden by CLI
    global_settings = config.get('global', {})
    for key, value in global_settings.items():
        arg_name = key.replace('-', '_')
        # Skip if explicitly set on CLI
        if arg_name in cli_args_set:
            continue
        # Special case: if --verbose-file-only is set on CLI, don't apply verbose from YAML
        if arg_name == 'verbose' and 'verbose_file_only' in cli_args_set:
            continue
        # Only apply if CLI didn't set it (i.e., still at default)
        if hasattr(args, arg_name):
            current = getattr(args, arg_name)
            # Check if this looks like a default value that wasn't explicitly set
            if arg_name in ['warmup'] and current == 10:  # Default warmup
                setattr(args, arg_name, value)
            elif arg_name in ['verbose', 'no_log', 'stress_test', 'dry_run', 'all_gpus', 'cpu_affinity', 'verbose_file_only']:
                if not current and value:  # Only set True values if current is False
                    setattr(args, arg_name, value)
            elif arg_name in ['log_file', 'log_dir', 'temp_dir'] and not current:
                setattr(args, arg_name, value)
    
    # Apply runtime settings (duration, iterations, thresholds)
    runtime_settings = config.get('runtime', {})
    for key, value in runtime_settings.items():
        arg_name = key.replace('-', '_')
        if hasattr(args, arg_name) and getattr(args, arg_name) is None or \
           (hasattr(args, arg_name) and isinstance(getattr(args, arg_name), (int, float)) and 
            arg_name in ['duration', 'min_iterations', 'max_iterations',
                         'temp_warn_C', 'temp_critical_C', 'power_warn_pct', 
                         'outlier_threshold_pct', 'efficiency_warn_pct']):
            # For numeric thresholds, check if still at default before overriding
            parser_defaults = {
                'temp_warn_C': 90.0,
                'temp_critical_C': 95.0,
                'power_warn_pct': 98.0,
                'outlier_threshold_pct': 15.0,
                'efficiency_warn_pct': 70.0,
                'min_iterations': 10
            }
            current = getattr(args, arg_name)
            if current is None or (arg_name in parser_defaults and current == parser_defaults[arg_name]):
                setattr(args, arg_name, value)
    
    # Store benchmark list for multiple invocations
    benchmarks = config.get('benchmarks', [])
    benchmark_list = []
    valid_benchmark_names = {'batched_gemm', 'convolution', 'fft', 'einsum', 'memory_traffic', 'heat_equation', 'schrodinger', 'atomic_contention', 'sparse_mm'}
    
    for idx, bench_config in enumerate(benchmarks):
        if not bench_config.get('enabled', True):
            continue
        
        # Validate benchmark name
        bench_name = bench_config.get('name', '')
        if bench_name not in valid_benchmark_names:
            print(f"WARNING: Config benchmark #{idx+1} has invalid name '{bench_name}'. Valid names: {', '.join(sorted(valid_benchmark_names))}")
            print(f"         This benchmark will be skipped during execution.")
        
        benchmark_list.append(bench_config)
    
    # Store the benchmark list in args
    args.benchmark_list = benchmark_list if benchmark_list else None
    
    # If config has benchmarks section, disable CLI benchmark flags
    # (config benchmarks take precedence over CLI flags to avoid duplication)
    if benchmark_list:
        args.batched_gemm = False
        args.convolution = False
        args.fft = False
        args.einsum = False
        args.memory_traffic = False
        args.heat_equation = False
        args.schrodinger = False
        print(f"Config file specifies {len(benchmark_list)} benchmark(s). CLI benchmark flags disabled.")
    
    # Final override: if --verbose-file-only was set on CLI, ensure verbose stays False
    # (it will be set to True in init_logging() but only for logging level, not stdout)
    if 'verbose_file_only' in cli_args_set and args.verbose_file_only:
        args.verbose = False
    
    return args


def build_parser():
    p = argparse.ArgumentParser("TORCH-HAMMER", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    precision_choices = ["bfloat16", "float16", "float32", "float64", "complex64", "complex128"]
    # global
    p.add_argument("--banner", action="store_true", help="Show ASCII banner at startup")
    # Easter egg: --forge is a hidden flag (not shown in help) for animated banner
    p.add_argument("--forge", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--config", type=str, help="Path to YAML configuration file")
    p.add_argument("--list-profiles", action="store_true", help="List available configuration profiles and exit")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--log-file", type=str)
    p.add_argument("--log-dir", type=str, help="Directory for per-GPU log files (multi-GPU only)")
    p.add_argument("--temp-dir", type=str, help="Directory for temp files (multi-GPU result collection). Falls back to TORCH_HAMMER_TEMP_DIR env var, then system temp.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-file-only", action="store_true", help="With --verbose and --log-file/--log-dir, suppress stdout (file only)")
    p.add_argument("--compact", action="store_true", help="Machine-readable CSV output to stdout (one row per benchmark). Suppresses normal log chatter. Combine with --verbose for extra telemetry columns.")
    p.add_argument("--dry-run", action="store_true", help="Show configuration and exit without running benchmarks")
    p.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the entire benchmark suite")
    p.add_argument("--repeat-delay", type=float, default=0, help="Delay in seconds between repeats (for thermal stabilization)")
    p.add_argument("--stress-test", action="store_true", help="Automatically calculate maximum stress parameters based on available memory")
    p.add_argument("--shuffle", action="store_true", help="Randomize benchmark execution order")
    p.add_argument("--device-index", type=int, default=0, help="Single GPU index (ignored if --all-gpus or --gpu-list)")
    p.add_argument("--all-gpus", action="store_true", help="Run on all available GPUs")
    p.add_argument("--gpu-list", type=str, help="Comma-separated GPU indices (e.g., '0,2,3')")
    p.add_argument("--cpu-affinity", action="store_true", default=True, help="Enable NUMA-aware CPU binding (default: enabled)")
    p.add_argument("--no-cpu-affinity", action="store_false", dest="cpu_affinity", help="Disable NUMA-aware CPU binding")
    p.add_argument("--cpu-gpu-map", type=str, help="Manual CPU-GPU mapping (e.g., '0:0-15,1:16-31')")
    p.add_argument("--cpu-list", type=str, help="CPU cores for CPU-only mode (e.g., '0-23,48-71' or 'all'). Default: all physical cores")
    p.add_argument("--parent-cpu", type=int, default=None, help="Pin parent process to specific CPU core (default: last core of first NUMA node, -1 to disable)")
    p.add_argument("--startup-delay-per-gpu", type=float, default=0, help="Staggered startup delay in seconds per GPU (GPU N waits N*delay seconds). Helps avoid ROCm memory allocator contention.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--skip-telemetry-first-n", type=int, default=10, help="Skip first N telemetry readings when calculating statistics (default: 10)")
    p.add_argument("--telemetry-interval-ms", type=int, default=100, help="Telemetry polling interval in milliseconds (default: 100). Higher values reduce telemetry resolution but may improve performance on some systems.")
    p.add_argument("--no-telemetry-thread", action="store_true", help="Disable background telemetry thread entirely (for debugging performance issues)")
    p.add_argument("--duration", type=float, help="Run for specified duration in seconds (overrides iteration counts)")
    p.add_argument("--min-iterations", type=int, default=10, help="Minimum iterations even if duration met")
    p.add_argument("--max-iterations", type=int, help="Maximum iterations regardless of duration")
    p.add_argument("--json-output", type=str, help="Path to output JSON file with all results and telemetry")
    p.add_argument("--summary-csv", type=str, help="Path to output CSV file with benchmark summary table")
    
    # Thermal/Performance thresholds (tunable across platforms)
    p.add_argument("--temp-warn-C", type=float, default=90.0, help="Temperature warning threshold in Celsius (default: 90)")
    p.add_argument("--temp-critical-C", type=float, default=95.0, help="Temperature critical threshold in Celsius (default: 95)")
    p.add_argument("--power-warn-pct", type=float, default=98.0, help="Power limit warning threshold in percent (default: 98)")
    p.add_argument("--outlier-threshold-pct", type=float, default=15.0, help="Multi-GPU outlier detection threshold in percent (default: 15)")
    p.add_argument("--efficiency-warn-pct", type=float, default=70.0, help="Hardware efficiency warning threshold in percent (default: 70)")
    p.add_argument("--baseline-file", type=str, help="Load hardware baselines from JSON/YAML file (optional)")
    p.add_argument("--no-validation", action="store_true", help="Disable hardware performance validation")
    
    # batched gemm
    p.add_argument("--batched-gemm", action="store_true")
    p.add_argument("--batch-count-gemm", type=int, default=128)
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--inner-loop-batched-gemm", type=int, default=10)
    p.add_argument("--precision-gemm", default="float32", choices=precision_choices)
    p.add_argument("--batched-gemm-TF32-mode", action="store_true")
    # convolution
    p.add_argument("--convolution", action="store_true")
    p.add_argument("--batch-count-convolution", type=int, default=128)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--out-channels", type=int, default=64)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--inner-loop-convolution", type=int, default=10)
    p.add_argument("--precision-convolution", default="float32", choices=precision_choices)
    # fft
    p.add_argument("--fft", action="store_true")
    p.add_argument("--batch-count-fft", type=int, default=128)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=128)
    p.add_argument("--inner-loop-fft", type=int, default=10)
    p.add_argument("--precision-fft", default="float32", choices=precision_choices)
    # einsum
    p.add_argument("--einsum", action="store_true")
    p.add_argument("--batch-count-einsum", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--inner-loop-einsum", type=int, default=10)
    p.add_argument("--precision-einsum", default="float32", choices=precision_choices)
    # memory
    p.add_argument("--memory-traffic", action="store_true")
    p.add_argument("--memory-size", type=int, default=1024)
    p.add_argument("--memory-iterations", type=int, default=10)
    p.add_argument("--memory-pattern", default="random", choices=["random", "streaming", "unit"], 
                   help="Memory access pattern: random (random indexing), streaming (sequential), unit (stride-1)")
    p.add_argument("--inner-loop-memory-traffic", type=int, default=10)
    p.add_argument("--precision-memory", default="float32", choices=precision_choices)
    # heat
    p.add_argument("--heat-equation", action="store_true")
    p.add_argument("--heat-grid-size", type=int, default=128)
    p.add_argument("--heat-time-steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--delta-t", type=float, default=0.01)
    p.add_argument("--precision-heat", default="float32", choices=precision_choices)
    p.add_argument("--inner-loop-heat-equation", type=int, default=10)
    # schrod
    p.add_argument("--schrodinger", action="store_true")
    p.add_argument("--schrodinger-grid-size", type=int, default=128)
    p.add_argument("--schrodinger-time-steps", type=int, default=100)
    p.add_argument("--schrodinger-delta-x", type=float, default=0.1)
    p.add_argument("--schrodinger-delta-t", type=float, default=0.01)
    p.add_argument("--schrodinger-hbar", type=float, default=1.0)
    p.add_argument("--schrodinger-mass", type=float, default=1.0)
    p.add_argument("--schrodinger-potential", default="harmonic", choices=["harmonic", "barrier"])
    p.add_argument("--precision-schrodinger", default="float32", choices=precision_choices)
    p.add_argument("--inner-loop-schrodinger", type=int, default=10)
    # atomic contention (L2 cache stress)
    p.add_argument("--atomic-contention", action="store_true", help="Stress L2 cache atomic units with scatter_add operations")
    p.add_argument("--atomic-target-size", type=int, default=1_000_000, help="Size of target array for atomic updates")
    p.add_argument("--atomic-num-updates", type=int, default=10_000_000, help="Number of atomic scatter_add updates per iteration")
    p.add_argument("--atomic-contention-range", type=int, default=1024, help="Max unique indices for contention (lower = more contention, higher = less)")
    p.add_argument("--inner-loop-atomic", type=int, default=50)
    p.add_argument("--precision-atomic", default="float32", choices=["float16", "bfloat16", "float32", "float64"])
    # sparse matrix multiply
    p.add_argument("--sparse-mm", action="store_true", help="Sparse matrix multiplication (SpMM) stress test")
    p.add_argument("--sparse-m", type=int, default=8192, help="Sparse matrix rows")
    p.add_argument("--sparse-n", type=int, default=8192, help="Dense matrix columns / output columns")
    p.add_argument("--sparse-k", type=int, default=8192, help="Sparse matrix cols / Dense matrix rows")
    p.add_argument("--sparse-density", type=float, default=0.10, help="Sparse matrix density (0.10 = 10%% non-zeros, higher = more stress)")
    p.add_argument("--inner-loop-sparse", type=int, default=50)
    p.add_argument("--precision-sparse", default="float32", choices=["float16", "bfloat16", "float32", "float64"])
    return p

def init_logging(a, gpu_index=None, tel_data=None):
    """Set up logging. Returns logger.
    
    Args:
        a: Arguments namespace
        gpu_index: GPU device index (optional)
        tel_data: Telemetry data dict with 'hostname', 'serial', 'model' keys (optional)
    """
    if a.no_log:
        logging.disable(logging.CRITICAL)
        return logging.getLogger("nul")
    
    # --verbose-file-only implies verbose mode
    if a.verbose_file_only:
        a.verbose = True
    
    # Compact mode: suppress normal log output (CSV goes to stdout separately).
    # Use WARNING so only real problems appear on stderr.
    compact = getattr(a, 'compact', False)
    if compact:
        level = logging.WARNING
    else:
        level = logging.DEBUG if a.verbose else logging.INFO
    # Determine handlers based on log file settings
    handlers = []
    
    # In compact mode route log messages to stderr so stdout stays pure CSV
    if compact:
        handlers.append(logging.StreamHandler(sys.stderr))
    # Add stdout handler unless --verbose-file-only is set
    elif not a.verbose_file_only:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    # Per-GPU log files for multi-GPU runs
    if a.log_dir and gpu_index is not None:
        Path(a.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Build filename with available telemetry data
        # Determine device type from telemetry
        device_type = "cpu"
        if tel_data:
            vendor = tel_data.get('vendor', '').upper()
            if vendor in ['NVIDIA', 'AMD', 'INTEL']:
                device_type = "gpu"
            elif vendor == 'APPLE' or tel_data.get('model', '').lower().find('apple') >= 0:
                device_type = "mps"
        
        filename_parts = [f"{device_type}{gpu_index}"]
        if tel_data:
            if tel_data.get('hostname'):
                filename_parts.append(tel_data['hostname'])
            if tel_data.get('serial'):
                # Truncate long serials for readability
                serial = str(tel_data['serial'])
                if len(serial) > 12:
                    serial = serial[:12]
                filename_parts.append(serial)
        
        filename = "_".join(filename_parts)
        extension = ".csv" if a.verbose else ".log"
        log_path = Path(a.log_dir) / f"{filename}{extension}"
        handlers.append(logging.FileHandler(str(log_path)))
    elif a.log_file:
        handlers.append(logging.FileHandler(a.log_file))
    
    # If no handlers were added (e.g., --verbose-file-only without --log-file), add NullHandler to prevent default stdout
    if not handlers:
        handlers.append(logging.NullHandler())
    
    # Configure the specific logger we'll use, not the root logger
    logger_name = f"torchhammer_gpu{gpu_index}" if gpu_index is not None else "torchhammer"
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers on this logger AND root logger
    # (This is critical to prevent handler accumulation across multiple runs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level and add our handlers
    logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False
    
    return logger


# ───────────────────────────────────────────────────────────────────────
# 7.  MAIN  ─────────────────────────────────────────────────────────────
def run_single_gpu(args, gpu_index: int, log=None):
    """Run benchmarks on a single device (GPU, MPS, or CPU) with optional CPU affinity."""
    import sys
    
    _quiet = getattr(args, 'compact', False)
    
    # Determine device type early for proper labeling
    # (Before we have the actual torch.device object)
    if torch.cuda.is_available():
        early_label = f"GPU{gpu_index}"
    elif torch.backends.mps.is_available():
        early_label = "MPS"
    else:
        early_label = "CPU"
    
    # Staggered startup to avoid ROCm/CUDA memory allocator contention
    # when multiple GPUs initialize simultaneously
    startup_delay = getattr(args, 'startup_delay_per_gpu', 0)
    if startup_delay > 0 and early_label.startswith("GPU"):
        delay = gpu_index * startup_delay
        if not _quiet:
            print(f"[{early_label}] Waiting {delay:.1f}s (staggered startup)...", file=sys.stderr, flush=True)
        time.sleep(delay)
    
    # Early progress message (before logging is set up)
    if not _quiet:
        print(f"[{early_label}] Initializing...", file=sys.stderr, flush=True)
    
    # Set unique process name for identification in ps/top/htop
    if SETPROCTITLE_AVAILABLE:
        import socket
        hostname = socket.gethostname().split('.', 1)[0]
        proc_suffix = f"gpu{gpu_index}" if early_label.startswith("GPU") else early_label.lower()
        setproctitle.setproctitle(f"torch-hammer-{proc_suffix}@{hostname}")
    
    # Set up device first
    if not _quiet:
        print(f"[{early_label}] Setting up PyTorch device...", file=sys.stderr, flush=True)
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(gpu_index)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    
    # Now we have the device, get canonical label
    dev_label = device_label(dev, gpu_index)
    
    # Get telemetry data early for log filename generation
    if not _quiet:
        print(f"[{dev_label}] Initializing telemetry...", file=sys.stderr, flush=True)
    tel = make_telemetry(gpu_index, dev)
    tel_data = tel.read()
    
    # Initialize logging with telemetry data for proper filename
    # In multiprocessing spawn mode, logger needs to be recreated in child process
    if log is None:
        log = init_logging(args, gpu_index, tel_data)
    
    # CPU-only mode: configure optimal threading (uses all physical cores by default)
    if dev.type == "cpu":
        setup_cpu_threading(dev, args, log)
    
    if not _quiet:
        print(f"[{dev_label}] Ready.", file=sys.stderr, flush=True)
    
    log.info(f"Using device {dev}")
    log.info(f"Initial telemetry: {_format_telemetry_compact(tel_data)}")
    log.info(f"  Device: {tel_data.get('model', 'Unknown')} (Serial: {tel_data.get('serial', 'N/A')})")
    
    # Log NUMA topology for debugging multi-GPU performance issues
    numa_node = get_gpu_numa_node(gpu_index)
    numa_cpus = get_numa_cpus(numa_node)
    if dev.type == "cuda":
        log.info(f"  NUMA: GPU {gpu_index} on NUMA node {numa_node}, CPUs: {format_cpu_list(numa_cpus)}")
    else:
        # CPU or MPS mode - NUMA info is about the CPU topology, not GPU
        log.info(f"  NUMA: node {numa_node}, CPUs: {format_cpu_list(numa_cpus)}")
    
    # Handle CPU affinity
    if args.cpu_gpu_map:
        cpu_map = parse_cpu_gpu_map(args.cpu_gpu_map)
        if gpu_index in cpu_map:
            cpus = cpu_map[gpu_index]
            set_cpu_affinity(cpus, quiet=True)
            log.info(f"CPU affinity: pinned to {format_cpu_list(cpus)} (manual)")
    elif args.cpu_affinity:
        # Use intelligent CPU distribution passed from main()
        if hasattr(args, '_cpu_distribution') and gpu_index in args._cpu_distribution:
            cpus = args._cpu_distribution[gpu_index]
            numa_node = get_gpu_numa_node(gpu_index)
            set_cpu_affinity(cpus, quiet=True)
            log.info(f"CPU affinity: NUMA{numa_node} → pinned to {format_cpu_list(cpus)}")
        else:
            # Fallback to old behavior (bind all NUMA CPUs)
            numa_node = get_gpu_numa_node(gpu_index)
            cpus = get_numa_cpus(numa_node)
            if cpus:
                set_cpu_affinity(cpus, quiet=True)
                log.info(f"CPU affinity: NUMA{numa_node} → pinned to {format_cpu_list(cpus)}")
    
    prn = VerbosePrinter(log, tel.schema(), gpu_index)
    
    # Clear statistics from initial idle reading
    tel.reset_stats()
    
    # Start background telemetry thread (unless disabled for debugging)
    # IMPORTANT: Use conservative interval (default 1000ms) to avoid AMD SMI blocking GPU operations
    # AMD amdsmi_get_gpu_metrics_info() can cause PCIe bus contention and GPU stalls
    tel_interval = getattr(args, 'telemetry_interval_ms', 1000)
    no_tel_thread = getattr(args, 'no_telemetry_thread', False)
    
    if no_tel_thread:
        # Create a dummy telemetry thread that doesn't poll
        tel_thread = TelemetryThread(tel, dev, sample_interval_ms=999999)  # Effectively disabled
        # Pre-populate with one reading so get_latest() works
        tel_thread.latest_reading = tel.read()
        log.info("Telemetry thread DISABLED (--no-telemetry-thread)")
    else:
        tel_thread = TelemetryThread(tel, dev, sample_interval_ms=tel_interval)
        tel_thread.start()
        log.info(f"Started background telemetry thread ({tel_interval}ms sampling)")
    
    # Pin main thread to remaining CPUs (excluding telemetry CPU) - only if CPU affinity is enabled
    if args.cpu_affinity or args.cpu_gpu_map:
        import os
        if hasattr(os, 'sched_getaffinity'):
            try:
                current_affinity = os.sched_getaffinity(0)
                if len(current_affinity) > 1:
                    # Use all CPUs except the highest one (reserved for telemetry)
                    main_cpus = current_affinity - {max(current_affinity)}
                    os.sched_setaffinity(0, main_cpus)
                    log.info(f"Main thread pinned to CPUs: {format_cpu_list(sorted(main_cpus))}, telemetry on CPU {max(current_affinity)}")
            except (AttributeError, OSError):
                pass

    # Validate at least one benchmark is selected
    benchmarks_selected = any([
        args.batched_gemm, args.convolution, args.fft, args.einsum,
        args.memory_traffic, args.heat_equation, args.schrodinger,
        args.atomic_contention, args.sparse_mm
    ]) or (hasattr(args, 'benchmark_list') and args.benchmark_list)
    
    if not benchmarks_selected:
        log.warning("No benchmarks selected. Use --batched-gemm, --convolution, --fft, --einsum, --memory-traffic, --heat-equation, --schrodinger, --atomic-contention, or --sparse-mm")
        return
    
    # If duration is specified without max_iterations, allow unlimited iterations
    # (duration will be the stopping criterion)
    if args.duration and not args.max_iterations:
        args.max_iterations = float('inf')
        log.info(f"Duration limit: {args.duration}s (iterations unlimited)")
    
    # Stress test mode: override parameters with memory-optimized values
    if args.stress_test:
        available_mb = get_available_memory_mb(dev)
        log.info(f"[Stress Test] Available memory: {available_mb:.0f} MB")
        
        if args.batched_gemm:
            params = calculate_stress_params("gemm", args.precision_gemm, available_mb, log)
            args.m = params["m"]
            args.n = params["n"]
            args.k = params["k"]
            args.batch_count_gemm = params["batch_count_gemm"]
        
        if args.convolution:
            params = calculate_stress_params("convolution", args.precision_convolution, available_mb, log)
            args.batch_count_convolution = params["batch_count_convolution"]
            args.in_channels = params["in_channels"]
            args.out_channels = params["out_channels"]
            args.height = params["height"]
            args.width = params["width"]
            args.kernel_size = params["kernel_size"]
        
        if args.fft:
            params = calculate_stress_params("fft", args.precision_fft, available_mb, log)
            args.batch_count_fft = params["batch_count_fft"]
            args.nx = params["nx"]
            args.ny = params["ny"]
            args.nz = params["nz"]
        
        if args.einsum:
            params = calculate_stress_params("einsum", args.precision_einsum, available_mb, log)
            args.batch_count_einsum = params["batch_count_einsum"]
            args.heads = params["heads"]
            args.seq_len = params["seq_len"]
            args.d_model = params["d_model"]
        
        if args.memory_traffic:
            params = calculate_stress_params("memory", args.precision_memory, available_mb, log)
            args.memory_size = params["memory_size"]
        
        if args.heat_equation:
            params = calculate_stress_params("heat", args.precision_heat, available_mb, log)
            args.heat_grid_size = params["heat_grid_size"]
        
        if args.schrodinger:
            params = calculate_stress_params("schrodinger", args.precision_schrodinger, available_mb, log)
            args.schrodinger_grid_size = params["schrodinger_grid_size"]
        
        if args.atomic_contention:
            params = calculate_stress_params("atomic", args.precision_atomic, available_mb, log)
            args.atomic_target_size = params["atomic_target_size"]
            args.atomic_num_updates = params["atomic_num_updates"]
        
        if args.sparse_mm:
            params = calculate_stress_params("sparse", args.precision_sparse, available_mb, log, args)
            args.sparse_m = params["sparse_m"]
            args.sparse_n = params["sparse_n"]
            args.sparse_k = params["sparse_k"]
            args.sparse_density = params["sparse_density"]
    
    # Dry-run mode: show configuration and exit
    if args.dry_run:
        log.info("=== DRY RUN MODE ===")
        tel_data = tel.read()
        log.info(f"Device: {dev}")
        log.info(f"GPU Model: {tel_data.get('model', 'N/A')}")
        if 'mem_total_MB' in tel_data:
            log.info(f"Total Memory: {tel_data.get('mem_total_MB', 'N/A')} MB")
        
        selected = []
        if args.batched_gemm:
            tensor_mb = (args.batch_count_gemm * args.m * args.k + args.batch_count_gemm * args.k * args.n) * 4 / 1024**2
            selected.append(f"  - Batched GEMM: {args.batch_count_gemm}x{args.m}x{args.n}x{args.k} ({args.precision_gemm}) ~{tensor_mb:.1f}MB")
        if args.convolution:
            tensor_mb = args.batch_count_convolution * args.in_channels * args.height * args.width * 4 / 1024**2
            selected.append(f"  - Convolution: {args.batch_count_convolution}x{args.in_channels}x{args.height}x{args.width} ~{tensor_mb:.1f}MB")
        if args.fft:
            tensor_mb = args.batch_count_fft * args.nx * args.ny * args.nz * 4 / 1024**2
            selected.append(f"  - 3-D FFT: {args.batch_count_fft}x{args.nx}x{args.ny}x{args.nz} ~{tensor_mb:.1f}MB")
        if args.einsum:
            tensor_mb = 2 * args.batch_count_einsum * args.heads * args.seq_len * args.d_model * 4 / 1024**2
            selected.append(f"  - Einsum: {args.batch_count_einsum}x{args.heads}x{args.seq_len}x{args.d_model} ~{tensor_mb:.1f}MB")
        if args.memory_traffic:
            tensor_mb = args.memory_size * 4 / 1024**2
            selected.append(f"  - Memory Traffic: {args.memory_size} elements ~{tensor_mb:.1f}MB")
        if args.heat_equation:
            tensor_mb = args.heat_grid_size ** 2 * 4 / 1024**2
            selected.append(f"  - Heat Equation: {args.heat_grid_size}x{args.heat_grid_size} grid, {args.heat_time_steps} steps ~{tensor_mb:.1f}MB")
        if args.schrodinger:
            tensor_mb = args.schrodinger_grid_size * 16 / 1024**2  # complex128
            selected.append(f"  - Schrödinger: {args.schrodinger_grid_size} grid, {args.schrodinger_time_steps} steps ~{tensor_mb:.1f}MB")
        if args.atomic_contention:
            tensor_mb = (args.atomic_target_size * 4 + args.atomic_num_updates * 12) / 1024**2  # target + indices + src
            selected.append(f"  - Atomic Contention: {args.atomic_num_updates} updates to {args.atomic_target_size} target ~{tensor_mb:.1f}MB")
        if args.sparse_mm:
            nnz = int(args.sparse_m * args.sparse_k * args.sparse_density)
            tensor_mb = (nnz * 20 + args.sparse_k * args.sparse_n * 4 + args.sparse_m * args.sparse_n * 4) / 1024**2
            selected.append(f"  - Sparse MM: {args.sparse_m}x{args.sparse_k} @ {args.sparse_density*100:.1f}% ({nnz} nnz) ~{tensor_mb:.1f}MB")
        
        log.info("Selected Benchmarks:")
        for s in selected:
            log.info(s)
        log.info("=== END DRY RUN ===")
        tel.shutdown()
        return

    benchmark_results = []
    
    # ── compact-mode helpers (closure over tel_data, args, gpu_index) ──
    _compact = getattr(args, 'compact', False)
    _compact_header_needed = _compact  # emit header before first row
    _is_single = (not getattr(args, 'all_gpus', False)
                  and not getattr(args, 'gpu_list', None))
    
    def _maybe_emit_compact(perf):
        """If --compact, emit a CSV row for the just-finished benchmark."""
        nonlocal _compact_header_needed
        if not _compact or perf is None:
            return
        import socket
        hostname = tel_data.get('hostname') or socket.gethostname().split('.', 1)[0]
        tel_s = perf.get('telemetry', {})
        row = {
            'hostname':       hostname,
            'gpu':            gpu_index,
            'gpu_model':      tel_data.get('model', ''),
            'serial':         tel_data.get('serial', ''),
            'benchmark':      perf['name'],
            'dtype':          perf.get('params', {}).get('dtype', ''),
            'iterations':     perf.get('iterations', ''),
            'runtime_s':      perf.get('runtime_s', ''),
            'min':            f"{perf['min']:.4f}",
            'mean':           f"{perf['mean']:.4f}",
            'max':            f"{perf['max']:.4f}",
            'unit':           perf['unit'],
            'power_avg_w':    f"{tel_s.get('power_W_mean', 0):.1f}" if tel_s.get('power_W_mean') else '',
            'temp_max_c':     f"{tel_s.get('temp_gpu_C_max', 0):.0f}" if tel_s.get('temp_gpu_C_max') else '',
        }
        if args.verbose:
            row.update({
                'sm_util_mean':     f"{tel_s['sm_util_mean']:.0f}" if 'sm_util_mean' in tel_s else '',
                'mem_bw_util_mean': f"{tel_s['mem_bw_util_mean']:.0f}" if 'mem_bw_util_mean' in tel_s else '',
                'gpu_clock_mean':   f"{tel_s['gpu_clock_mean']:.0f}" if 'gpu_clock_mean' in tel_s else '',
                'mem_used_gb_mean': f"{tel_s['mem_used_MB_mean'] / 1024:.2f}" if 'mem_used_MB_mean' in tel_s else '',
                'throttled':        'true' if perf.get('throttled') else 'false',
            })
        # In single-GPU mode each process handles its own header;
        # in multi-GPU mode the parent emits the header before spawning.
        _emit_compact_csv(row, verbose=args.verbose,
                          header=(_compact_header_needed and _is_single))
        _compact_header_needed = False
    
    # Repeat loop for running benchmark suite multiple times
    for repeat_num in range(1, args.repeats + 1):
        # Update repeat number in verbose printer
        prn.repeat_num = repeat_num
        
        if args.repeats > 1:
            log.info(f"\n{'='*60}")
            log.info(f"REPEAT {repeat_num}/{args.repeats}")
            log.info(f"{'='*60}\n")
        
        # Optional delay between repeats (for thermal stabilization)
        if repeat_num > 1 and args.repeat_delay > 0:
            log.info(f"Waiting {args.repeat_delay}s before next repeat...")
            time.sleep(args.repeat_delay)
        
        # Check if running from config file with benchmark list
        if hasattr(args, 'benchmark_list') and args.benchmark_list:
            # Shuffle benchmark list if requested
            benchmark_list = args.benchmark_list[:]
            if args.shuffle:
                random.shuffle(benchmark_list)
                log.info(f"Shuffled benchmark execution order")
            
            log.info(f"Running {len(benchmark_list)} benchmark(s) from config file:")
            for idx, bench_config in enumerate(benchmark_list):
                bench_name = bench_config.get('name', 'UNNAMED')
                precision = bench_config.get('precision', 'default')
                log.info(f"  {idx+1}. {bench_name} ({precision})")
            
            # Run benchmarks from config file list (supports multiple instances)
            for idx, bench_config in enumerate(benchmark_list):
                bench_name = bench_config.get('name', '')
                
                # Save original values to restore later
                original_values = {}
                
                # Apply benchmark-specific parameters
                if bench_name == 'batched_gemm':
                    original_values['precision_gemm'] = args.precision_gemm
                    original_values['batch_count_gemm'] = args.batch_count_gemm
                    original_values['m'] = args.m
                    original_values['n'] = args.n
                    original_values['k'] = args.k
                    original_values['batched_gemm_TF32_mode'] = args.batched_gemm_TF32_mode
                    original_values['inner_loop_batched_gemm'] = args.inner_loop_batched_gemm
                    
                    args.precision_gemm = bench_config.get('precision', 'float32')
                    args.batch_count_gemm = bench_config.get('batch_count', args.batch_count_gemm)
                    args.m = bench_config.get('m', args.m)
                    args.n = bench_config.get('n', args.n)
                    args.k = bench_config.get('k', args.k)
                    args.batched_gemm_TF32_mode = bench_config.get('tf32_mode', False)
                    args.inner_loop_batched_gemm = bench_config.get('inner_loop', args.inner_loop_batched_gemm)
                    
                    # DEBUG: Log parameters being used
                    log.info(f"[GPU{gpu_index} GEMM CONFIG] B={args.batch_count_gemm}, M={args.m}, N={args.n}, K={args.k}, dtype={args.precision_gemm}, iters={args.inner_loop_batched_gemm}")
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = batched_gemm_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'convolution':
                    original_values['precision_convolution'] = args.precision_convolution
                    original_values['batch_count_convolution'] = args.batch_count_convolution
                    original_values['in_channels'] = args.in_channels
                    original_values['out_channels'] = args.out_channels
                    original_values['height'] = args.height
                    original_values['width'] = args.width
                    original_values['kernel_size'] = args.kernel_size
                    original_values['inner_loop_convolution'] = args.inner_loop_convolution
                    
                    args.precision_convolution = bench_config.get('precision', 'float32')
                    args.batch_count_convolution = bench_config.get('batch_count', args.batch_count_convolution)
                    args.in_channels = bench_config.get('in_channels', args.in_channels)
                    args.out_channels = bench_config.get('out_channels', args.out_channels)
                    args.height = bench_config.get('height', args.height)
                    args.width = bench_config.get('width', args.width)
                    args.kernel_size = bench_config.get('kernel_size', args.kernel_size)
                    args.inner_loop_convolution = bench_config.get('inner_loop', args.inner_loop_convolution)
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = convolution_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'fft':
                    original_values['precision_fft'] = args.precision_fft
                    original_values['batch_count_fft'] = args.batch_count_fft
                    original_values['nx'] = args.nx
                    original_values['ny'] = args.ny
                    original_values['nz'] = args.nz
                    original_values['inner_loop_fft'] = args.inner_loop_fft
                    
                    args.precision_fft = bench_config.get('precision', 'float32')
                    args.batch_count_fft = bench_config.get('batch_count', args.batch_count_fft)
                    args.nx = bench_config.get('nx', args.nx)
                    args.ny = bench_config.get('ny', args.ny)
                    args.nz = bench_config.get('nz', args.nz)
                    args.inner_loop_fft = bench_config.get('inner_loop', args.inner_loop_fft)
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = fft_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'einsum':
                    original_values['precision_einsum'] = args.precision_einsum
                    original_values['batch_count_einsum'] = args.batch_count_einsum
                    original_values['heads'] = args.heads
                    original_values['seq_len'] = args.seq_len
                    original_values['d_model'] = args.d_model
                    original_values['inner_loop_einsum'] = args.inner_loop_einsum
                    
                    args.precision_einsum = bench_config.get('precision', 'float32')
                    args.batch_count_einsum = bench_config.get('batch_count', args.batch_count_einsum)
                    args.heads = bench_config.get('heads', args.heads)
                    args.seq_len = bench_config.get('seq_len', args.seq_len)
                    args.d_model = bench_config.get('head_dim', args.d_model)
                    args.inner_loop_einsum = bench_config.get('inner_loop', args.inner_loop_einsum)
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = einsum_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'memory_traffic':
                    original_values['precision_memory'] = args.precision_memory
                    original_values['memory_size'] = args.memory_size
                    original_values['memory_iterations'] = args.memory_iterations
                    original_values['inner_loop_memory_traffic'] = args.inner_loop_memory_traffic
                    
                    args.precision_memory = bench_config.get('precision', 'float32')
                    args.memory_size = bench_config.get('size', args.memory_size)
                    args.memory_iterations = bench_config.get('iterations', args.memory_iterations)
                    args.inner_loop_memory_traffic = bench_config.get('inner_loop', args.inner_loop_memory_traffic)
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = memory_traffic_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'heat_equation':
                    original_values['precision_heat'] = getattr(args, 'precision_heat', 'float64')
                    original_values['heat_grid_size'] = getattr(args, 'heat_grid_size', 128)
                    original_values['heat_time_steps'] = getattr(args, 'heat_time_steps', 100)
                    original_values['alpha'] = getattr(args, 'alpha', 0.01)
                    original_values['delta_t'] = getattr(args, 'delta_t', 0.01)
                    original_values['inner_loop_heat_equation'] = getattr(args, 'inner_loop_heat_equation', 10)
                    
                    args.precision_heat = bench_config.get('precision', 'float64')
                    args.heat_grid_size = bench_config.get('grid_size', getattr(args, 'heat_grid_size', 128))
                    args.heat_time_steps = bench_config.get('time_steps', getattr(args, 'heat_time_steps', 100))
                    args.alpha = bench_config.get('alpha', getattr(args, 'alpha', 0.01))
                    args.delta_t = bench_config.get('delta_t', getattr(args, 'delta_t', 0.01))
                    args.inner_loop_heat_equation = bench_config.get('inner_loop', getattr(args, 'inner_loop_heat_equation', 10))
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = laplacian_heat_equation(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'schrodinger':
                    original_values['precision_schrodinger'] = getattr(args, 'precision_schrodinger', 'complex128')
                    original_values['schrodinger_grid_size'] = getattr(args, 'schrodinger_grid_size', 128)
                    original_values['schrodinger_time_steps'] = getattr(args, 'schrodinger_time_steps', 100)
                    original_values['inner_loop_schrodinger'] = getattr(args, 'inner_loop_schrodinger', 10)
                    
                    args.precision_schrodinger = bench_config.get('precision', 'complex128')
                    args.schrodinger_grid_size = bench_config.get('grid_size', getattr(args, 'schrodinger_grid_size', 128))
                    args.schrodinger_time_steps = bench_config.get('time_steps', getattr(args, 'schrodinger_time_steps', 100))
                    args.inner_loop_schrodinger = bench_config.get('inner_loop', getattr(args, 'inner_loop_schrodinger', 10))
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = schrodinger_equation(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'atomic_contention':
                    original_values['precision_atomic'] = getattr(args, 'precision_atomic', 'float32')
                    original_values['atomic_target_size'] = getattr(args, 'atomic_target_size', 1_000_000)
                    original_values['atomic_num_updates'] = getattr(args, 'atomic_num_updates', 10_000_000)
                    original_values['atomic_contention_range'] = getattr(args, 'atomic_contention_range', 1024)
                    original_values['inner_loop_atomic'] = getattr(args, 'inner_loop_atomic', 50)
                    
                    args.precision_atomic = bench_config.get('precision', 'float32')
                    args.atomic_target_size = bench_config.get('target_size', getattr(args, 'atomic_target_size', 1_000_000))
                    args.atomic_num_updates = bench_config.get('num_updates', getattr(args, 'atomic_num_updates', 10_000_000))
                    args.atomic_contention_range = bench_config.get('contention_range', getattr(args, 'atomic_contention_range', 1024))
                    args.inner_loop_atomic = bench_config.get('inner_loop', getattr(args, 'inner_loop_atomic', 50))
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = atomic_contention_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                elif bench_name == 'sparse_mm':
                    original_values['precision_sparse'] = getattr(args, 'precision_sparse', 'float32')
                    original_values['sparse_m'] = getattr(args, 'sparse_m', 8192)
                    original_values['sparse_n'] = getattr(args, 'sparse_n', 8192)
                    original_values['sparse_k'] = getattr(args, 'sparse_k', 8192)
                    original_values['sparse_density'] = getattr(args, 'sparse_density', 0.01)
                    original_values['inner_loop_sparse'] = getattr(args, 'inner_loop_sparse', 50)
                    
                    args.precision_sparse = bench_config.get('precision', 'float32')
                    args.sparse_m = bench_config.get('m', getattr(args, 'sparse_m', 8192))
                    args.sparse_n = bench_config.get('n', getattr(args, 'sparse_n', 8192))
                    args.sparse_k = bench_config.get('k', getattr(args, 'sparse_k', 8192))
                    args.sparse_density = bench_config.get('density', getattr(args, 'sparse_density', 0.01))
                    args.inner_loop_sparse = bench_config.get('inner_loop', getattr(args, 'inner_loop_sparse', 50))
                    
                    # Reset telemetry stats before benchmark for per-benchmark isolation
                    tel.reset_stats()
                    perf = sparse_mm_test(args, dev, log, tel, tel_thread, prn)
                    
                    # Free GPU memory
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    if perf:
                        benchmark_results.append(perf)
                
                else:
                    # Unknown benchmark name in config file
                    log.warning(f"Unknown benchmark name '{bench_name}' in config file (benchmark #{idx+1}). Valid names: batched_gemm, convolution, fft, einsum, memory_traffic, heat_equation, schrodinger, atomic_contention, sparse_mm")
                    continue
                
                # Emit compact CSV row (if --compact) before restoring params
                _maybe_emit_compact(perf)
                
                # Restore original values after each benchmark
                for key, value in original_values.items():
                    setattr(args, key, value)
    
        else:
            # Run benchmarks based on CLI flags (original behavior)
            if args.batched_gemm:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = batched_gemm_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.convolution:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = convolution_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.fft:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = fft_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.einsum:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = einsum_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.memory_traffic:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = memory_traffic_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.heat_equation:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = laplacian_heat_equation(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.schrodinger:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = schrodinger_equation(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.atomic_contention:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = atomic_contention_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
            if args.sparse_mm:
                tel.reset_stats()  # Per-benchmark telemetry isolation
                perf = sparse_mm_test(args, dev, log, tel, tel_thread, prn)
                _maybe_emit_compact(perf)
                benchmark_results.append(perf) if perf else None
    
    # End of repeat loop

    # Stop telemetry thread and get final reading
    tel_thread.stop()
    tel_data = tel_thread.get_latest()
    tel_stats = tel.get_stats(skip_first_n=args.skip_telemetry_first_n)
    
    # Only log these in non-verbose mode (verbose mode shows CSV only)
    if not args.verbose:
        log.info(f"Final telemetry: {_format_telemetry_compact(tel_data)}")
        tel.shutdown()
        log.info("="*80)
        log.info(f"[OK] Benchmark run finished on {dev_label}")
        log.info("="*80)
    else:
        # In verbose mode, show throttle summary only for THERMAL throttling
        # Power cap throttling is expected/normal when power limits are enforced
        thermal_throttled = False
        if hasattr(tel, 'hw_slowdown_count') and (tel.hw_slowdown_count > 0 or tel.sw_slowdown_count > 0):
            thermal_throttled = True  # NVIDIA thermal
        if hasattr(tel, 'thermal_throttle_count') and tel.thermal_throttle_count > 0:
            thermal_throttled = True  # AMD thermal
        
        if thermal_throttled:
            log.info("="*80)
            if hasattr(tel, 'hw_slowdown_count'):  # NVIDIA
                log.info(f"[WARN] THERMAL THROTTLING DETECTED on {dev_label}:")
                if tel.hw_slowdown_count > 0:
                    log.info(f"   Hardware Thermal Slowdown: {tel.hw_slowdown_count} samples")
                if tel.sw_slowdown_count > 0:
                    log.info(f"   Software Thermal Slowdown: {tel.sw_slowdown_count} samples")
            elif hasattr(tel, 'thermal_throttle_count'):  # AMD
                log.info(f"[WARN] THERMAL THROTTLING DETECTED on {dev_label}:")
                log.info(f"   Thermal Throttle: {tel.thermal_throttle_count} samples")
            log.info("="*80)
        tel.shutdown()
    
    # Return results for multi-GPU summary
    return {
        'gpu_index': gpu_index,
        'device': str(dev),
        'model': tel_data.get('model', 'Unknown'),
        'serial': tel_data.get('serial', 'N/A'),
        'hostname': tel_data.get('hostname', 'N/A'),
        'final_telemetry': tel_data,
        'telemetry_stats': tel_stats,
        'benchmarks': benchmark_results,
    }


# Worker function for multiprocessing - must be at module level for pickling
def _run_gpu_worker(args, gpu_idx, result_file):
    """Worker function that runs benchmark and writes result to temp file."""
    import pickle
    import sys
    
    # DON'T set up logging here - let run_single_gpu() call init_logging() 
    # so it respects --verbose-file-only and other flags properly
    
    try:
        # Pass log=None so run_single_gpu will initialize logging correctly
        result = run_single_gpu(args, gpu_idx, log=None)
        
        if result:
            # Check if any benchmarks actually completed successfully
            benchmarks = result.get('benchmarks', [])
            successful_benchmarks = [b for b in benchmarks if b is not None]
            
            if successful_benchmarks:
                # Write result to pickle file for parent process
                with open(result_file, 'wb') as f:
                    pickle.dump(result, f)
            else:
                # All benchmarks failed (returned None)
                sys.stderr.write(f"GPU {gpu_idx}: All benchmarks failed\n")
                sys.exit(1)
        else:
            sys.stderr.write(f"GPU {gpu_idx}: run_single_gpu returned no result\n")
            sys.exit(1)
    except Exception as e:
        import traceback
        sys.stderr.write(f"GPU {gpu_idx}: Failed in worker: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)  # Exit with error code


def main():
    args = build_parser().parse_args()
    
    # Banner handling: --forge (easter egg) > --banner > nothing
    if args.forge:
        print_forge_banner()
    elif args.banner:
        print_banner()
    
    # Load and apply configuration file if specified
    config = None
    if args.config:
        config = load_config(args.config)
        args = apply_config_to_args(args, config)
    
    # Load external hardware baselines if provided
    if args.baseline_file:
        custom_baselines = load_hardware_baselines(args.baseline_file)
        if custom_baselines:
            args._hardware_baselines = {**HARDWARE_BASELINES, **custom_baselines}
            print(f"Loaded hardware baselines from: {args.baseline_file}")
        else:
            args._hardware_baselines = HARDWARE_BASELINES
    elif args.no_validation:
        args._hardware_baselines = {}  # Empty dict = skip validation
    else:
        args._hardware_baselines = HARDWARE_BASELINES  # Use built-in defaults
        
    # Handle --list-profiles
    if args.list_profiles:
        if not args.config:
            print("No configuration file specified. Use --config <path>")
            sys.exit(1)
        print(f"Configuration: {args.config}")
        print(f"Profile: {config.get('profile', 'Unnamed')}")
        print(f"Platform: {config.get('platform', 'Not specified')}")
        print(f"\\nEnabled Benchmarks:")
        for bench in config.get('benchmarks', []):
            if bench.get('enabled', True):
                print(f"  - {bench.get('name')} ({bench.get('precision', 'default')})")
        sys.exit(0)
    
    log = init_logging(args)
    
    # Determine which GPUs to run on
    gpu_indices = []
    if args.all_gpus and torch.cuda.is_available():
        gpu_indices = list(range(torch.cuda.device_count()))
        log.info(f"Running on all {len(gpu_indices)} GPUs")
    elif args.gpu_list:
        gpu_indices = [int(x.strip()) for x in args.gpu_list.split(',')]
        log.info(f"Running on GPUs: {gpu_indices}")
    else:
        gpu_indices = [args.device_index]
        # Device-aware logging
        if torch.cuda.is_available():
            log.info(f"Running on single GPU: {args.device_index}")
        elif torch.backends.mps.is_available():
            log.info("Running on MPS (Apple Metal)")
        else:
            log.info("Running on CPU")
    
    # Compute intelligent CPU distribution for multi-GPU to avoid overloading
    if args.cpu_affinity and len(gpu_indices) > 1:
        cpu_distribution = distribute_cpus_for_gpus(gpu_indices)
        args._cpu_distribution = cpu_distribution
        # Compact summary of CPU distribution
        dist_summary = ", ".join(f"GPU{g}→{format_cpu_list(c)}" for g, c in sorted(cpu_distribution.items()))
        log.info(f"CPU distribution: {dist_summary}")
    
    # Run on each GPU
    results = []
    if len(gpu_indices) == 1:
        # Single GPU - direct execution
        result = run_single_gpu(args, gpu_indices[0], log)
        if result:
            results.append(result)
    else:
        # Multi-GPU - parallel execution via multiprocessing with 'spawn' method
        # CUDA requires 'spawn' start method, not 'fork'
        multiprocessing.set_start_method('spawn', force=True)
        
        # Pin parent process to minimize interference with GPU workers
        # Parent is mostly idle (spawning workers, collecting results)
        # Default: pin to last core of first NUMA (least likely to conflict with GPU worker allocation)
        if args.cpu_affinity and args.parent_cpu != -1:  # -1 explicitly disables pinning
            try:
                if args.parent_cpu is not None:
                    # User specified explicit core
                    parent_core = args.parent_cpu
                else:
                    # Default: use last core of first NUMA node (safest choice)
                    numa_nodes = sorted(set(get_gpu_numa_node(g) for g in gpu_indices))
                    if numa_nodes:
                        first_numa_cpus = get_numa_cpus(numa_nodes[0])
                        if first_numa_cpus:
                            parent_core = first_numa_cpus[-1]  # Last core
                        else:
                            parent_core = None
                    else:
                        parent_core = None
                
                if parent_core is not None:
                    set_cpu_affinity([parent_core])
                    log.info(f"Parent process pinned to CPU {parent_core}")
            except Exception as e:
                log.warning(f"Could not pin parent process: {e}")
        
        log.info(f"Launching {len(gpu_indices)} parallel processes...")
        
        # In compact mode, emit the CSV header once from the parent process
        # before workers start printing rows
        if getattr(args, 'compact', False):
            cols = _compact_csv_columns(args.verbose)
            print(",".join(cols), flush=True)
        
        # Use temp files for result collection to avoid IPC overhead on NUMA 0
        # Temp directory configurable via --temp-dir, TORCH_HAMMER_TEMP_DIR env var, or system default
        import tempfile
        import pickle
        
        temp_dir = getattr(args, 'temp_dir', None) or os.environ.get('TORCH_HAMMER_TEMP_DIR') or tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        log.info(f"Using temp directory: {temp_dir}")
        
        processes = []
        result_files = []
        
        for gpu_idx in gpu_indices:
            # Create temp file for this GPU's results
            result_file = os.path.join(temp_dir, f'torch_hammer_gpu{gpu_idx}_{os.getpid()}.pkl')
            result_files.append((gpu_idx, result_file))
            log.info(f"GPU {gpu_idx} will write results to: {result_file}")
            
            p = multiprocessing.Process(target=_run_gpu_worker, args=(args, gpu_idx, result_file))
            p.start()
            processes.append((gpu_idx, p))
            log.info(f"Started process for GPU {gpu_idx} (PID: {p.pid})")
        
        # Wait for all processes to complete
        for gpu_idx, p in processes:
            p.join()
            if p.exitcode == 0:
                log.info(f"GPU {gpu_idx} completed successfully")
            else:
                log.error(f"GPU {gpu_idx} failed with exit code {p.exitcode}")
        
        # Explicitly close processes to avoid zombies
        for gpu_idx, p in processes:
            p.close()
        
        # Collect results from temp files
        results = []
        for gpu_idx, result_file in result_files:
            try:
                if os.path.exists(result_file):
                    with open(result_file, 'rb') as f:
                        result = pickle.load(f)
                    results.append(result)
                    os.unlink(result_file)  # Clean up temp file
                else:
                    log.warning(f"Result file not found for GPU {gpu_idx}")
            except Exception as e:
                log.error(f"Failed to read results for GPU {gpu_idx}: {e}")
        
        # Sort by GPU index
        results = sorted(results, key=lambda x: x['gpu_index'])
    
    # Display unified multi-GPU summary (skip in compact mode — CSV rows already emitted)
    if len(results) > 1 and not getattr(args, 'compact', False):
        log.info("")
        log.info("="*80)
        log.info("MULTI-GPU SUMMARY")
        log.info("="*80)
        
        # Display benchmark performance comparison first
        all_benchmarks = []  # Use list to preserve order and handle duplicates
        for result in results:
            for bench in result.get('benchmarks', []):
                if bench:
                    # Create unique identifier based on name and key parameters
                    bench_name = bench['name']
                    params = bench.get('params', {})
                    
                    # Find if this exact benchmark config already exists
                    found = False
                    for entry in all_benchmarks:
                        if entry['name'] == bench_name and entry['params'] == params:
                            # Same benchmark, add this GPU's results
                            entry['gpus'][result['gpu_index']] = bench
                            found = True
                            break
                    
                    if not found:
                        # New benchmark configuration
                        all_benchmarks.append({
                            'name': bench_name,
                            'unit': bench['unit'],
                            'params': params,
                            'gpus': {result['gpu_index']: bench}
                        })
        
        if all_benchmarks:
            log.info("")
            log.info("BENCHMARK RESULTS - ALL TESTS")
            log.info("=" * 80)
            for bench_data in all_benchmarks:
                log.info("")
                log.info(f"{'─'*80}")
                log.info(f"Test: {bench_data['name']}")
                log.info(f"{'─'*80}")
                
                # Display test parameters (from first GPU, should be same for all)
                first_gpu = min(bench_data['gpus'].keys())
                params = bench_data['gpus'][first_gpu].get('params', {})
                if params:
                    param_parts = []
                    if 'dtype' in params:
                        param_parts.append(f"dtype={params['dtype']}")
                    if 'tf32' in params and params['tf32']:
                        param_parts.append("TF32=enabled")
                    if 'batch' in params:
                        param_parts.append(f"batch={params['batch']}")
                    if 'm' in params and 'n' in params and 'k' in params:
                        param_parts.append(f"M×N×K={params['m']}×{params['n']}×{params['k']}")
                    if 'in_channels' in params:
                        param_parts.append(f"in_ch={params['in_channels']}")
                    if 'out_channels' in params:
                        param_parts.append(f"out_ch={params['out_channels']}")
                    if 'height' in params and 'width' in params:
                        param_parts.append(f"H×W={params['height']}×{params['width']}")
                    if 'kernel' in params:
                        param_parts.append(f"kernel={params['kernel']}")
                    if 'nx' in params and 'ny' in params and 'nz' in params:
                        param_parts.append(f"grid={params['nx']}×{params['ny']}×{params['nz']}")
                    if 'heads' in params:
                        param_parts.append(f"heads={params['heads']}")
                    if 'seq_len' in params:
                        param_parts.append(f"seq_len={params['seq_len']}")
                    if 'head_dim' in params:
                        param_parts.append(f"head_dim={params['head_dim']}")
                    if 'size' in params:
                        param_parts.append(f"size={params['size']}")
                    if 'iterations' in params:
                        param_parts.append(f"iterations={params['iterations']}")
                    
                    if param_parts:
                        log.info(f"Parameters: {', '.join(param_parts)}")
                
                # Display performance for each GPU with outlier detection
                log.info(f"Performance ({bench_data['unit']}): min / mean / max")
                
                # Calculate statistics for outlier detection
                gpu_means = [bench_data['gpus'][idx]['mean'] for idx in bench_data['gpus'].keys()]
                if len(gpu_means) > 1:
                    overall_mean = statistics.mean(gpu_means)
                    overall_stdev = statistics.stdev(gpu_means) if len(gpu_means) > 1 else 0
                    coef_var = (overall_stdev / overall_mean * 100) if overall_mean > 0 else 0
                    
                    fastest_gpu = max(bench_data['gpus'].keys(), key=lambda idx: bench_data['gpus'][idx]['mean'])
                    slowest_gpu = min(bench_data['gpus'].keys(), key=lambda idx: bench_data['gpus'][idx]['mean'])
                    
                    # Get outlier threshold from args (default 15%)
                    outlier_threshold = args.outlier_threshold_pct
                    
                    for gpu_idx in sorted(bench_data['gpus'].keys()):
                        bench = bench_data['gpus'][gpu_idx]
                        deviation_pct = abs(bench['mean'] - overall_mean) / overall_mean * 100 if overall_mean > 0 else 0
                        
                        # Build status indicators
                        status = []
                        if deviation_pct > outlier_threshold:
                            status.append("[OUTLIER]")
                        if gpu_idx == fastest_gpu:
                            status.append("FASTEST")
                        if gpu_idx == slowest_gpu and len(gpu_means) > 2:  # Only show if 3+ GPUs
                            status.append("SLOWEST")
                        
                        status_str = f" [{', '.join(status)}]" if status else ""
                        log.info(f"  GPU {gpu_idx}: {bench['min']:>10.2f} / {bench['mean']:>10.2f} / {bench['max']:>10.2f}{status_str}")
                        
                        # Show deviation percentage for outliers
                        if deviation_pct > outlier_threshold:
                            log.info(f"          ({deviation_pct:+.1f}% deviation from mean)")
                    
                    # Show overall variability
                    if coef_var > 5:
                        log.info(f"  Coefficient of Variation: {coef_var:.1f}% (spread across GPUs)")
                else:
                    # Single GPU, no outlier detection
                    for gpu_idx in sorted(bench_data['gpus'].keys()):
                        bench = bench_data['gpus'][gpu_idx]
                        log.info(f"  GPU {gpu_idx}: {bench['min']:>10.2f} / {bench['mean']:>10.2f} / {bench['max']:>10.2f}")
                
                # Display telemetry statistics for each GPU
                log.info("")
                log.info("Telemetry Statistics:")
                for gpu_idx in sorted(bench_data['gpus'].keys()):
                    bench = bench_data['gpus'][gpu_idx]
                    tel_stats = bench.get('telemetry', {})
                    
                    if not tel_stats:
                        log.info(f"  GPU {gpu_idx}: No telemetry data")
                        continue
                    
                    log.info(f"  GPU {gpu_idx}:")
                    
                    # SM Utilization
                    if 'sm_util_mean' in tel_stats:
                        log.info(f"    SM Util:     {tel_stats.get('sm_util_min', 'N/A')}% / {tel_stats['sm_util_mean']:.0f}% / {tel_stats.get('sm_util_max', 'N/A')}%")
                    
                    # Memory BW Utilization
                    if 'mem_bw_util_mean' in tel_stats:
                        log.info(f"    Mem BW Util: {tel_stats.get('mem_bw_util_min', 'N/A')}% / {tel_stats['mem_bw_util_mean']:.0f}% / {tel_stats.get('mem_bw_util_max', 'N/A')}%")
                    
                    # Temperature
                    if 'temp_gpu_C_mean' in tel_stats:
                        log.info(f"    Temperature: {tel_stats.get('temp_gpu_C_min', 'N/A')}°C / {tel_stats['temp_gpu_C_mean']:.1f}°C / {tel_stats.get('temp_gpu_C_max', 'N/A')}°C")
                    
                    # Power
                    if 'power_W_mean' in tel_stats:
                        power_min = tel_stats.get('power_W_min', 0)
                        power_max = tel_stats.get('power_W_max', 0)
                        log.info(f"    Power:       {power_min:.0f}W / {tel_stats['power_W_mean']:.0f}W / {power_max:.0f}W")
                    
                    # Memory Usage
                    if 'mem_used_MB_mean' in tel_stats:
                        min_gb = tel_stats.get('mem_used_MB_min', 0) / 1024
                        mean_gb = tel_stats['mem_used_MB_mean'] / 1024
                        max_gb = tel_stats.get('mem_used_MB_max', 0) / 1024
                        log.info(f"    Memory Used: {min_gb:.1f}GB / {mean_gb:.1f}GB / {max_gb:.1f}GB")
                    
                    # VBST Sync (NVIDIA)
                    if 'vbst_sync_mean' in tel_stats:
                        log.info(f"    VBST Sync:   {tel_stats.get('vbst_sync_min', 'N/A')} / {tel_stats['vbst_sync_mean']:.1f} / {tel_stats.get('vbst_sync_max', 'N/A')}")
        
        # Display tabular benchmark summary
        log.info("")
        log.info("="*80)
        log.info("BENCHMARK SUMMARY")
        log.info("="*80)
        
        if results:
            # Group benchmarks by test name + precision
            benchmark_groups = {}
            for result in results:
                gpu_idx = result['gpu_index']
                serial = result.get('serial', 'N/A')
                tel_stats = result.get('telemetry_stats', {})
                
                for bench in result.get('benchmarks', []):
                    if not bench:
                        continue
                    # Create unique key for each test variant
                    dtype = bench.get('params', {}).get('dtype', 'unknown')
                    test_key = f"{bench['name']}_{dtype}"
                    
                    if test_key not in benchmark_groups:
                        benchmark_groups[test_key] = {
                            'name': bench['name'],
                            'dtype': dtype,
                            'unit': bench['unit'],
                            'results': []
                        }
                    
                    # Determine status
                    throttled = tel_stats.get('throttled', False) or bench.get('telemetry', {}).get('throttled', False)
                    efficiency = bench.get('efficiency_pct', None)
                    
                    if throttled:
                        status = "FAIL"
                        notes = "Throttled"
                    elif efficiency is not None and efficiency < args.efficiency_warn_pct:
                        status = "FAIL"
                        notes = f"Low efficiency ({efficiency:.0f}%)"
                    else:
                        status = "PASS"
                        notes = ""
                    
                    # Use per-benchmark telemetry if available, fallback to overall stats
                    bench_tel = bench.get('telemetry', {})
                    benchmark_groups[test_key]['results'].append({
                        'gpu': gpu_idx,
                        'serial': serial,
                        'performance': bench['mean'],
                        'power_avg': bench_tel.get('power_W_mean', tel_stats.get('power_W_mean', 0)),
                        'temp_max': bench_tel.get('temp_gpu_C_max', tel_stats.get('temp_gpu_C_max', 0)),
                        'status': status,
                        'notes': notes
                    })
            
            # Display each benchmark group as a table
            csv_data = []  # For optional CSV export
            
            for test_key in sorted(benchmark_groups.keys()):
                group = benchmark_groups[test_key]
                results_list = group['results']
                
                # Sort by GPU index
                results_list.sort(key=lambda x: x['gpu'])
                
                # Calculate fastest/slowest
                if len(results_list) > 1:
                    fastest_perf = max(r['performance'] for r in results_list)
                    for r in results_list:
                        if r['performance'] == fastest_perf and not r['notes']:
                            r['notes'] = "Fastest"
                
                log.info("")
                log.info(f"Test: {group['name']} ({group['dtype']})")
                log.info(f"{'GPU':<4} | {'Serial':<12} | {'Performance':<13} | {'Power(avg)':<10} | {'Temp(max)':<9} | {'Status':<6} | Notes")
                log.info(f"{'-'*4}+{'-'*14}+{'-'*15}+{'-'*12}+{'-'*11}+{'-'*8}+{'-'*20}")
                
                for r in results_list:
                    perf_str = f"{r['performance']:.1f} {group['unit']}"
                    power_str = f"{r['power_avg']:.0f}W" if r['power_avg'] > 0 else "N/A"
                    temp_str = f"{r['temp_max']:.0f}C" if r['temp_max'] > 0 else "N/A"
                    
                    log.info(f"{r['gpu']:<4} | {r['serial']:<12} | {perf_str:<13} | {power_str:<10} | {temp_str:<9} | {r['status']:<6} | {r['notes']}")
                    
                    # Collect for CSV export
                    csv_data.append({
                        'test': group['name'],
                        'dtype': group['dtype'],
                        'gpu': r['gpu'],
                        'serial': r['serial'],
                        'performance': r['performance'],
                        'unit': group['unit'],
                        'power_avg_W': r['power_avg'],
                        'temp_max_C': r['temp_max'],
                        'status': r['status'],
                        'notes': r['notes']
                    })
                
                # Aggregate stats if multi-GPU
                if len(results_list) > 1:
                    values = [r['performance'] for r in results_list]
                    aggregate = sum(values)
                    cv = (statistics.stdev(values) / statistics.mean(values) * 100.0) if len(values) > 1 else 0.0
                    log.info(f"")
                    log.info(f"Aggregate: {aggregate:.1f} {group['unit']} across {len(results_list)} GPUs | Variation: CV={cv:.1f}%")
            
            # Optional CSV export
            if hasattr(args, 'summary_csv') and args.summary_csv and csv_data:
                try:
                    import csv
                    import socket
                    from datetime import datetime
                    from pathlib import Path
                    
                    # Make filename unique per node to avoid conflicts at scale
                    hostname = socket.gethostname().split('.', 1)[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    csv_path = args.summary_csv
                    # Insert hostname_timestamp before extension
                    path_obj = Path(csv_path)
                    unique_filename = f"{path_obj.stem}_{hostname}_{timestamp}{path_obj.suffix}"
                    csv_path = path_obj.parent / unique_filename
                    
                    with open(csv_path, 'w', newline='') as csvfile:
                        fieldnames = ['test', 'dtype', 'gpu', 'serial', 'performance', 'unit', 'power_avg_W', 'temp_max_C', 'status', 'notes']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)
                    log.info(f"")
                    log.info(f"Summary exported to CSV: {csv_path}")
                except Exception as e:
                    log.warning(f"Failed to export CSV summary: {e}")
        
        log.info("")
        log.info("="*80)
    
    # Export JSON if requested
    if args.json_output and results:
        export_json_results(results, args, log)


def export_json_results(results, args, log):
    """Export benchmark results to JSON file."""
    import json
    import socket
    from datetime import datetime
    from pathlib import Path
    
    # Make filename unique per node to avoid conflicts at scale
    hostname = socket.gethostname().split('.', 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = args.json_output
    # Insert hostname_timestamp before extension
    path_obj = Path(output_path)
    unique_filename = f"{path_obj.stem}_{hostname}_{timestamp}{path_obj.suffix}"
    output_path = path_obj.parent / unique_filename
    
    try:
        # Build comprehensive JSON structure
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hostname": hostname,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "config_file": args.config if args.config else None,
            },
            "runtime_args": {
                "duration": args.duration,
                "min_iterations": args.min_iterations,
                "max_iterations": args.max_iterations,
                "warmup": args.warmup,
                "verbose": args.verbose,
            },
            "gpus": []
        }
        
        # Add CUDA version if available
        if torch.cuda.is_available():
            export_data["metadata"]["cuda_version"] = torch.version.cuda
        
        # Add per-GPU results
        for result in results:
            gpu_data = {
                "gpu_index": result['gpu_index'],
                "model": result['model'],
                "serial": result['serial'],
                "benchmarks": [],
                "final_telemetry": result['final_telemetry'],
                "telemetry_stats": result['telemetry_stats']
            }
            
            # Add benchmark results
            for bench in result.get('benchmarks', []):
                if bench:
                    bench_data = {
                        "name": bench['name'],
                        "unit": bench['unit'],
                        "min": bench['min'],
                        "mean": bench['mean'],
                        "max": bench['max'],
                        "params": bench.get('params', {}),
                        "telemetry": bench.get('telemetry', {}),
                        "iteration_telemetry": bench.get('iteration_telemetry', []),
                        "validation": bench.get('validation', {}),
                    }
                    gpu_data["benchmarks"].append(bench_data)
            
            export_data["gpus"].append(gpu_data)
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        log.info(f"JSON results exported to: {output_path}")
        
    except Exception as e:
        log.error(f"Failed to export JSON: {e}")


if __name__ == "__main__":
    main()
