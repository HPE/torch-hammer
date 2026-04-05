# Torch Hammer  

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Tests](https://img.shields.io/badge/tests-211%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-53%25-yellow.svg)](tests/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

```
            _____              _       _   _                                      
           |_   _|__  _ __ ___| |__   | | | | __ _ _ __ ___  _ __ ___   ___ _ __ 
             | |/ _ \| '__/ __| '_ \  | |_| |/ _` | '_ ` _ \| '_ ` _ \ / _ \ '__|
             | | (_) | | | (__| | | | |  _  | (_| | | | | | | | | | | |  __/ |   
             |_|\___/|_|  \___|_| |_| |_| |_|\__,_|_| |_| |_|_| |_| |_|\___|_|   
                                                                        
                              Forged with PyTorch
                        GPU/CPU/APU Micro-Benchmark Suite
```

_A portable, PyTorch micro-benchmark suite for stress testing CPUs, GPUs, and APUs_

> **© Copyright 2025-2026 Hewlett Packard Enterprise Development LP**  
> Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE.md) for details.

#### Background
Torch Hammer is a benchmarking utility designed to stress test and evaluate the performance of CPUs, GPUs, and APUs using PyTorch. The inspiration for Torch Hammer arose from the increasing heterogeneity of workloads, hardware, and runtime environments across the HPC & AI industry. The rich HPC & AI ecosystem inspired Torch Hammer, a tool which aims to:
- Offer a variety of highly-parametrized tests that can push hardware power/thermal limits
- Characterize hardware performance and identify slow components
- Be portable so that it can be run across different platforms
- Provide an easy to maintain mechanism to add new tests

When building Torch Hammer, I was inspired by my undergraduate work in quantum chemistry, differential equations, and my more recent interest in AI. Some of the tests you will find in Torch Hammer are small kernels of the types of problems I would work out by hand for my assignments as a student at the University of Minnesota. Others are reflective of patterns commonly used in AI/ML workloads.

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Tested Platforms](#tested-platforms)  
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
5. [Command-line Reference](#command-line-reference)  
6. [Verbose-Mode](#Verbose)  
7. [Compact Mode](#compact-mode)  
8. [Syslog Mode](#syslog-mode)  
9. [Telemetry Back-ends](#telemetry-back-ends)  
10. [Examples](#examples)  
11. [Contributing](#contributing)  
12. [Project Governance](#project-governance)
13. [License](#license)  

---

## Key Features
- **Cross-platform**: CUDA, ROCm, Metal (MPS), CPU – automatically selected.
- **Nine micro benchmarks**
  1. Batched GEMM (matrix multiply)
  2. 2-D Convolution
  3. 3-D FFT
  4. Einsum (attention-style tensor contraction)
  5. Random memory traffic
  6. Laplacian heat-equation solver
  7. 1-D time-dependent Schrödinger equation
  8. Atomic contention (L2 cache stress via scatter_add)
  9. Sparse matrix multiplication (SpMM)
- **Precision choices**: `float16`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128`.
- **Tensor Core support**: Optional TF32 mode on newer platforms.
- **Live telemetry**
  - NVIDIA: power, temperature, utilization, clock, memory controller activity, memory utilization, GPU temperature, SM utilization
  - AMD ROCm: power, temperature, utilization, clock
  - CPU: basic vendor & model info.
- **Verbose logging**: every iteration, every telemetry field, one comma-separated line.
- **Compact CSV mode**: machine-readable CSV to stdout — one row per benchmark, pipe-friendly.
- **Syslog / dmesg mode**: structured `key=value` entries to syslog with auto-derived severity; optional `/dev/kmsg` output for kernel ring-buffer correlation.

---

## Tested Platforms

Torch Hammer has been validated on **50+ hardware configurations** spanning NVIDIA, AMD, Arm, and Apple platforms.

### HPC / Data Center Accelerators

| Vendor | Accelerator |
|--------|-------------|
| AMD | Instinct MI300A, Instinct MI250X |
| NVIDIA | B200, GH200, H100, A100 |

### NVIDIA GPUs

| Family | GPUs |
|--------|------|
| **GeForce (Turing)** | GTX 1660 SUPER, RTX 2060 SUPER, RTX 2080 Ti, RTX 2080 SUPER |
| **GeForce (Ampere)** | RTX 3060 Ti, RTX 3060 Mobile, RTX 3070, RTX 3070 Ti, RTX 3080, RTX 3080 Ti, RTX 3090 |
| **GeForce (Ada Lovelace)** | RTX 4060 Ti, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 4090, RTX 4090 D |
| **GeForce (Blackwell)** | RTX 5060 Ti, RTX 5070, RTX 5070 Ti, RTX 5080, RTX 5090 |
| **Professional** | Quadro RTX 6000, Titan RTX, RTX A4000, RTX A5000, RTX A6000, RTX 6000 Ada, RTX PRO 4000, RTX PRO 5000, RTX PRO 6000 Server, RTX PRO 6000 Workstation |
| **Data Center** | A10, A40, A100 PCIe, A100 SXM, A800, L4, L40S, H100 NVL, H100 SXM, H200, B200 |

### CPUs

| Vendor | Processor |
|--------|----------|
| AMD | EPYC Rome, EPYC Milan, EPYC Trento |
| Arm | Cortex-A76 |
| NVIDIA | Grace |

### Apple Silicon

| Chip |
|------|
| M3 Pro |

---
## Installation

### Prerequisites

* Python ≥ 3.8  
* PyTorch ≥ 1.10 (see [PyTorch installation guide](https://pytorch.org/get-started/locally/))

### Quick Start

```bash
# Clone the repository
git clone https://github.com/HPE/torch-hammer.git
cd torch-hammer

# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Make executable
chmod +x torch-hammer.py
```

> **Note:** After the steps above, complete the [Platform-Specific Setup](#platform-specific-setup) below for your hardware (NVIDIA, AMD, Apple Silicon, or CPU-only) before running benchmarks.

### Platform-Specific Setup

#### NVIDIA Setup
```bash
# Install PyTorch with CUDA support
# Replace 'cu126' with your CUDA version: cu118, cu121, cu124, cu126, etc.
# Check your CUDA version with: nvcc --version or nvidia-smi
# Examples:
#   CUDA 11.8 → cu118
#   CUDA 12.1 → cu121
#   CUDA 12.6 → cu126
#   CUDA 13.x → cu130 (when available)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install telemetry library
pip install nvidia-ml-py
```

#### AMD ROCm Setup
```bash
# Detect ROCm version and install matching PyTorch
ROCM_VER=$(cat /opt/rocm/.info/version | cut -d'-' -f1 | cut -d'.' -f1,2)
pip install torch --index-url https://download.pytorch.org/whl/rocm${ROCM_VER}

# Add ROCm's amdsmi to Python path (auto-detects Python version)
export PYTHONPATH=/opt/rocm/lib/$(ls /opt/rocm/lib | grep python):$PYTHONPATH
```

#### Apple Silicon (MPS) Setup
```bash
# Install PyTorch (MPS backend included automatically)
pip install torch
```

#### CPU-Only Setup
```bash
# Install PyTorch CPU-only build
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Optional Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| `pyyaml` | YAML config file support | `pip install pyyaml` |
| `setproctitle` | Custom process names in `ps`/`top` | `pip install setproctitle` |

### Telemetry Dependencies

| Platform | Telemetry Library | Installation |
|----------|------------------|--------------|
| **NVIDIA** | `pynvml` | `pip install nvidia-ml-py` |
| **AMD ROCm** | `amdsmi` | Bundled with ROCm (must match ROCm version) |
| **Apple/CPU** | Built-in | No additional install needed |

---

## Quick Start
Benchmark a single matrix-multiply workload on the first device:
```bash
./torch-hammer.py --batched-gemm
```

### Tuning for Device Memory

Default parameters target GPUs with 80GB+ VRAM. For GPUs with less memory, reduce problem sizes to avoid out-of-memory errors:

| GPU VRAM | Recommended GEMM Size | Example |
|----------|----------------------|---------|
| 80GB+ (H100, A100-80GB, MI300X) | 16384×16384 | `--m 16384 --n 16384 --k 16384` |
| 40-48GB (A100-40GB, A6000) | 8192×8192 | `--m 8192 --n 8192 --k 8192` |
| 16-24GB (RTX 4090, A5000) | 4096×4096 | `--m 4096 --n 4096 --k 4096` |
| 8-12GB (RTX 3080, T4) | 2048×2048 | `--m 2048 --n 2048 --k 2048` |

**Auto-scaling option:** Use `--stress-test` to automatically scale problem sizes based on available GPU memory:
```bash
./torch-hammer.py --batched-gemm --stress-test
```

**Other memory-sensitive parameters:**
- FFT: `--nx`/`--ny`/`--nz` (default 128³ uses ~16MB, try 64 for small GPUs)
- Sparse MM: `--sparse-m`, `--sparse-n`, `--sparse-k` (default 8192)
- Heat/Schrödinger: `--heat-grid-size`, `--schrodinger-grid-size`

---

## Command-line Reference
`./torch-hammer.py -h` prints the full help.  
The most important switches are summarised below (defaults in _italics_).

### Global
| Option | Description |
|--------|-------------|
| **Logging & Output** | |
| `--no-log` | Disable all logging. |
| `--log-file <path>` | Append all log lines to `path`. |
| `--log-dir <path>` | Directory for per-GPU log files (multi-GPU runs). |
| `--verbose` | One line per iteration (see examples). |
| `--verbose-file-only` | With `--log-file` or `--log-dir`, suppress stdout (file only). |
| `--compact` | Machine-readable CSV to stdout (one row per benchmark). See [Compact Mode](#compact-mode). |
| `--syslog` | Emit structured `key=value` entries to syslog after each benchmark. See [Syslog Mode](#syslog-mode). |
| `--syslog-dmesg` | Also write syslog messages to `/dev/kmsg` (requires `--syslog`). Needs root or `CAP_SYSLOG`. |
| `--banner` | Show ASCII banner at startup. |
| `--json-output <path>` | Write all results and telemetry to a JSON file. |
| `--summary-csv <path>` | Write benchmark summary table to a CSV file. |
| **Configuration** | |
| `--config <path>` | Path to YAML configuration file (see [YAML Configuration](#yaml-configuration)). |
| `--list-profiles` | List available configuration profiles and exit. |
| `--dry-run` | Show configuration and exit without running benchmarks. |
| **Device Selection** | |
| `--device-index <int>` | GPU index to use (default: _0_). Ignored if `--all-gpus` or `--gpu-list`. |
| `--all-gpus` | Run on all available GPUs in parallel. |
| `--gpu-list <indices>` | Comma-separated GPU indices (e.g., `0,2,3`). |
| **CPU Affinity & NUMA** | |
| `--cpu-affinity` | NUMA-aware CPU binding (default: _enabled_, Linux only). |
| `--no-cpu-affinity` | Disable NUMA-aware CPU binding. |
| `--cpu-gpu-map <mapping>` | Manual CPU-GPU binding (e.g., `0:0-15,1:16-31`). |
| `--cpu-list <cores>` | CPU cores for CPU-only mode (e.g., `0-23,48-71` or `all`). Default: all physical cores. |
| `--parent-cpu <int>` | Pin parent process to a specific CPU core (default: last core of first NUMA node; `-1` to disable). |
| **Iteration & Duration** | |
| `--warmup <int>` | Warm-up iterations before timing (default: _10_). |
| `--duration <float>` | Run each benchmark for specified seconds (overrides iteration counts). |
| `--min-iterations <int>` | Minimum iterations even if `--duration` is met (default: _10_). |
| `--max-iterations <int>` | Maximum iterations regardless of `--duration`. |
| `--repeats <int>` | Number of times to repeat entire benchmark suite (default: _1_). |
| `--repeat-delay <float>` | Delay in seconds between repeats for thermal stabilization (default: _0_). |
| **Stress & Scheduling** | |
| `--stress-test` | Automatically calculate maximum stress parameters based on available GPU memory. |
| `--shuffle` | Randomize benchmark execution order. |
| `--startup-delay-per-gpu <float>` | Staggered startup delay per GPU in seconds (GPU _N_ waits _N_ × delay). Helps avoid ROCm memory allocator contention (default: _0_). |
| **Telemetry Tuning** | |
| `--skip-telemetry-first-n <int>` | Skip first _N_ telemetry readings when calculating statistics (default: _10_). |
| `--telemetry-interval-ms <int>` | Telemetry polling interval in milliseconds (default: _100_). Higher values reduce resolution but may improve performance. |
| `--no-telemetry-thread` | Disable the background telemetry thread entirely (for debugging). |
| **Thermal / Performance Thresholds** | |
| `--temp-warn-C <float>` | Temperature warning threshold in °C (default: _90_). |
| `--temp-critical-C <float>` | Temperature critical threshold in °C (default: _95_). |
| `--power-warn-pct <float>` | Power-limit warning threshold in % (default: _98_). |
| `--outlier-threshold-pct <float>` | Multi-GPU outlier detection threshold in % (default: _15_). |
| `--efficiency-warn-pct <float>` | Hardware efficiency warning threshold in % (default: _70_). |
| `--baseline-file <path>` | Load hardware baselines from JSON/YAML file for performance validation. |
| `--no-validation` | Disable hardware performance validation. |
| **Miscellaneous** | |
| `--temp-dir <path>` | Directory for temp files (multi-GPU result collection). Falls back to `TORCH_HAMMER_TEMP_DIR` env var, then system temp. |

### Supported Precisions

Each benchmark has its own `--precision-<test>` flag. The available data types vary by benchmark:

| Benchmark | `float16` | `bfloat16` | `float32` | `float64` | `complex64` | `complex128` | Default |
|-----------|:---------:|:----------:|:---------:|:---------:|:-----------:|:------------:|---------|
| Batched GEMM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| Convolution | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| FFT | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| Einsum | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| Memory Traffic | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| Heat Equation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| Schrödinger | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | float32 |
| **Atomic Contention** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | float32 |
| **Sparse MM** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | float32 |

> **Note:** Atomic Contention does not support complex types. Sparse MM only supports `float32` and `float64` (PyTorch limitation for `torch.sparse.mm`). Using TF32 mode (`--batched-gemm-TF32-mode`) forces `float32` regardless of `--precision-gemm`.

### Batched GEMM
| Option | Description |
|--------|-------------|
| `--batched-gemm` | Enable this benchmark. |
| `--batch-count-gemm <int>` | Batch size (default: 128). |
| `--m / --n / --k <int>` | Matrix sizes M×K · K×N (default: 512 each). |
| `--inner-loop-batched-gemm <int>` | Timed iterations (default: 10). |
| `--precision-gemm <dtype>` | Data type (default: _float32_). See [Supported Precisions](#supported-precisions). |
| `--batched-gemm-TF32-mode` | Allow TF32 (if hardware support exists). |

### Convolution
| Option | Description |
|--------|-------------|
| `--convolution` | Enable convolution test. |
| `--batch-count-convolution <int>` | Batch size (default: 128). |
| `--in-channels / --out-channels <int>` | Default: 3 / 64. |
| `--height / --width <int>` | Input size (default: 128×128). |
| `--kernel-size <int>` | Kernel size (default: 3). |
| `--inner-loop-convolution <int>` | Timed iterations (default: 10). |
| `--precision-convolution <dtype>` | Data type (default: _float32_). See [Supported Precisions](#supported-precisions). |

### FFT
| Option | Description |
|--------|-------------|
| `--fft` | Enable 3-D FFT test. |
| `--batch-count-fft <int>` | Batch size (default: 128). |
| `--nx / --ny / --nz <int>` | Grid size (default: 128³). |
| `--inner-loop-fft <int>` | Timed iterations (default: 10). |
| `--precision-fft <dtype>` | Data type (default: _float32_). See [Supported Precisions](#supported-precisions). |

### Einsum
| Option | Description |
|--------|-------------|
| `--einsum` | Enable einsum test. |
| `--batch-count-einsum <int>` | Batch size (default: 128). |
| `--heads / --seq-len / --d-model <int>` | Default: 8 / 128 / 64. |
| `--inner-loop-einsum <int>` | Timed iterations (default: 10). |
| `--precision-einsum <dtype>` | Data type (default: _float32_). See [Supported Precisions](#supported-precisions). |

### Memory Traffic
| Option | Description |
|--------|-------------|
| `--memory-traffic` | Enable random traffic test. |
| `--memory-size <int>` | Elements in array (default: _1024_). |
| `--memory-iterations <int>` | Inner loop per timing (default: _10_). |
| `--memory-pattern <pattern>` | Access pattern: `random` (random indexing), `streaming` (sequential), `unit` (stride-1). Default: _random_. |
| `--inner-loop-memory-traffic <int>` | Timed iterations (default: _10_). |
| `--precision-memory <dtype>` | Data type (default: _float32_). Supports all 6 precisions. |

### Heat Equation
| Option | Description |
|--------|-------------|
| `--heat-equation` | Enable Laplacian stencil solver. |
| `--heat-grid-size <int>` | Grid size (default: 128). |
| `--heat-time-steps <int>` | Steps (default: 100). |
| `--alpha <float>` | Thermal diffusivity (default: 0.01). |
| `--delta-t <float>` | Time increment (default: 0.01). |
| `--inner-loop-heat-equation <int>` | Timed iterations (default: _10_). |
| `--precision-heat <dtype>` | Data type (default: _float32_). See [Supported Precisions](#supported-precisions). |

### Schrödinger Equation
| Option | Description |
|--------|-------------|
| `--schrodinger` | Enable quantum simulation. |
| `--schrodinger-grid-size <int>` | Grid points (default: 128). |
| `--schrodinger-time-steps <int>` | Steps (default: 100). |
| `--schrodinger-delta-x / --schrodinger-delta-t <float>` | Default: 0.1 / 0.01. |
| `--schrodinger-hbar / --schrodinger-mass <float>` | Default: 1.0 / 1.0. |
| `--schrodinger-potential {harmonic, barrier}` | Potential (default: _harmonic_). |
| `--inner-loop-schrodinger <int>` | Timed iterations (default: _10_). |
| `--precision-schrodinger <dtype>` | Data type (default: _float32_). See [Supported Precisions](#supported-precisions). |

### Atomic Contention (L2 Cache Stress)
| Option | Description |
|--------|-------------|
| `--atomic-contention` | Enable L2 cache atomic stress test. |
| `--atomic-target-size <int>` | Size of target array (default: 1,000,000). |
| `--atomic-num-updates <int>` | Number of scatter_add updates per iter (default: 10,000,000). |
| `--atomic-contention-range <int>` | Max unique indices; lower = more contention (default: 1024). |
| `--inner-loop-atomic <int>` | Timed iterations (default: 50). |
| `--precision-atomic <dtype>` | Data type (default: _float32_). **No complex types** — choices: `float16`, `bfloat16`, `float32`, `float64`. |

### Sparse Matrix Multiplication (SpMM)
| Option | Description |
|--------|-------------|
| `--sparse-mm` | Enable sparse matrix multiply test. |
| `--sparse-m <int>` | Sparse matrix rows (default: 8192). |
| `--sparse-n <int>` | Dense/output columns (default: 8192). |
| `--sparse-k <int>` | Sparse cols / Dense rows (default: 8192). |
| `--sparse-density <float>` | Sparsity (0.10 = 10% non-zeros, default: 0.10). |
| `--inner-loop-sparse <int>` | Timed iterations (default: 50). |
| `--precision-sparse <dtype>` | Data type (default: _float32_). **No complex types** — choices: `float16`, `bfloat16`, `float32`, `float64`. |

---

## YAML Configuration

For complex test suites, use YAML configuration files instead of long command lines. Supports all benchmarks with full parameter control and multiple test instances.

Example `config.yaml`:
```yaml
# Global settings
verbose: true
log_file: "stress-test.log"
device_index: 0
warmup: 20
repeats: 3
repeat_delay: 10
# Output modes (uncomment to enable)
# compact: true          # Machine-readable CSV to stdout
# syslog: true           # Structured key=value entries to syslog
# syslog_dmesg: true     # Also write to /dev/kmsg (requires syslog: true)

# Benchmark suite - can specify same test multiple times with different params
benchmarks:
  - name: batched_gemm
    precision: float32
    batch_count: 128
    m: 4096
    n: 4096
    k: 4096
    inner_loop: 100
    
  - name: batched_gemm
    precision: float64
    batch_count: 64
    m: 8192
    n: 8192
    k: 8192
    TF32_mode: false
    
  - name: convolution
    precision: bfloat16
    batch_count: 256
    in_channels: 64
    out_channels: 128
    
  - name: fft
    precision: float32
    nx: 256
    ny: 256
    nz: 256
    
  - name: heat_equation
    precision: float64
    grid_size: 32768
    time_steps: 100
```

Run with:
```bash
./torch-hammer.py --config config.yaml
```

CLI arguments override YAML settings:
```bash
# Use config but override to run on GPU 2
./torch-hammer.py --config config.yaml --device-index 2

# Suppress stdout while keeping verbose CSV in log file
./torch-hammer.py --config config.yaml --verbose-file-only
```

See `config-examples/` directory for ready-to-use configurations (`quick-test.yaml`, `stress-test.yaml`).

---

## Hardware Baselines (Performance Validation)

Torch Hammer can compare your measured results against expected hardware performance and flag
issues automatically. This is **opt-in** — no validation runs unless you supply a baseline file.

### Quick Start

```bash
# Run with baseline validation
./torch-hammer.py --batched-gemm --baseline-file baselines/example.yaml

# Run without validation (the default)
./torch-hammer.py --batched-gemm

# Explicitly disable validation even if a baseline file is loaded
./torch-hammer.py --batched-gemm --baseline-file my_baselines.yaml --no-validation
```

### Creating a Baseline File

#### Step 1 — Find your GPU model name

The model name in the baseline file must match what Torch Hammer detects.
Run a quick test and look for the `model` field:

```bash
./torch-hammer.py --batched-gemm --inner-loop-batched-gemm 1 --verbose 2>&1 | grep model
# → 'model': 'NVIDIA GH200 120GB'

# Or use vendor tools directly:
nvidia-smi --query-gpu=name --format=csv,noheader   # NVIDIA
rocm-smi --showproductname                            # AMD
```

#### Step 2 — Create a YAML or JSON file

**YAML** (recommended — supports comments):
```yaml
"NVIDIA GH200 120GB":
  benchmarks:
    batched_gemm:
      float32:
        target_gflops: 49000.0
        min_efficiency: 90.0
      float64:
        target_gflops: 43000.0
        min_efficiency: 85.0
      tf32:
        target_gflops: 252000.0
        min_efficiency: 85.0
    heat_equation:
      float64:
        target_mlups: 20000.0
        min_efficiency: 80.0
    memory_traffic:
      float32:
        target_gbps: 4500.0
        min_efficiency: 75.0
```

This target-based format lets you set per-benchmark, per-dtype expected values
and minimum efficiency thresholds. If a benchmark/dtype pair isn't found in
`benchmarks:`, validation falls back to the top-level `fp32_tflops`/`fp64_tflops`
values automatically.

**JSON** alternative:
```json
{
  "NVIDIA GH200 120GB": {
    "fp32_tflops": 51.0,
    "fp64_tflops": 25.5,
    "tf32_tflops": 756.0,
    "memory_bandwidth_gbps": 4800.0,
    "tdp_watts": 900
  }
}
```

#### Step 3 — Multi-GPU baseline file

For sites with multiple GPU types, list them all in one file.
Keys must match the model string exactly (case-sensitive):

```yaml
# my_baselines.yaml
# Keys must match the model string exactly (case-sensitive)
# The performance figures below are examples, please check vendor 
# datasheets for specific figures 
"NVIDIA GH200 120GB":
  fp32_tflops: 51.0        # FP32 peak TFLOPS
  fp64_tflops: 25.5        # FP64 peak TFLOPS
  tf32_tflops: 756.0       # TF32 peak TFLOPS (Tensor Core)
  memory_bandwidth_gbps: 4800.0   # HBM3e bandwidth in GB/s
  tdp_watts: 900           # Thermal Design Power in Watts

"AMD Instinct MI300X":
  fp32_tflops: 163.4
  fp64_tflops: 81.7
  memory_bandwidth_gbps: 5300.0
  tdp_watts: 750

"NVIDIA A100-SXM4-80GB":
  fp32_tflops: 19.5
  fp64_tflops: 9.7
  tf32_tflops: 156.0
  memory_bandwidth_gbps: 2039.0
  tdp_watts: 400

"AMD Instinct MI250X":
  fp32_tflops: 47.9
  fp64_tflops: 47.9
  memory_bandwidth_gbps: 3276.8
  tdp_watts: 500
```

### Baseline Fields Reference

| Field | Unit | Used By |
|-------|------|---------|
| `fp32_tflops` | TFLOPS | GEMM, Convolution, FFT, Einsum (float32) |
| `fp64_tflops` | TFLOPS | GEMM, Convolution, FFT, Einsum (float64) |
| `fp16_tflops` | TFLOPS | GEMM, Convolution, FFT, Einsum (float16/bfloat16) |
| `tf32_tflops` | TFLOPS | GEMM with `--batched-gemm-TF32-mode` |
| `memory_bandwidth_gbps` | GB/s | Memory Traffic, Heat Equation, Schrödinger |
| `tdp_watts` | Watts | Informational (not used for validation today) |

### Validation Output

When baselines are loaded, Torch Hammer reports efficiency alongside results:

```
✅ Excellent performance: 96.3% of target (49000.0 gflops)
✅ Good performance: 82.1% of target (43000.0 gflops)
[WARN] Performance below target: 58.2% of expected 49000.0 gflops (threshold: 70%)
```

### Tuning Thresholds

The warning thresholds can be adjusted per-run via CLI flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--efficiency-warn-pct` | 70% | Flag results below this % of peak/target |
| `--temp-warn-C` | 90°C | Temperature warning threshold |
| `--temp-critical-C` | 95°C | Temperature critical threshold |
| `--power-warn-pct` | 98% | Power-limit proximity warning |
| `--outlier-threshold-pct` | 15% | Multi-GPU outlier detection |

See `baselines/` directory for ready-to-use example files.

---

## Verbose
`--verbose` Presents every iteration into a single row:


Save it directly to a file via `--log-file myrun.txt`.

---

## Compact Mode

`--compact` produces **machine-readable CSV on stdout** — one row per benchmark,
with all log chatter suppressed. 

### Basic Usage

```bash
# CSV to stdout, warnings-only on stderr
./torch-hammer.py --compact --batched-gemm --fft > results.csv

# Pipe-friendly: only CSV reaches the file
./torch-hammer.py --compact --batched-gemm --fft 2>/dev/null > results.csv
```

### Columns (14 base)

| Column | Description |
|--------|-------------|
| `hostname` | Node hostname |
| `gpu` | GPU index (0, 1, …) |
| `gpu_model` | GPU model string |
| `serial` | GPU serial number |
| `benchmark` | Benchmark name (e.g. `Batched GEMM`) |
| `dtype` | Data type used |
| `iterations` | Number of timed iterations completed |
| `runtime_s` | Wall-clock time for the timed loop (seconds) |
| `min` | Minimum performance value |
| `mean` | Mean performance value |
| `max` | Maximum performance value |
| `unit` | Performance unit (`GFLOP/s`, `GB/s`, etc.) |
| `power_avg_w` | Average power draw (watts) |
| `temp_max_c` | Peak GPU temperature (°C) |

### Verbose Extras (`--compact --verbose`, 19 columns)

Adding `--verbose` appends five telemetry columns:

| Column | Description |
|--------|-------------|
| `sm_util_mean` | Mean SM / CU utilisation (%) |
| `mem_bw_util_mean` | Mean memory-bandwidth utilisation (%) |
| `gpu_clock_mean` | Mean GPU clock (MHz) |
| `mem_used_gb_mean` | Mean memory used (GB) |
| `throttled` | `true` / `false` — whether throttling was detected |

```bash
# 19-column CSV with extra telemetry
./torch-hammer.py --compact --verbose --batched-gemm > detailed.csv
```

### Multi-GPU

In multi-GPU mode (`--all-gpus` / `--gpu-list`) the parent process prints a
single CSV header before spawning workers; each worker appends its own data
rows.  The normal multi-GPU summary table is suppressed — the CSV **is** the
summary.

```bash
./torch-hammer.py --compact --all-gpus --batched-gemm > results.csv
```

### Behavior Notes

- **stdout** = pure CSV (header + data rows).  
- **stderr** = warnings / errors only (log level `WARNING`).  
- `--compact` does **not** emit per-iteration lines; only `--verbose` does.  
- Combine `--compact --verbose` to get extra telemetry **columns** on the summary row (not extra rows).

---

## Syslog Mode

`--syslog` emits **structured `key=value` entries to the system log** after each
benchmark completes.  This enables integration with log aggregators such as
**Splunk**, **Elastic**, and **journald** without scraping CSV files.

### Basic Usage

```bash
# Emit to syslog (LOG_USER facility)
./torch-hammer.py --syslog --batched-gemm --fft

# Also write to dmesg ring buffer (requires root / CAP_SYSLOG)
sudo ./torch-hammer.py --syslog --syslog-dmesg --all-gpus

# Combine with compact CSV output
./torch-hammer.py --syslog --compact --batched-gemm > results.csv
```

### Message Types

| Tag | When | Example |
|-----|------|---------|
| `RUN_START` | Before first benchmark | `RUN_START run_id=a3f1b2c4 ts=2025-… host=node01 gpus=4` |
| `BENCH_RESULT` | After each benchmark | `BENCH_RESULT run_id=a3f1b2c4 hostname=node01 gpu=0 benchmark=Batched_GEMM mean=42.5 unit=TFLOPS …` |
| `RUN_END` | After last benchmark | `RUN_END run_id=a3f1b2c4 ts=2025-… passed=9 failed=0 total_elapsed=123.5s` |

### Severity Derivation

Syslog priority is derived from your existing threshold flags — no extra
configuration required:

| Condition | Severity |
|-----------|----------|
| `status=FAIL` | `LOG_ERR` |
| `temp_max_c ≥ --temp-critical-C` | `LOG_CRIT` |
| `temp_max_c ≥ --temp-warn-C` | `LOG_WARNING` |
| `throttled=true` | `LOG_WARNING` |
| `efficiency_pct < --efficiency-warn-pct` | `LOG_WARNING` |
| Clean pass | `LOG_INFO` |

### KV Fields

Every message includes a `run_id` — the first 8 hex characters of a UUID4
generated at invocation time.  Use it to correlate all messages from a single
run (e.g. `journalctl | grep run_id=a3f1b2c4`).

Each `BENCH_RESULT` entry contains the same fields as
[Compact Mode](#compact-mode) plus `status`, `efficiency_pct` (when baselines
are loaded), and `throttled` (when detected).  Spaces in values are replaced
with underscores.

### dmesg Integration (`--syslog-dmesg`)

`--syslog-dmesg` writes the same messages to `/dev/kmsg` so they appear in
`dmesg` alongside kernel events.  This is useful for correlating GPU issues
with Xid/MCE errors.


If `/dev/kmsg` is unavailable or the process lacks permissions, a warning is
printed and execution continues with syslog only — **dmesg failure never
crashes the benchmark**.

### Multi-GPU

In multi-GPU mode the parent generates a **single `run_id`** shared by all
workers and the parent framing messages.  Every `RUN_START`, `BENCH_RESULT`,
and `RUN_END` from the same invocation carries the same `run_id`:

```bash
journalctl -t torch-hammer -t torch-hammer-mgpu | grep run_id=c4d4a27d
```

The parent emits a framing `RUN_START` (with `gpus=N`) and `RUN_END` with
aggregate pass/fail counts using the `torch-hammer-mgpu` syslog tag.

### Behavior Notes

- `--syslog-dmesg` requires `--syslog` (same pattern as `--verbose-file-only` requires `--log-file`).
- Syslog uses the `LOG_USER` facility with tag `torch-hammer` (or `torch-hammer-mgpu` for the parent frame).
- All messages are emitted **per-benchmark** (not batched) for crash survivability.
- `--syslog` and `--compact` can be combined — they are independent output channels.

---

## Telemetry Back-ends
| Hardware | Requirements | Reported Fields* |
|----------|--------------|------------------|
| NVIDIA GPU | `pynvml` (nvidia-ml-py) | power, temperature, GPU utilization, clock, memory controller activity, memory utilization, GPU temperature, SM utilization, HBM temperature, thermal throttling status, power limit warnings, ECC errors |
| AMD ROCm GPU/APU | `amdsmi` Python library | GPU utilization, memory utilization, GPU clock, memory clock, power, temperature (edge/hotspot/memory), serial, throttle status, power cap, thermal warnings |
| CPU only | None | Vendor, architecture |

\* AMD telemetry uses native `amdsmi` Python API for ROCm 6.1+ with comprehensive metrics including throttle detection, power limit monitoring, and thermal warnings. Per-processor-type field sets automatically adapt for GPU vs CPU/APU.

---

## Examples
Run the GEMM test against an NVIDIA GH200 module
```
./torch-hammer.py --batched-gemm --k 16384 --m 4224 --n 2048 --verbose
2025-10-28T21:19:11 INFO    Using device cuda:0
2025-10-28T21:19:11 INFO    Initial telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 2, 'mem_bw_util': 0, 'mem_util': '0.6', 'gpu_clock': 1980, 'mem_clock': 2619, 'power_W': 106.075, 'temp_gpu_C': 34}
2025-10-28T21:19:20 INFO    iter, test, dtype, gflops, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, power_W, temp_gpu_C, temp_hbm_C
2025-10-28T21:19:20 INFO    1, gemm, float32, 49098.63, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1770, 2619, 567.324, 55,
2025-10-28T21:19:20 INFO    2, gemm, float32, 49008.41, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1770, 2619, 566.19, 55,
2025-10-28T21:19:21 INFO    3, gemm, float32, 49011.46, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1785, 2619, 566.174, 55,
2025-10-28T21:19:22 INFO    4, gemm, float32, 49035.82, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1785, 2619, 567.226, 55,
2025-10-28T21:19:23 INFO    5, gemm, float32, 49054.14, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1785, 2619, 566.38, 55,
2025-10-28T21:19:23 INFO    6, gemm, float32, 49003.10, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1770, 2619, 565.378, 55,
2025-10-28T21:19:24 INFO    7, gemm, float32, 49041.47, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1770, 2619, 565.83, 55,
2025-10-28T21:19:25 INFO    8, gemm, float32, 49069.36, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1785, 2619, 567.327, 55,
2025-10-28T21:19:26 INFO    9, gemm, float32, 48968.03, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1755, 2619, 565.995, 56,
2025-10-28T21:19:26 INFO    10, gemm, float32, 49032.55, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 29, 56.6, 1770, 2619, 566.18, 56,
2025-10-28T21:19:26 INFO    [Batched GEMM] 48968.03/49032.30/49098.63 GFLOP/s (min/mean/max) {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 29, 'mem_util': '56.6', 'gpu_clock': 1770, 'mem_clock': 2619, 'power_W': 566.18, 'temp_gpu_C': 56}
2025-10-28T21:19:26 INFO    Final telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 29, 'mem_util': '56.6', 'gpu_clock': 1770, 'mem_clock': 2619, 'power_W': 566.18, 'temp_gpu_C': 56}
2025-10-28T21:19:26 INFO    Benchmark run finished.
```

Run the GEMM test against an NVIDIA GH200 module using the float64 data type
```
./torch-hammer.py --batch-count-gemm=106 --batched-gemm --k 16384 --m 4224 --n 2048 --precision-gemm float64 --verbose
2025-10-28T23:10:54 INFO    Using device cuda:0
2025-10-28T23:10:54 INFO    Initial telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 2, 'mem_bw_util': 0, 'mem_util': '0.6', 'gpu_clock': 1980, 'mem_clock': 2619, 'power_W': 106.409, 'temp_gpu_C': 35}
2025-10-28T23:11:01 INFO    iter, test, dtype, gflops, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, power_W, temp_gpu_C, temp_hbm_C
2025-10-28T23:11:01 INFO    1, gemm, float64, 43615.45, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 50, 93.3, 1335, 2619, 565.706, 53,
2025-10-28T23:11:02 INFO    2, gemm, float64, 43564.93, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 50, 93.3, 1320, 2619, 565.575, 53,
2025-10-28T23:11:03 INFO    3, gemm, float64, 43479.01, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 51, 93.3, 1320, 2619, 565.031, 53,
2025-10-28T23:11:03 INFO    4, gemm, float64, 43452.38, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 51, 93.3, 1320, 2619, 564.582, 54,
2025-10-28T23:11:04 INFO    5, gemm, float64, 43375.87, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 50, 93.3, 1320, 2619, 564.172, 54,
2025-10-28T23:11:05 INFO    6, gemm, float64, 43448.23, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 51, 93.3, 1320, 2619, 563.456, 54,
2025-10-28T23:11:05 INFO    7, gemm, float64, 43404.56, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 51, 93.3, 1320, 2619, 563.842, 54,
2025-10-28T23:11:06 INFO    8, gemm, float64, 43397.94, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 50, 93.3, 1305, 2619, 563.7, 54,
2025-10-28T23:11:07 INFO    9, gemm, float64, 43472.86, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 51, 93.3, 1320, 2619, 563.753, 54,
2025-10-28T23:11:08 INFO    10, gemm, float64, 43436.16, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 51, 93.3, 1335, 2619, 563.484, 54,
2025-10-28T23:11:08 INFO    [Batched GEMM] 43375.87/43464.74/43615.45 GFLOP/s (min/mean/max) {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 51, 'mem_util': '93.3', 'gpu_clock': 1335, 'mem_clock': 2619, 'power_W': 563.484, 'temp_gpu_C': 54}
2025-10-28T23:11:08 INFO    Final telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 51, 'mem_util': '93.3', 'gpu_clock': 1335, 'mem_clock': 2619, 'power_W': 563.484, 'temp_gpu_C': 54}
2025-10-28T23:11:08 INFO    Benchmark run finished.
```


Run the GEMM test, but utilize Tensor Cores (if available):
```
./torch-hammer.py --batch-count-gemm=106 --batched-gemm --k 16384 --m 4224 --n 2048 --batched-gemm-TF32-mode --verbose
2025-10-28T23:23:40 INFO    Using device cuda:0
2025-10-28T23:23:40 INFO    Initial telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 0, 'mem_bw_util': 0, 'mem_util': '0.6', 'gpu_clock': 1980, 'mem_clock': 2619, 'power_W': 106.708, 'temp_gpu_C': 34}
2025-10-28T23:23:41 INFO    iter, test, dtype, gflops, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, power_W, temp_gpu_C, temp_hbm_C
2025-10-28T23:23:41 INFO    1, gemm, float32, 256246.11, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 67, 47.0, 1260, 2619, 559.151, 48,
2025-10-28T23:23:41 INFO    2, gemm, float32, 253168.00, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 67, 47.0, 1260, 2619, 558.875, 48,
2025-10-28T23:23:41 INFO    3, gemm, float32, 251230.15, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 70, 47.0, 1230, 2619, 559.197, 48,
2025-10-28T23:23:41 INFO    4, gemm, float32, 252279.34, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 70, 47.0, 1245, 2619, 557.965, 48,
2025-10-28T23:23:42 INFO    5, gemm, float32, 253690.52, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 72, 47.0, 1245, 2619, 557.214, 48,
2025-10-28T23:23:42 INFO    6, gemm, float32, 251942.72, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 68, 47.0, 1260, 2619, 556.396, 49,
2025-10-28T23:23:42 INFO    7, gemm, float32, 252252.57, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 68, 47.0, 1245, 2619, 558.838, 49,
2025-10-28T23:23:42 INFO    8, gemm, float32, 252393.06, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 69, 47.0, 1245, 2619, 561.048, 49,
2025-10-28T23:23:42 INFO    9, gemm, float32, 252073.11, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 69, 47.0, 1260, 2619, 559.898, 49,
2025-10-28T23:23:42 INFO    10, gemm, float32, 252572.02, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 67, 47.0, 1260, 2619, 558.2, 49,
2025-10-28T23:23:42 INFO    [Batched GEMM] 251230.15/252784.76/256246.11 GFLOP/s (min/mean/max) {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 67, 'mem_util': '47.0', 'gpu_clock': 1260, 'mem_clock': 2619, 'power_W': 558.2, 'temp_gpu_C': 49}
2025-10-28T23:23:42 INFO    Final telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 67, 'mem_util': '47.0', 'gpu_clock': 1260, 'mem_clock': 2619, 'power_W': 558.2, 'temp_gpu_C': 49}
2025-10-28T23:23:42 INFO    Benchmark run finished.
```

Run the Laplacian Heat Equation
```
./torch-hammer.py --heat-equation --heat-grid-size 32768 --heat-time-steps 100 --alpha .000127 --delta-t .001 --precision-heat float64 --verbose
2025-10-28T23:41:52 INFO    Using device cuda:0
2025-10-28T23:41:52 INFO    Initial telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 2, 'mem_bw_util': 0, 'mem_util': '0.6', 'gpu_clock': 1980, 'mem_clock': 2619, 'power_W': 105.937, 'temp_gpu_C': 34}
2025-10-28T23:41:52 INFO    iter, test, dtype, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, power_W, temp_gpu_C, temp_hbm_C
2025-10-28T23:41:52 INFO    0.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    1.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    2.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    3.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    4.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    5.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    6.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    7.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    8.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    9.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    10.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    11.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    12.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    13.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    14.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    15.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    16.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    17.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    18.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    19.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    20.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    21.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    22.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    23.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    24.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    25.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    26.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    27.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    28.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    29.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    30.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 113.762, 35,
2025-10-28T23:41:52 INFO    31.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 2, 0, 42.8, 1980, 2619, 154.74, 35,
2025-10-28T23:41:52 INFO    32.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 93, 100, 42.8, 1980, 2619, 154.74, 37,
2025-10-28T23:41:52 INFO    33.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 93, 100, 42.8, 1635, 2619, 204.161, 37,
2025-10-28T23:41:52 INFO    34.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 93, 100, 42.8, 1635, 2619, 204.161, 39,
2025-10-28T23:41:52 INFO    35.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1575, 2619, 256.126, 39,
2025-10-28T23:41:52 INFO    36.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1575, 2619, 256.126, 41,
2025-10-28T23:41:52 INFO    37.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1605, 2619, 305.641, 41,
2025-10-28T23:41:52 INFO    38.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1605, 2619, 305.641, 42,
2025-10-28T23:41:52 INFO    39.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1605, 2619, 365.741, 42,
2025-10-28T23:41:52 INFO    40.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1605, 2619, 365.741, 42,
2025-10-28T23:41:52 INFO    41.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1620, 2619, 409.326, 42,
2025-10-28T23:41:52 INFO    42.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1620, 2619, 409.326, 44,
2025-10-28T23:41:52 INFO    43.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1545, 2619, 464.929, 44,
2025-10-28T23:41:53 INFO    44.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1575, 2619, 464.929, 44,
2025-10-28T23:41:53 INFO    45.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1575, 2619, 506.784, 44,
2025-10-28T23:41:53 INFO    46.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1560, 2619, 506.784, 44,
2025-10-28T23:41:53 INFO    47.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1560, 2619, 552.12, 44,
2025-10-28T23:41:53 INFO    48.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1395, 2619, 552.12, 44,
2025-10-28T23:41:53 INFO    49.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1395, 2619, 589.225, 44,
2025-10-28T23:41:53 INFO    50.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1380, 2619, 585.284, 44,
2025-10-28T23:41:53 INFO    51.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1380, 2619, 585.284, 44,
2025-10-28T23:41:53 INFO    52.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1380, 2619, 576.641, 44,
2025-10-28T23:41:53 INFO    53.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1380, 2619, 576.641, 44,
2025-10-28T23:41:53 INFO    54.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1485, 2619, 572.522, 44,
2025-10-28T23:41:53 INFO    55.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1485, 2619, 572.522, 45,
2025-10-28T23:41:53 INFO    56.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 571.509, 45,
2025-10-28T23:41:53 INFO    57.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 571.509, 45,
2025-10-28T23:41:53 INFO    58.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 570.536, 45,
2025-10-28T23:41:53 INFO    59.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 570.536, 45,
2025-10-28T23:41:53 INFO    60.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 569.789, 45,
2025-10-28T23:41:53 INFO    61.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 569.789, 46,
2025-10-28T23:41:53 INFO    62.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1545, 2619, 570.795, 46,
2025-10-28T23:41:54 INFO    63.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1545, 2619, 570.795, 46,
2025-10-28T23:41:54 INFO    64.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1515, 2619, 569.745, 46,
2025-10-28T23:41:54 INFO    65.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1515, 2619, 569.745, 46,
2025-10-28T23:41:54 INFO    66.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1425, 2619, 566.433, 46,
2025-10-28T23:41:54 INFO    67.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1395, 2619, 566.433, 46,
2025-10-28T23:41:54 INFO    68.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1395, 2619, 567.298, 46,
2025-10-28T23:41:54 INFO    69.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1395, 2619, 567.298, 46,
2025-10-28T23:41:54 INFO    70.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1395, 2619, 568.042, 46,
2025-10-28T23:41:54 INFO    71.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1425, 2619, 568.042, 46,
2025-10-28T23:41:54 INFO    72.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1425, 2619, 569.252, 46,
2025-10-28T23:41:54 INFO    73.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1440, 2619, 570.087, 46,
2025-10-28T23:41:54 INFO    74.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1440, 2619, 570.087, 46,
2025-10-28T23:41:54 INFO    75.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1515, 2619, 569.602, 46,
2025-10-28T23:41:54 INFO    76.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1515, 2619, 569.602, 46,
2025-10-28T23:41:54 INFO    77.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.971, 46,
2025-10-28T23:41:54 INFO    78.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.971, 46,
2025-10-28T23:41:54 INFO    79.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 569.572, 46,
2025-10-28T23:41:54 INFO    80.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1530, 2619, 569.572, 47,
2025-10-28T23:41:55 INFO    81.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 569.063, 47,
2025-10-28T23:41:55 INFO    82.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 569.063, 47,
2025-10-28T23:41:55 INFO    83.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1455, 2619, 567.604, 47,
2025-10-28T23:41:55 INFO    84.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1455, 2619, 567.604, 47,
2025-10-28T23:41:55 INFO    85.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1425, 2619, 567.397, 47,
2025-10-28T23:41:55 INFO    86.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1425, 2619, 567.397, 47,
2025-10-28T23:41:55 INFO    87.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1410, 2619, 567.496, 47,
2025-10-28T23:41:55 INFO    88.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1410, 2619, 567.496, 47,
2025-10-28T23:41:55 INFO    89.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1410, 2619, 568.607, 47,
2025-10-28T23:41:55 INFO    90.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1410, 2619, 568.607, 46,
2025-10-28T23:41:55 INFO    91.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1425, 2619, 569.332, 46,
2025-10-28T23:41:55 INFO    92.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1485, 2619, 569.332, 46,
2025-10-28T23:41:55 INFO    93.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1485, 2619, 569.982, 46,
2025-10-28T23:41:55 INFO    94.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 569.982, 47,
2025-10-28T23:41:55 INFO    95.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.653, 47,
2025-10-28T23:41:55 INFO    96.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.653, 47,
2025-10-28T23:41:55 INFO    97.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.576, 47,
2025-10-28T23:41:55 INFO    98.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.576, 47,
2025-10-28T23:41:55 INFO    99.00, heat, float64, NVIDIA, NVIDIA GH200 120GB, nid001000, 0, 1652422128547, 100, 100, 42.8, 1500, 2619, 568.041, 47,
2025-10-28T23:41:57 INFO    [Heat] 20048.3 MLUPS total 5.36s {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 100, 'mem_util': '42.8', 'gpu_clock': 1455, 'mem_clock': 2619, 'power_W': 567.965, 'temp_gpu_C': 47}
2025-10-28T23:41:57 INFO    Final telemetry {'vendor': 'NVIDIA', 'model': 'NVIDIA GH200 120GB', 'device_id': 0, 'hostname': 'nid001000', 'serial': '1652422128547', 'sm_util': 100, 'mem_bw_util': 100, 'mem_util': '42.8', 'gpu_clock': 1455, 'mem_clock': 2619, 'power_W': 567.965, 'temp_gpu_C': 47}
2025-10-28T23:41:57 INFO    Benchmark run finished.
```

### Multi-GPU Parallel Execution

Run benchmarks on all GPUs simultaneously with per-GPU log files:
```bash
./torch-hammer.py --batched-gemm --all-gpus --log-dir ./logs --verbose-file-only

# Each GPU writes to separate file: logs/gpu0_<timestamp>.log, logs/gpu1_<timestamp>.log, etc.
# No stdout pollution - all CSV data goes to files only
```

Run on specific GPU subset with NUMA-aware CPU binding (enabled by default):
```bash
./torch-hammer.py --batched-gemm --gpu-list "0,2,3"
2025-12-08T15:23:10 INFO    GPU 0 on NUMA node 0, using all NUMA CPUs: 0-31
2025-12-08T15:23:10 INFO    GPU 2 on NUMA node 1, using all NUMA CPUs: 32-63
2025-12-08T15:23:10 INFO    GPU 3 on NUMA node 1, using all NUMA CPUs: 32-63
```

Manual CPU-GPU mapping for fine-grained control:
```bash
./torch-hammer.py --batched-gemm --gpu-list "0,1" --cpu-gpu-map "0:0-15,1:16-31"
```

### Repeated Runs for Statistical Analysis

Run benchmark suite multiple times for stability testing:
```bash
# Run 10 times to gather statistical distribution
./torch-hammer.py --batched-gemm --repeats 10

============================================================
REPEAT 1/10
============================================================
[... benchmark output ...]

============================================================
REPEAT 2/10
============================================================
[... benchmark output ...]
```

With thermal stabilization delay between repeats:
```bash
# Run 5 times with 30-second cooling period between runs
./torch-hammer.py --batched-gemm --repeats 5 --repeat-delay 30
```

Verbose mode includes repeat number in CSV output:
```bash
./torch-hammer.py --batched-gemm --repeats 3 --verbose --log-file stability.csv

# CSV format:
# repeat, iter, test, dtype, gflops, vendor, model, ...
# 1, 1, gemm, float32, 49123.45, NVIDIA, ...
# 1, 2, gemm, float32, 49087.23, NVIDIA, ...
# ...
# 2, 1, gemm, float32, 48956.12, NVIDIA, ...
# 2, 2, gemm, float32, 48998.76, NVIDIA, ...
```

### Duration-Based Testing

Run each benchmark for a specific duration instead of fixed iterations:
```bash
# Run GEMM for 60 seconds per repeat, repeated 3 times
./torch-hammer.py --batched-gemm --duration 60 --repeats 3

# Total runtime: ~60s × 3 repeats = ~180s
# (Plus optional --repeat-delay between repeats)
```

Combined with multi-GPU for cluster stress testing:
```bash
# All 8 GPUs run for 5 minutes each (NUMA binding enabled by default)
./torch-hammer.py --all-gpus \
  --batched-gemm --convolution --fft \
  --duration 300 \
  --log-dir ./stress-test-logs \
  --verbose-file-only
```

---

## Contributing
We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) 
before submitting pull requests.

**Key requirements:**
- All commits must be signed off (DCO) - see [CONTRIBUTING.md](CONTRIBUTING.md#developer-certificate-of-origin-dco)
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Run syntax check: `python3 -m py_compile torch-hammer.py`

**Areas where help is especially welcome:**
* Additional benchmarks or kernels  
* Extended telemetry support:
  - Additional AMD metrics (PCIe bandwidth, per-CU utilization)
  - macOS Metal Performance Shaders (MPS) telemetry
* Packaging / CI for wheels and Homebrew formula  
* Windows support and testing

**Communication:**
- **Issues & PRs**: Use the GitHub Issues and Pull Requests tabs

---

## Testing

Torch Hammer includes a comprehensive test suite to ensure reliability across updates.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_parsing.py -v      # CLI argument parsing
pytest tests/test_compact.py -v      # Compact CSV output mode
pytest tests/test_syslog.py -v       # Syslog output mode
pytest tests/test_utilities.py -v    # Timer, helpers, validation
pytest tests/test_telemetry.py -v    # Telemetry classes
pytest tests/test_smoke.py -v        # Benchmark smoke tests (CPU)

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

### Test Categories

| Test File | Coverage | Description |
|-----------|----------|-------------|
| `test_parsing.py` | CLI & config | Argument parsing, CPU-GPU mapping, validation |
| `test_compact.py` | Compact mode | CSV output, columns, header control, logging suppression |
| `test_syslog.py` | Syslog mode | SyslogReporter, DmesgWriter, KV formatting, priority derivation, pass/fail counting, run_id |
| `test_utilities.py` | Core helpers | Timer, VerbosePrinter, GFLOP calculations |
| `test_telemetry.py` | Telemetry | Class structure, thread behavior, factory |
| `test_smoke.py` | Benchmarks | Run each benchmark on CPU with minimal iterations |

### Writing New Tests

When contributing new features, please include tests:

```python
# tests/test_myfeature.py
def test_my_new_function(th):
    """Test description."""
    result = th.my_new_function(args)
    assert result is not None
    assert result["expected_key"] == expected_value
```

The `th` fixture (defined in `conftest.py`) provides access to the torch-hammer module.

**Note:** GPU-specific tests are skipped on machines without GPU hardware. The smoke tests run all benchmarks on CPU to verify basic functionality without requiring specialized hardware.

### CI Workflow (GitHub Actions)

The GitHub Actions workflow (`.github/workflows/gpu-functional.yml`) runs the
full pytest suite on `ubuntu-latest` across multiple Python versions (3.9, 3.11,
3.12) using CPU-only PyTorch.  No GPU hardware is required.

```bash
# Run locally — identical to what CI runs
pytest tests/ -v
```

For GPU-specific regression testing on HPC clusters, an exhaustive **92-test**
ReFrame suite lives in `reframe/ci_functional_checks.py`.  See
`reframe/README.md` for details on running it with real GPU hardware.

---

## Project Governance

This project is maintained by Hewlett Packard Enterprise. Contributions are reviewed 
by the maintainer team and merged following the process described in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE.md) file for details.

```
Copyright 2024-2026 Hewlett Packard Enterprise Development LP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

