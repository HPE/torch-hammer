# ReFrame Integration for Torch Hammer

> ⚠️ **EXPERIMENTAL**: This ReFrame integration is experimental and under active development. APIs, test structure, and configuration may change without notice. Feedback and contributions are welcome!

This directory contains [ReFrame](https://reframe-hpc.readthedocs.io/) regression tests for the Torch Hammer GPU benchmark suite.

## Prerequisites

1. **ReFrame** >= 4.0
   ```bash
   pip install reframe-hpc
   ```

2. **Torch Hammer dependencies**
   ```bash
   pip install torch nvidia-ml-py  # or amdsmi for AMD
   ```

## Quick Start

### Local Testing (Single GPU)

```bash
# Run all tests with local scheduler
reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py -r

# Run specific test
reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py \
  -n TorchHammerGEMM -r

# Run with specific parameters
reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py \
  -n 'TorchHammerGEMM%precision=float32%matrix_size=8192' -r

# Dry run (show what would run)
reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py --list-detailed
```

### HPC Cluster Usage

1. **Configure your system** in `settings.py`:
   - Set `hostnames` pattern to match your login/compute nodes
   - Configure `scheduler` (slurm, pbs, etc.)
   - Set `access` flags for partitions/accounts
   - Configure `modules` for your environment

2. **Run on cluster**:
   ```bash
   reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py \
     --system=slurm-gpu-cluster:nvidia-a100 -r
   ```

## Available Tests

| Test Class | Description | Key Parameters | Performance Metric |
|------------|-------------|----------------|-------------------|
| `TorchHammerGEMM` | Matrix multiply | precision, matrix_size, tf32_mode | GFLOP/s |
| `TorchHammerConvolution` | 2D convolution | precision, batch_size, kernel_size | img/s |
| `TorchHammerFFT` | 3D FFT | precision, fft_size | GFLOP/s |
| `TorchHammerEinsum` | Attention-style | precision, seq_len, num_heads | GFLOP/s |
| `TorchHammerMemory` | Memory bandwidth | precision, memory_pattern | GB/s |
| `TorchHammerHeat` | Stencil solver | precision, grid_size | MLUPS |
| `TorchHammerSchrodinger` | Quantum simulation | precision, grid_size | iter/s |
| `TorchHammerAtomic` | Atomic contention | precision, contention_range | Mops/s |
| `TorchHammerSparse` | Sparse SpMM | precision, density | GFLOP/s |
| `TorchHammerFullSuite` | All benchmarks | - | Multiple |
| `TorchHammerMultiGPU` | Multi-GPU parallel | num_gpus | GFLOP/s |

## Parameterization

Tests use ReFrame's `parameter()` to generate variants:

```python
# TorchHammerGEMM generates tests for all combinations:
precision = parameter(['float32', 'float16', 'bfloat16', 'float64'])
matrix_size = parameter([4096, 8192, 16384])
# -> 12 test variants (4 precisions × 3 sizes)
```

Filter specific variants:
```bash
# Only float32 tests
reframe -n 'TorchHammerGEMM%precision=float32' -r

# Only 8192 matrix size
reframe -n 'TorchHammerGEMM%matrix_size=8192' -r

# Specific combination
reframe -n 'TorchHammerGEMM%precision=float16%matrix_size=16384' -r
```

## Setting Performance References

Add expected performance baselines in your config:

```python
# In settings.py or test file
reference = {
    'slurm-gpu-cluster:nvidia-a100': {
        'gemm_gflops': (150000, -0.10, 0.10, 'GFLOP/s'),  # 150 TFLOP/s ±10%
        'memory_bandwidth': (2000, -0.05, None, 'GB/s'),   # ≥1900 GB/s
    },
    'slurm-gpu-cluster:amd-mi300': {
        'gemm_gflops': (180000, -0.10, 0.10, 'GFLOP/s'),
    }
}
```

Reference tuple format: `(expected_value, lower_threshold, upper_threshold, unit)`
- Thresholds are fractional (0.10 = 10%)
- Use `None` for unbounded

## Output & Logging

Performance results are logged to:
- **Console**: Real-time progress
- **reframe.log**: Full debug log
- **perflogs/**: CSV performance history

Example performance log entry:
```
2025-01-15T10:30:00|TorchHammerGEMM|slurm-gpu-cluster|nvidia-a100|cuda-12|gemm_gflops=152340.5
```

## Customization

### Add Custom Test Variants

```python
@rfm.simple_test
class MyCustomGEMM(TorchHammerGEMM):
    """Custom GEMM test with TF32 enabled."""
    tf32_mode = variable(bool, value=True)
    matrix_size = parameter([16384])  # Override default sizes
```

### Add Site-Specific Environment

In `settings.py`:
```python
{
    'name': 'mysite-cuda',
    'modules': ['cuda/12.4', 'cudnn/9.0', 'python/3.11'],
    'env_vars': [
        ['PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'],
    ],
    'target_systems': ['mysite-cluster']
}
```

## Troubleshooting

### Test Not Finding Torch Hammer Script

Check the `torch_hammer_script` variable in test class:
```python
torch_hammer_script = variable(str, value='../torch-hammer.py')
```

Adjust path relative to test file location.

### Regex Not Matching Output

Run torch-hammer manually to check output format:
```bash
python3 torch-hammer.py --batched-gemm --m 4096 --n 4096 --k 4096 --duration 10
```

Expected format:
```
[GPU0 Batched GEMM] Performance: 48123.45 / 49456.78 / 50789.01 GFLOP/s (min/mean/max)
```

### Module Load Failures

Ensure modules are available and correctly named in your `settings.py`.

## CI Functional Test Suite

`ci_functional_checks.py` is a separate, exhaustive check file designed for
CI pipelines.  It covers **every benchmark × precision × parameter
combination** (92 tests total) using deliberately tiny tensor sizes so the
full suite finishes in minutes.

### Test Breakdown (92 tests)

| Check class | Parameters | Count |
|-------------|-----------|-------|
| **Individual benchmark × precision** | | **67** |
| `CI_GEMM` | 6 precisions | 6 |
| `CI_GEMM_TF32` | TF32 mode | 1 |
| `CI_Conv` | 6 precisions | 6 |
| `CI_FFT` | 6 precisions | 6 |
| `CI_Einsum` | 6 precisions | 6 |
| `CI_Memory` | 6 precisions × 3 patterns | 18 |
| `CI_Heat` | 6 precisions | 6 |
| `CI_Schrodinger` | 6 precisions × 2 potentials | 12 |
| `CI_Atomic` | 4 real precisions | 4 |
| `CI_Sparse` | 2 supported precisions (float32, float64) | 2 |
| **Precision matrix (all benchmarks together)** | | **16** |
| `CI_PrecisionMatrixStandard` | 7 benchmarks × 6 precisions | 6 |
| `CI_PrecisionMatrixAll` | 9 benchmarks × 4 real precisions | 4 |
| `CI_PrecisionMatrixAtomic` | Atomic + GEMM × 4 real precisions | 4 |
| `CI_PrecisionMatrixSparse` | Sparse + GEMM × 2 supported precisions | 2 |
| **Output / control-path** | | **9** |
| `CI_FullSuite` | All 9 benchmarks | 1 |
| `CI_CompactCSV` | `--compact` | 1 |
| `CI_CompactVerbose` | `--compact --verbose` | 1 |
| `CI_JSONOutput` | `--json-output` | 1 |
| `CI_DryRun` | `--dry-run` | 1 |
| `CI_Repeats` | `--repeats=2` | 1 |
| `CI_ConfigYAML` | `--config` | 1 |
| `CI_StressTest` | `--stress-test` | 1 |
| `CI_Shuffle` | `--shuffle` | 1 |
| **Total** | | **92** |

### Running locally

```bash
# All 92 CI checks
reframe -C reframe/settings.py -c reframe/ci_functional_checks.py -r -t ci

# Only GEMM precision tests
reframe -C reframe/settings.py -c reframe/ci_functional_checks.py -r -t gemm

# Only the precision matrix (all benchmarks together per dtype)
reframe -C reframe/settings.py -c reframe/ci_functional_checks.py -r -t precision-matrix

# Only output-mode tests (compact, JSON)
reframe -C reframe/settings.py -c reframe/ci_functional_checks.py -r -t output

# Only control-path tests (dry-run, repeats, shuffle, etc.)
reframe -C reframe/settings.py -c reframe/ci_functional_checks.py -r -t control
```

### GitHub Actions

Two workflows cover the full test pyramid:

| Workflow | File | Runner | What it tests |
|----------|------|--------|---------------|
| **Functional Tests** | `.github/workflows/gpu-functional.yml` | `ubuntu-latest` (no GPU) | pytest suite (~287 tests) — parsing, smoke, compact, syslog, telemetry, utilities |
| **GPU Functional Tests (ReFrame)** | `.github/workflows/gpu-reframe.yml` | `[self-hosted, gpu]` | ReFrame CI suite (92 tests) — every benchmark × precision on real GPU hardware |

The ReFrame workflow uses a matrix strategy with one leg per tag:

```
gemm | conv | fft | einsum | memory | heat | schrodinger | atomic | sparse | precision-matrix | full | output | control
```

To enable the ReFrame workflow, register a self-hosted runner with labels
`self-hosted` and `gpu` that has at least one CUDA GPU and PyTorch installed.
See [GitHub docs on self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners).

## CI/CD Integration

Example GitLab CI:
```yaml
benchmark:
  stage: test
  script:
    - module load cuda/12.4 python/3.11
    - pip install reframe-hpc
    - reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py -r
  artifacts:
    paths:
      - perflogs/
    expire_in: 30 days
  tags:
    - gpu
```

## Contributing

When adding new benchmarks to torch-hammer:
1. Add corresponding test class in `torch_hammer_checks.py`
2. Add a matching CI check in `ci_functional_checks.py` for every supported precision
3. Define `@performance_function` matching output regex
4. Add parameter variants for key test dimensions
5. Update this README

## License

Same as Torch Hammer (see root LICENSE file).
