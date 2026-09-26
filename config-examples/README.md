# Torch Hammer Configuration Examples

This directory contains example configuration files for running Torch Hammer benchmarks.

## Usage

Run benchmarks with a configuration file:
```bash
./torch-hammer.py --config config-examples/quick-test.yaml
```

Preview what a configuration will run:
```bash
./torch-hammer.py --config config-examples/stress-test.yaml --list-profiles
```

Override config settings via CLI:
```bash
./torch-hammer.py --config config-examples/quick-test.yaml --verbose --all-gpus
```

## Available Configurations

| File | Purpose | Duration |
|------|---------|----------|
| `quick-test.yaml` | Fast sanity check | ~2 minutes |
| `stress-test.yaml` | Comprehensive test suite | ~15-30 minutes |
| `platform-stress.yaml` | Full precision sweep across all benchmarks | ~30-45 minutes per GPU |

## Configuration File Format

Config files support two key naming styles:

### Short generic keys (used in quick-test.yaml, stress-test.yaml)
```yaml
benchmarks:
  - name: batched_gemm
    precision: float32         # Short key
    batch_count: 128           # Short key
    inner_loop: 50             # Short key
```

### Full argparse attribute names (used in platform-stress.yaml)
```yaml
benchmarks:
  - name: batched_gemm
    precision_gemm: float32    # Full attribute name
    batch_count_gemm: 128      # Full attribute name
    inner_loop_batched_gemm: 50  # Full attribute name
```

Both styles work in all config files. Short keys are checked first for backwards compatibility.

### Global settings
```yaml
profile: "Descriptive Name"
description: "What this config does"

global:
  warmup: 10              # Warmup iterations
  verbose: true           # Enable verbose output
  all_gpus: true          # Run on all GPUs
  cpu_affinity: false     # Boolean flags accept true or false
  stress_test: true       # Auto-size benchmarks to fill available memory
  duration: 60            # Run each benchmark for 60s (also accepted under runtime:)
  csv_output: results.csv # Compact CSV to a file (stdout unchanged); <hostname>_<timestamp> is inserted

runtime:
  temp_warn_C: 85.0       # Temperature warning threshold
  temp_critical_C: 92.0   # Temperature critical threshold
```

Any `store_true`/`store_false` CLI flag can be set under `global:` with `true` or `false`,
so `cpu_affinity: false` disables the default-on NUMA binding. Negated CLI spellings are
accepted too: `no_cpu_affinity: true` is the same as `cpu_affinity: false`. CLI flags always
win over the config file, in either spelling (`--no-cpu-affinity` beats `cpu_affinity: true`).

`duration`, `min_iterations` and `max_iterations` may live under either `global:` or `runtime:`.

When `stress_test` is on (from `global:` or `--stress-test`), stress sizing overrides the
explicit per-benchmark sizes (m/n/k, grid sizes, batch counts, ...) for every benchmark in the
`benchmarks:` list, exactly as it does for CLI-selected benchmarks. Precision, time steps,
patterns and inner-loop counts are left as written.

## Creating Custom Configurations

1. Copy an existing config as a starting point
2. Adjust parameters for your use case
3. Use `--stress-test` (or `stress_test: true` under `global:`) to auto-scale sizes based on GPU memory

## Available Benchmarks

| Name | Description | Key Parameters | Supported Precisions |
|------|-------------|----------------|---------------------|
| `batched_gemm` | Matrix multiplication | m, n, k, batch_count | All 6 |
| `convolution` | 2D convolution | height, width, channels, kernel_size | All 6 |
| `fft` | 3D FFT | nx, ny, nz | float16, float32, float64, complex64, complex128 |
| `einsum` | Attention pattern | heads, seq_len, d_model | All 6 |
| `memory_traffic` | Memory bandwidth | memory_size, memory_pattern | All 6 |
| `heat_equation` | Stencil computation | heat_grid_size, heat_time_steps | All 6 |
| `schrodinger` | Quantum simulation | schrodinger_grid_size | All 6 |
| `atomic_contention` | L2 cache stress | atomic_target_size | float16, bfloat16, float32, float64 |
| `sparse_mm` | Sparse matrix multiply | sparse_m, sparse_n, sparse_density | float16, bfloat16, float32, float64 |

## Precision Options

- `float16` - Half precision
- `bfloat16` - Brain floating point
- `float32` - Single precision (default)
- `float64` - Double precision
- `complex64` - Complex single
- `complex128` - Complex double

**Note:** FFT does not support `bfloat16`. Atomic contention and sparse MM do not support complex types.
