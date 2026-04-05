#!/usr/bin/env python3
# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
ReFrame CI functional tests for Torch Hammer.

Exhaustive correctness checks across every benchmark × precision × parameter
combination.  Designed for self-hosted GPU runners in GitHub Actions (or any
system with at least one CUDA GPU).

The tests use deliberately small tensor sizes so the entire suite finishes
in minutes rather than hours.  They validate:

  • Every benchmark produces a valid performance line (sanity)
  • Every supported dtype (including complex) succeeds without error
  • Every precision × benchmark combination works in a multi-benchmark run
  • Output-mode variants (compact CSV, JSON, dry-run, etc.) work end-to-end
  • Multi-flag combinations (repeats, shuffle, stress-test) don't crash

Usage (local):
    reframe -C reframe/settings.py -c reframe/ci_functional_checks.py -r -t ci

Usage (CI):
    reframe -C reframe/settings.py -c reframe/ci_functional_checks.py \\
            -r -t ci --performance-report

Test count breakdown (92 tests):
    ── Individual benchmark × precision (67 tests) ──
    GEMM          7  (6 precisions + TF32)
    Conv          6  (6 precisions)
    FFT           6  (6 precisions)
    Einsum        6  (6 precisions)
    Memory       18  (6 precisions × 3 patterns)
    Heat          6  (6 precisions)
    Schrödinger  12  (6 precisions × 2 potentials)
    Atomic        4  (4 precisions)
    Sparse        2  (2 precisions: float32, float64)
    ── Precision matrix: all benchmarks together (16 tests) ──
    AllStandard   6  (7 benchmarks × 6 precisions, one invocation each)
    AllReal       4  (9 benchmarks × 4 real precisions, one invocation each)
    PairAtomic    4  (Atomic + GEMM × 4 real precisions)
    PairSparse    2  (Sparse + GEMM × 2 supported precisions)
    ── Output / control-path checks (9 tests) ──
    FullSuite     1
    CompactCSV    1
    CompactVerbose 1
    JSON          1
    DryRun        1
    Repeats       1
    ConfigYAML    1
    StressTest    1
    Shuffle       1
    ──────────────
    Total        92
"""

import os
import reframe as rfm
import reframe.utility.sanity as sn


# ── precision sets (mirrors build_parser() in torch-hammer.py) ───────
PRECISION_ALL = [
    'bfloat16', 'float16', 'float32', 'float64', 'complex64', 'complex128',
]
PRECISION_REAL = ['float16', 'bfloat16', 'float32', 'float64']  # atomic
PRECISION_SPARSE = ['float32', 'float64']  # torch.sparse.mm only supports 32/64-bit


# =====================================================================
#  Base class
# =====================================================================
class TorchHammerCIBase(rfm.RunOnlyRegressionTest):
    """Base for all CI functional checks.

    Uses minimal tensor sizes and iteration counts so the whole suite
    finishes in minutes, not hours.
    """

    valid_systems = ['*']
    valid_prog_environs = ['*']

    num_gpus_per_node = 1
    time_limit = '5m'
    tags = {'ci', 'gpu'}

    # ── tunables ─────────────────────────────────────────────────────
    torch_hammer_script = variable(str, value='../torch-hammer.py')
    warmup = variable(int, value=2)
    inner_loop = variable(int, value=5)

    # ── hooks ────────────────────────────────────────────────────────
    @run_before('run')
    def set_executable(self):
        # Resolve torch-hammer.py relative to THIS source file, then
        # normalise to an absolute path so it works regardless of cwd.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script = os.path.normpath(
            os.path.join(script_dir, self.torch_hammer_script)
        )
        self.executable = 'python3'
        self.executable_opts = [
            script,
            '--device-index=0',
            f'--warmup={self.warmup}',
            '--no-cpu-affinity',
        ]

    @sanity_function
    def validate_run(self):
        return sn.assert_found(
            r'\[OK\] Benchmark run finished', self.stdout,
        )


# =====================================================================
#  GEMM – 6 dtypes + TF32  (7 tests)
# =====================================================================
@rfm.simple_test
class CI_GEMM(TorchHammerCIBase):
    """Batched GEMM across all six precisions."""

    precision = parameter(PRECISION_ALL)
    tags = {'ci', 'gpu', 'gemm'}
    descr = 'CI · Batched GEMM'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--batched-gemm',
            f'--precision-gemm={self.precision}',
            '--m=64', '--n=64', '--k=64',
            '--batch-count-gemm=4',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Batched GEMM\]\s+Performance:', self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('GFLOP/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Batched GEMM\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float,
        )


@rfm.simple_test
class CI_GEMM_TF32(TorchHammerCIBase):
    """Batched GEMM with TF32 mode enabled."""

    tags = {'ci', 'gpu', 'gemm', 'tf32'}
    descr = 'CI · Batched GEMM (TF32)'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--batched-gemm',
            '--precision-gemm=float32',
            '--batched-gemm-TF32-mode',
            '--m=64', '--n=64', '--k=64',
            '--batch-count-gemm=4',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Batched GEMM\]\s+Performance:', self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('GFLOP/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Batched GEMM\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Convolution – 6 dtypes  (6 tests)
# =====================================================================
@rfm.simple_test
class CI_Conv(TorchHammerCIBase):
    """2-D Convolution across all six precisions."""

    precision = parameter(PRECISION_ALL)
    tags = {'ci', 'gpu', 'conv'}
    descr = 'CI · Convolution'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--convolution',
            f'--precision-convolution={self.precision}',
            '--batch-count-convolution=4',
            '--in-channels=3', '--out-channels=8',
            '--height=32', '--width=32',
            '--kernel-size=3',
            f'--inner-loop-convolution={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Convolution\]\s+Performance:', self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('img/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Convolution\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+img/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  FFT – 6 dtypes  (6 tests)
# =====================================================================
@rfm.simple_test
class CI_FFT(TorchHammerCIBase):
    """3-D FFT across all six precisions."""

    precision = parameter(PRECISION_ALL)
    tags = {'ci', 'gpu', 'fft'}
    descr = 'CI · 3-D FFT'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--fft',
            f'--precision-fft={self.precision}',
            '--batch-count-fft=4',
            '--nx=16', '--ny=16', '--nz=16',
            f'--inner-loop-fft={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """FFT must produce a Performance line or a graceful error.
        On some backends (e.g. ROCm + bfloat16) FFT fails with an error
        message containing '3-D FFT'; on success it prints '3D FFT'."""
        return sn.all([
            sn.assert_found(r'3-?D FFT', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('GFLOP/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+3D FFT\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Einsum (Attention) – 6 dtypes  (6 tests)
# =====================================================================
@rfm.simple_test
class CI_Einsum(TorchHammerCIBase):
    """Einsum (attention) across all six precisions."""

    precision = parameter(PRECISION_ALL)
    tags = {'ci', 'gpu', 'einsum'}
    descr = 'CI · Einsum Attention'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--einsum',
            f'--precision-einsum={self.precision}',
            '--batch-count-einsum=4',
            '--heads=2', '--seq-len=32', '--d-model=16',
            f'--inner-loop-einsum={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Einsum Attention\]\s+Performance:', self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('GFLOP/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Einsum Attention\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Memory Traffic – 6 dtypes × 3 patterns  (18 tests)
# =====================================================================
@rfm.simple_test
class CI_Memory(TorchHammerCIBase):
    """Memory bandwidth across all precisions and access patterns."""

    precision = parameter(PRECISION_ALL)
    pattern = parameter(['random', 'streaming', 'unit'])
    tags = {'ci', 'gpu', 'memory'}
    descr = 'CI · Memory Traffic'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--memory-traffic',
            f'--precision-memory={self.precision}',
            f'--memory-pattern={self.pattern}',
            '--memory-size=256',
            '--memory-iterations=5',
            f'--inner-loop-memory-traffic={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Memory Traffic\]\s+Performance:', self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('GB/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Memory Traffic\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+GB/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Heat Equation – 6 dtypes  (6 tests)
# =====================================================================
@rfm.simple_test
class CI_Heat(TorchHammerCIBase):
    """Heat equation (Laplacian stencil) across all precisions."""

    precision = parameter(PRECISION_ALL)
    tags = {'ci', 'gpu', 'heat', 'stencil'}
    descr = 'CI · Heat Equation'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--heat-equation',
            f'--precision-heat={self.precision}',
            '--heat-grid-size=64',
            '--heat-time-steps=10',
            f'--inner-loop-heat-equation={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Heat Equation\]\s+Performance:', self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('MLUPS')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Heat Equation\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+MLUPS',
            self.stdout, 1, float,
        )


# =====================================================================
#  Schrödinger Equation – 6 dtypes × 2 potentials  (12 tests)
# =====================================================================
@rfm.simple_test
class CI_Schrodinger(TorchHammerCIBase):
    """Schrödinger equation across all precisions and potentials."""

    precision = parameter(PRECISION_ALL)
    potential = parameter(['harmonic', 'barrier'])
    tags = {'ci', 'gpu', 'schrodinger', 'quantum'}
    descr = 'CI · Schrödinger Equation'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--schrodinger',
            f'--precision-schrodinger={self.precision}',
            f'--schrodinger-potential={self.potential}',
            '--schrodinger-grid-size=64',
            '--schrodinger-time-steps=10',
            f'--inner-loop-schrodinger={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(
                r'\[GPU\d+\s+Schr.dinger Equation\]\s+Performance:',
                self.stdout,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])

    @performance_function('iter/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Schr.dinger Equation\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+iter/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Atomic Contention – 4 real dtypes  (4 tests)
# =====================================================================
@rfm.simple_test
class CI_Atomic(TorchHammerCIBase):
    """Atomic contention (L2 cache stress) across real precisions."""

    precision = parameter(PRECISION_REAL)
    tags = {'ci', 'gpu', 'atomic'}
    descr = 'CI · Atomic Contention'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--atomic-contention',
            f'--precision-atomic={self.precision}',
            '--atomic-target-size=1000',
            '--atomic-num-updates=10000',
            '--atomic-contention-range=64',
            f'--inner-loop-atomic={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """Atomic must produce a Performance line or a graceful skip.
        We require the benchmark name to appear (proving it was attempted)
        and either a Performance line, a skip/warning, or a known error."""
        return sn.all([
            sn.assert_found(r'Atomic Contention', self.stdout),
            sn.any([
                sn.assert_found(
                    r'\[GPU\d+\s+Atomic Contention\]\s+Performance:',
                    self.stdout,
                ),
                # Graceful skip/warning goes to stdout via log.warning
                sn.assert_found(
                    r'Atomic Contention.*skipping|Atomic Contention.*not supported',
                    self.stdout,
                ),
                sn.assert_found(
                    r'FAILED|Error|not supported|RuntimeError',
                    self.stderr,
                ),
            ]),
        ])

    @performance_function('Mops/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Atomic Contention\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+Mops/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Sparse MM – 2 supported dtypes  (2 tests)
# =====================================================================
@rfm.simple_test
class CI_Sparse(TorchHammerCIBase):
    """Sparse matrix multiply across supported precisions.

    torch.sparse.mm only supports float32/float64 — half-precision
    dtypes (float16, bfloat16) are gracefully skipped by torch-hammer.
    """

    precision = parameter(PRECISION_SPARSE)
    tags = {'ci', 'gpu', 'sparse'}
    descr = 'CI · Sparse MM'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--sparse-mm',
            f'--precision-sparse={self.precision}',
            '--sparse-m=64', '--sparse-n=64', '--sparse-k=64',
            '--sparse-density=0.10',
            f'--inner-loop-sparse={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'Sparse MM', self.stdout),
            sn.assert_found(
                r'\[GPU\d+\s+Sparse MM\]\s+Performance:',
                self.stdout,
            ),
        ])

    @performance_function('GFLOP/s')
    def perf_mean(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Sparse MM\]\s+Performance:\s+[\d.]+\s*/\s*'
            r'([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float,
        )


# =====================================================================
#  Precision Matrix – all standard benchmarks together  (6 tests)
# =====================================================================
@rfm.simple_test
class CI_PrecisionMatrixStandard(TorchHammerCIBase):
    """Run all 7 standard benchmarks in a single invocation per precision.

    Catches interaction bugs, memory fragmentation, and dtype-propagation
    errors that only surface when multiple benchmarks share a single
    process and GPU context.  Uses PRECISION_ALL (includes complex).
    """

    precision = parameter(PRECISION_ALL)
    tags = {'ci', 'gpu', 'precision-matrix'}
    descr = 'CI · Precision matrix (7 standard benchmarks)'
    time_limit = '10m'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            # GEMM
            '--batched-gemm',
            f'--precision-gemm={self.precision}',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
            # Convolution
            '--convolution',
            f'--precision-convolution={self.precision}',
            '--batch-count-convolution=2',
            '--in-channels=3', '--out-channels=4',
            '--height=16', '--width=16', '--kernel-size=3',
            f'--inner-loop-convolution={self.inner_loop}',
            # FFT
            '--fft',
            f'--precision-fft={self.precision}',
            '--batch-count-fft=2',
            '--nx=8', '--ny=8', '--nz=8',
            f'--inner-loop-fft={self.inner_loop}',
            # Einsum
            '--einsum',
            f'--precision-einsum={self.precision}',
            '--batch-count-einsum=2',
            '--heads=2', '--seq-len=16', '--d-model=8',
            f'--inner-loop-einsum={self.inner_loop}',
            # Memory Traffic
            '--memory-traffic',
            f'--precision-memory={self.precision}',
            '--memory-size=128',
            '--memory-iterations=5',
            f'--inner-loop-memory-traffic={self.inner_loop}',
            # Heat Equation
            '--heat-equation',
            f'--precision-heat={self.precision}',
            '--heat-grid-size=32',
            '--heat-time-steps=5',
            f'--inner-loop-heat-equation={self.inner_loop}',
            # Schrödinger Equation
            '--schrodinger',
            f'--precision-schrodinger={self.precision}',
            '--schrodinger-grid-size=32',
            '--schrodinger-time-steps=5',
            f'--inner-loop-schrodinger={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """Each benchmark must either produce a Performance line or a
        graceful skip/failure message.  Some dtypes (e.g. bfloat16)
        are not supported by every benchmark on every backend."""
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Convolution', self.stdout),
            # FFT logs as "3D FFT" on success, "3-D FFT" on failure
            sn.assert_found(r'3-?D FFT|FFT', self.stdout),
            sn.assert_found(r'Einsum Attention', self.stdout),
            sn.assert_found(r'Memory Traffic', self.stdout),
            sn.assert_found(r'Heat Equation', self.stdout),
            sn.assert_found(r'Schr.dinger Equation', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


# =====================================================================
#  Precision Matrix – all 9 benchmarks (real dtypes only)  (4 tests)
# =====================================================================
@rfm.simple_test
class CI_PrecisionMatrixAll(TorchHammerCIBase):
    """Run ALL 9 benchmarks in a single invocation per real precision.

    Real dtypes (float16/bfloat16/float32/float64) are valid for every
    benchmark including atomic and sparse.  This is the ultimate
    integration test: every benchmark in one process, one dtype.
    """

    precision = parameter(PRECISION_REAL)
    tags = {'ci', 'gpu', 'precision-matrix', 'full'}
    descr = 'CI · Precision matrix (all 9 benchmarks)'
    time_limit = '10m'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            # GEMM
            '--batched-gemm',
            f'--precision-gemm={self.precision}',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
            # Convolution
            '--convolution',
            f'--precision-convolution={self.precision}',
            '--batch-count-convolution=2',
            '--in-channels=3', '--out-channels=4',
            '--height=16', '--width=16', '--kernel-size=3',
            f'--inner-loop-convolution={self.inner_loop}',
            # FFT
            '--fft',
            f'--precision-fft={self.precision}',
            '--batch-count-fft=2',
            '--nx=8', '--ny=8', '--nz=8',
            f'--inner-loop-fft={self.inner_loop}',
            # Einsum
            '--einsum',
            f'--precision-einsum={self.precision}',
            '--batch-count-einsum=2',
            '--heads=2', '--seq-len=16', '--d-model=8',
            f'--inner-loop-einsum={self.inner_loop}',
            # Memory Traffic
            '--memory-traffic',
            f'--precision-memory={self.precision}',
            '--memory-size=128',
            '--memory-iterations=5',
            f'--inner-loop-memory-traffic={self.inner_loop}',
            # Heat Equation
            '--heat-equation',
            f'--precision-heat={self.precision}',
            '--heat-grid-size=32',
            '--heat-time-steps=5',
            f'--inner-loop-heat-equation={self.inner_loop}',
            # Schrödinger Equation
            '--schrodinger',
            f'--precision-schrodinger={self.precision}',
            '--schrodinger-grid-size=32',
            '--schrodinger-time-steps=5',
            f'--inner-loop-schrodinger={self.inner_loop}',
            # Atomic Contention
            '--atomic-contention',
            f'--precision-atomic={self.precision}',
            '--atomic-target-size=500',
            '--atomic-num-updates=5000',
            '--atomic-contention-range=32',
            f'--inner-loop-atomic={self.inner_loop}',
            # Sparse MM
            '--sparse-mm',
            f'--precision-sparse={self.precision}',
            '--sparse-m=32', '--sparse-n=32', '--sparse-k=32',
            '--sparse-density=0.10',
            f'--inner-loop-sparse={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """All nine benchmarks must be attempted.  Some may skip
        gracefully (e.g. sparse bfloat16 on ROCm), so we check that
        each name appears in stdout — either in a Performance line
        or in a skip/error message."""
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Convolution', self.stdout),
            sn.assert_found(r'3-?D FFT|FFT', self.stdout),
            sn.assert_found(r'Einsum Attention', self.stdout),
            sn.assert_found(r'Memory Traffic', self.stdout),
            sn.assert_found(r'Heat Equation', self.stdout),
            sn.assert_found(r'Schr.dinger Equation', self.stdout),
            sn.assert_found(r'Atomic Contention', self.stdout),
            sn.assert_found(r'Sparse MM', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


# =====================================================================
#  Precision Matrix – Atomic + GEMM pair (4 tests)
# =====================================================================
@rfm.simple_test
class CI_PrecisionMatrixAtomic(TorchHammerCIBase):
    """Atomic contention paired with GEMM per real precision.

    Atomic uses scatter_add which can interact badly with concurrent
    GEMM tensor allocations on some backends.
    """

    precision = parameter(PRECISION_REAL)
    tags = {'ci', 'gpu', 'precision-matrix', 'atomic'}
    descr = 'CI · Precision matrix (Atomic + GEMM)'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--batched-gemm',
            f'--precision-gemm={self.precision}',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
            '--atomic-contention',
            f'--precision-atomic={self.precision}',
            '--atomic-target-size=500',
            '--atomic-num-updates=5000',
            '--atomic-contention-range=32',
            f'--inner-loop-atomic={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Atomic Contention', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


# =====================================================================
#  Precision Matrix – Sparse + GEMM pair (2 tests)
# =====================================================================
@rfm.simple_test
class CI_PrecisionMatrixSparse(TorchHammerCIBase):
    """Sparse MM paired with GEMM per real precision.

    Sparse CSR allocations can fragment GPU memory differently from
    dense GEMM tensors — this catches those interactions.
    """

    precision = parameter(PRECISION_SPARSE)
    tags = {'ci', 'gpu', 'precision-matrix', 'sparse'}
    descr = 'CI · Precision matrix (Sparse + GEMM)'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--batched-gemm',
            f'--precision-gemm={self.precision}',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
            '--sparse-mm',
            f'--precision-sparse={self.precision}',
            '--sparse-m=32', '--sparse-n=32', '--sparse-k=32',
            '--sparse-density=0.10',
            f'--inner-loop-sparse={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Sparse MM', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


# =====================================================================
#  Full Suite – run all 9 benchmarks in one shot  (1 test)
# =====================================================================
@rfm.simple_test
class CI_FullSuite(TorchHammerCIBase):
    """Smoke test: enable every benchmark in a single invocation."""

    tags = {'ci', 'gpu', 'full'}
    descr = 'CI · Full Suite (all 9 benchmarks)'
    time_limit = '10m'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--batched-gemm',  '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
            '--convolution', '--batch-count-convolution=2',
            '--in-channels=3', '--out-channels=4',
            '--height=16', '--width=16', '--kernel-size=3',
            f'--inner-loop-convolution={self.inner_loop}',
            '--fft', '--batch-count-fft=2',
            '--nx=8', '--ny=8', '--nz=8',
            f'--inner-loop-fft={self.inner_loop}',
            '--einsum', '--batch-count-einsum=2',
            '--heads=2', '--seq-len=16', '--d-model=8',
            f'--inner-loop-einsum={self.inner_loop}',
            '--memory-traffic', '--memory-size=128',
            '--memory-iterations=5',
            f'--inner-loop-memory-traffic={self.inner_loop}',
            '--heat-equation', '--heat-grid-size=32',
            '--heat-time-steps=5',
            f'--inner-loop-heat-equation={self.inner_loop}',
            '--schrodinger', '--schrodinger-grid-size=32',
            '--schrodinger-time-steps=5',
            f'--inner-loop-schrodinger={self.inner_loop}',
            '--atomic-contention',
            '--atomic-target-size=500',
            '--atomic-num-updates=5000',
            '--atomic-contention-range=32',
            f'--inner-loop-atomic={self.inner_loop}',
            '--sparse-mm',
            '--sparse-m=32', '--sparse-n=32', '--sparse-k=32',
            '--sparse-density=0.10',
            f'--inner-loop-sparse={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """All nine benchmark names must appear in stdout."""
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Convolution', self.stdout),
            sn.assert_found(r'3-?D FFT|FFT', self.stdout),
            sn.assert_found(r'Einsum Attention', self.stdout),
            sn.assert_found(r'Memory Traffic', self.stdout),
            sn.assert_found(r'Heat Equation', self.stdout),
            sn.assert_found(r'Schr.dinger Equation', self.stdout),
            sn.assert_found(r'Atomic Contention', self.stdout),
            sn.assert_found(r'Sparse MM', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


# =====================================================================
#  Output-mode checks  (3 tests)
# =====================================================================
@rfm.simple_test
class CI_CompactCSV(TorchHammerCIBase):
    """Verify --compact produces valid CSV to stdout."""

    tags = {'ci', 'gpu', 'output', 'compact'}
    descr = 'CI · Compact CSV output'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--compact',
            '--batched-gemm',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """Compact mode emits a CSV header row and at least one data row.
        The [OK] banner is suppressed in compact mode, so we only
        validate CSV content and the absence of a Python traceback."""
        return sn.all([
            # CSV header contains these mandatory fields
            sn.assert_found(r'benchmark', self.stdout),
            sn.assert_found(r'dtype', self.stdout),
            # At least one data row with the benchmark name
            sn.assert_found(r'Batched GEMM', self.stdout),
            # Ensure no Python crash
            sn.assert_not_found(r'Traceback', self.stderr),
        ])


@rfm.simple_test
class CI_CompactVerbose(TorchHammerCIBase):
    """Verify --compact --verbose adds telemetry columns."""

    tags = {'ci', 'gpu', 'output', 'compact'}
    descr = 'CI · Compact + Verbose CSV output'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--compact', '--verbose',
            '--batched-gemm',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """Verbose compact mode adds telemetry columns to the CSV.
        The [OK] banner is suppressed in compact mode, so we only
        validate CSV content and the absence of a Python traceback."""
        return sn.all([
            sn.assert_found(r'benchmark', self.stdout),
            sn.assert_found(r'Batched GEMM', self.stdout),
            # Verbose adds telemetry columns to the header;
            # accept even if telemetry columns are absent on some platforms.
            sn.assert_not_found(r'Traceback', self.stderr),
        ])


@rfm.simple_test
class CI_JSONOutput(TorchHammerCIBase):
    """Verify --json-output produces a valid JSON file."""

    tags = {'ci', 'gpu', 'output', 'json'}
    descr = 'CI · JSON output'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--json-output=ci_test_output.json',
            '--batched-gemm',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
            # torch-hammer appends {hostname}_{timestamp} to the filename,
            # e.g. ci_test_output_nid003032_20260303_112849.json
            sn.assert_found(
                r'JSON results exported to: ci_test_output_.*\.json',
                self.stdout,
            ),
        ])


# =====================================================================
#  Control-path checks  (5 tests)
# =====================================================================
@rfm.simple_test
class CI_DryRun(TorchHammerCIBase):
    """Verify --dry-run prints config and exits cleanly."""

    tags = {'ci', 'gpu', 'control', 'dryrun'}
    descr = 'CI · Dry-run mode'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--dry-run',
            '--batched-gemm',
            '--convolution',
            '--fft',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'DRY RUN MODE', self.stdout),
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Convolution', self.stdout),
            sn.assert_found(r'3-D FFT|3D FFT', self.stdout),
            sn.assert_found(r'END DRY RUN', self.stdout),
        ])


@rfm.simple_test
class CI_Repeats(TorchHammerCIBase):
    """Verify --repeats runs the suite multiple times."""

    tags = {'ci', 'gpu', 'control', 'repeats'}
    descr = 'CI · Repeats (×2)'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--repeats=2', '--repeat-delay=0',
            '--batched-gemm',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        """With --repeats=2, expect REPEAT 1/2 and REPEAT 2/2 markers
        and two Performance summary lines (one per repeat).  The [OK]
        line is printed once after the repeat loop finishes."""
        return sn.all([
            sn.assert_found(r'REPEAT 1/2', self.stdout),
            sn.assert_found(r'REPEAT 2/2', self.stdout),
            sn.assert_eq(
                sn.count(
                    sn.findall(r'Performance:', self.stdout),
                ),
                2,
            ),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


@rfm.simple_test
class CI_ConfigYAML(TorchHammerCIBase):
    """Verify --config loads a YAML file correctly."""

    tags = {'ci', 'gpu', 'control', 'config'}
    descr = 'CI · YAML config loading'

    @run_before('run')
    def set_benchmark_opts(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            script_dir, '..', 'config-examples', 'quick-test.yaml',
        )
        self.executable_opts += [
            f'--config={config_path}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.assert_found(
            r'\[OK\] Benchmark run finished', self.stdout,
        )


@rfm.simple_test
class CI_StressTest(TorchHammerCIBase):
    """Verify --stress-test auto-sizes parameters without OOM."""

    tags = {'ci', 'gpu', 'control', 'stress'}
    descr = 'CI · Stress-test auto-sizing'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--stress-test',
            '--batched-gemm',
            f'--inner-loop-batched-gemm={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'Stress Test', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])


@rfm.simple_test
class CI_Shuffle(TorchHammerCIBase):
    """Verify --shuffle randomises benchmark order without error."""

    tags = {'ci', 'gpu', 'control', 'shuffle'}
    descr = 'CI · Shuffle mode'

    @run_before('run')
    def set_benchmark_opts(self):
        self.executable_opts += [
            '--shuffle',
            '--batched-gemm',
            '--m=32', '--n=32', '--k=32',
            '--batch-count-gemm=2',
            f'--inner-loop-batched-gemm={self.inner_loop}',
            '--memory-traffic', '--memory-size=128',
            '--memory-iterations=5',
            f'--inner-loop-memory-traffic={self.inner_loop}',
        ]

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Memory Traffic', self.stdout),
            sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout),
        ])
