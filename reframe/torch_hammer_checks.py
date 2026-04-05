#!/usr/bin/env python3
"""
ReFrame regression tests for Torch Hammer GPU benchmarks.

This module provides ReFrame-compatible tests wrapping torch-hammer.py
for HPC system validation and performance regression testing.

Usage:
    reframe -c reframe/torch_hammer_checks.py -r

Configuration:
    Set valid_systems and valid_prog_environs in your ReFrame config
    to match your HPC system partitions and programming environments.
"""

import os
import reframe as rfm
import reframe.utility.sanity as sn


class TorchHammerBase(rfm.RunOnlyRegressionTest):
    """Base class for all Torch Hammer benchmarks."""
    
    # Override these in your site config
    valid_systems = ['*']
    valid_prog_environs = ['*']
    
    # Common settings
    num_gpus_per_node = 1
    time_limit = '30m'
    
    # Torch Hammer script location (relative to test file)
    torch_hammer_script = variable(str, value='../torch-hammer.py')
    
    # Common benchmark parameters
    duration = variable(int, value=60)  # seconds
    warmup = variable(int, value=10)
    
    # Device selection
    device_index = variable(int, value=0)
    
    @run_before('run')
    def set_executable(self):
        """Set up the torch-hammer executable and common options."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.executable = f'python3 {os.path.join(script_dir, self.torch_hammer_script)}'
        
        # Common options
        self.executable_opts = [
            f'--device-index={self.device_index}',
            f'--warmup={self.warmup}',
        ]
        
        # Add duration if specified
        if self.duration > 0:
            self.executable_opts.append(f'--duration={self.duration}')
    
    @sanity_function
    def validate_run(self):
        """Validate that the benchmark completed successfully."""
        return sn.assert_found(r'\[OK\] Benchmark run finished', self.stdout)


# ============================================================================
# BATCHED GEMM BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerGEMM(TorchHammerBase):
    """Batched GEMM (Matrix Multiply) benchmark."""
    
    descr = 'Torch Hammer Batched GEMM Benchmark'
    tags = {'gpu', 'compute', 'gemm'}
    
    # GEMM-specific parameters
    precision = parameter(['float32', 'float16', 'bfloat16', 'float64'])
    matrix_size = parameter([4096, 8192, 16384])
    batch_count = variable(int, value=1)
    tf32_mode = variable(bool, value=False)
    
    @run_before('run')
    def set_gemm_options(self):
        """Add GEMM-specific options."""
        self.executable_opts.extend([
            '--batched-gemm',
            f'--precision-gemm={self.precision}',
            f'--m={self.matrix_size}',
            f'--n={self.matrix_size}',
            f'--k={self.matrix_size}',
            f'--batch-count-gemm={self.batch_count}',
        ])
        if self.tf32_mode:
            self.executable_opts.append('--batched-gemm-TF32-mode')
    
    @performance_function('GFLOP/s')
    def gemm_gflops(self):
        """Extract GEMM mean performance in GFLOP/s.
        
        Output format: [GPU{id} Batched GEMM] Performance: {min} / {mean} / {max} GFLOP/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Batched GEMM\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )
    
    @performance_function('GFLOP/s')
    def gemm_gflops_min(self):
        """Extract minimum GEMM performance."""
        return sn.extractsingle(
            r'\[GPU\d+\s+Batched GEMM\]\s+Performance:\s+([\d.]+)\s*/\s*[\d.]+\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )
    
    @performance_function('GFLOP/s')
    def gemm_gflops_max(self):
        """Extract maximum GEMM performance."""
        return sn.extractsingle(
            r'\[GPU\d+\s+Batched GEMM\]\s+Performance:\s+[\d.]+\s*/\s*[\d.]+\s*/\s*([\d.]+)\s+GFLOP/s',
            self.stdout, 1, float
        )


# ============================================================================
# CONVOLUTION BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerConvolution(TorchHammerBase):
    """2D Convolution benchmark."""
    
    descr = 'Torch Hammer 2D Convolution Benchmark'
    tags = {'gpu', 'compute', 'conv', 'dl'}
    
    precision = parameter(['float32', 'float16'])
    batch_size = variable(int, value=32)
    in_channels = variable(int, value=64)
    out_channels = variable(int, value=128)
    height = variable(int, value=224)
    width = variable(int, value=224)
    kernel_size = variable(int, value=3)
    
    @run_before('run')
    def set_conv_options(self):
        """Add convolution-specific options."""
        self.executable_opts.extend([
            '--convolution',
            f'--precision-convolution={self.precision}',
            f'--batch-count-convolution={self.batch_size}',
            f'--in-channels={self.in_channels}',
            f'--out-channels={self.out_channels}',
            f'--height={self.height}',
            f'--width={self.width}',
            f'--kernel-size={self.kernel_size}',
        ])
    
    @performance_function('img/s')
    def conv_throughput(self):
        """Extract convolution throughput in images/second.
        
        Output format: [GPU{id} Convolution] Performance: {min} / {mean} / {max} img/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Convolution\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+img/s',
            self.stdout, 1, float
        )


# ============================================================================
# FFT BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerFFT(TorchHammerBase):
    """3D FFT benchmark."""
    
    descr = 'Torch Hammer 3D FFT Benchmark'
    tags = {'gpu', 'compute', 'fft', 'spectral'}
    
    precision = parameter(['float32', 'complex64'])
    fft_size = parameter([256, 512])
    batch_count = variable(int, value=10)
    
    @run_before('run')
    def set_fft_options(self):
        """Add FFT-specific options."""
        self.executable_opts.extend([
            '--fft',
            f'--precision-fft={self.precision}',
            f'--batch-count-fft={self.batch_count}',
            f'--nx={self.fft_size}',
            f'--ny={self.fft_size}',
            f'--nz={self.fft_size}',
        ])
    
    @performance_function('GFLOP/s')
    def fft_gflops(self):
        """Extract FFT performance in GFLOP/s.
        
        Output format: [GPU{id} 3D FFT] Performance: {min} / {mean} / {max} GFLOP/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+3D FFT\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )


# ============================================================================
# EINSUM (ATTENTION) BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerEinsum(TorchHammerBase):
    """Einsum (Attention-style) benchmark."""
    
    descr = 'Torch Hammer Einsum Attention Benchmark'
    tags = {'gpu', 'compute', 'einsum', 'attention', 'transformer'}
    
    precision = parameter(['float32', 'float16', 'bfloat16'])
    batch_size = variable(int, value=8)
    num_heads = variable(int, value=12)
    seq_len = variable(int, value=512)
    d_model = variable(int, value=64)
    
    @run_before('run')
    def set_einsum_options(self):
        """Add einsum-specific options."""
        self.executable_opts.extend([
            '--einsum',
            f'--precision-einsum={self.precision}',
            f'--batch-count-einsum={self.batch_size}',
            f'--heads={self.num_heads}',
            f'--seq-len={self.seq_len}',
            f'--d-model={self.d_model}',
        ])
    
    @performance_function('GFLOP/s')
    def einsum_gflops(self):
        """Extract einsum performance in GFLOP/s.
        
        Output format: [GPU{id} Einsum Attention] Performance: {min} / {mean} / {max} GFLOP/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Einsum Attention\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )


# ============================================================================
# MEMORY TRAFFIC BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerMemory(TorchHammerBase):
    """Memory bandwidth benchmark."""
    
    descr = 'Torch Hammer Memory Traffic Benchmark'
    tags = {'gpu', 'memory', 'bandwidth'}
    
    precision = parameter(['float32', 'float64'])
    memory_pattern = parameter(['random', 'streaming', 'unit'])
    memory_size = variable(int, value=100_000_000)
    
    @run_before('run')
    def set_memory_options(self):
        """Add memory-specific options."""
        self.executable_opts.extend([
            '--memory-traffic',
            f'--precision-memory={self.precision}',
            f'--memory-size={self.memory_size}',
            f'--memory-pattern={self.memory_pattern}',
        ])
    
    @performance_function('GB/s')
    def memory_bandwidth(self):
        """Extract memory bandwidth in GB/s.
        
        Output format: [GPU{id} Memory Traffic] Performance: {min} / {mean} / {max} GB/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Memory Traffic\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GB/s',
            self.stdout, 1, float
        )


# ============================================================================
# HEAT EQUATION (STENCIL) BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerHeat(TorchHammerBase):
    """Heat equation (Laplacian stencil) benchmark."""
    
    descr = 'Torch Hammer Heat Equation Stencil Benchmark'
    tags = {'gpu', 'compute', 'stencil', 'pde'}
    
    precision = parameter(['float32', 'float64'])
    grid_size = variable(int, value=4096)
    time_steps = variable(int, value=1000)
    
    @run_before('run')
    def set_heat_options(self):
        """Add heat equation-specific options."""
        self.executable_opts.extend([
            '--heat-equation',
            f'--precision-heat={self.precision}',
            f'--heat-grid-size={self.grid_size}',
            f'--heat-time-steps={self.time_steps}',
        ])
    
    @performance_function('MLUPS')
    def heat_mlups(self):
        """Extract heat equation performance in MLUPS (million lattice updates/s).
        
        Output format: [GPU{id} Heat Equation] Performance: {min} / {mean} / {max} MLUPS
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Heat Equation\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+MLUPS',
            self.stdout, 1, float
        )


# ============================================================================
# SCHRÖDINGER EQUATION BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerSchrodinger(TorchHammerBase):
    """Schrödinger equation (quantum) benchmark."""
    
    descr = 'Torch Hammer Schrödinger Equation Benchmark'
    tags = {'gpu', 'compute', 'quantum', 'pde', 'complex'}
    
    precision = parameter(['complex64', 'complex128'])
    grid_size = variable(int, value=65536)
    time_steps = variable(int, value=10000)
    potential = variable(str, value='harmonic')
    
    @run_before('run')
    def set_schrodinger_options(self):
        """Add Schrödinger-specific options."""
        self.executable_opts.extend([
            '--schrodinger',
            f'--precision-schrodinger={self.precision}',
            f'--schrodinger-grid-size={self.grid_size}',
            f'--schrodinger-time-steps={self.time_steps}',
            f'--schrodinger-potential={self.potential}',
        ])
    
    @performance_function('iter/s')
    def schrodinger_iters(self):
        """Extract Schrödinger performance in iterations/second.
        
        Output format: [GPU{id} Schrödinger Equation] Performance: {min} / {mean} / {max} iter/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Schr.dinger Equation\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+iter/s',
            self.stdout, 1, float
        )


# ============================================================================
# ATOMIC CONTENTION BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerAtomic(TorchHammerBase):
    """Atomic operations contention benchmark."""
    
    descr = 'Torch Hammer Atomic Contention Benchmark'
    tags = {'gpu', 'memory', 'atomic', 'contention'}
    
    precision = parameter(['float32', 'float64'])
    target_size = variable(int, value=1_000_000)
    num_updates = variable(int, value=10_000_000)
    contention_range = variable(int, value=1024)
    
    @run_before('run')
    def set_atomic_options(self):
        """Add atomic contention-specific options."""
        self.executable_opts.extend([
            '--atomic-contention',
            f'--precision-atomic={self.precision}',
            f'--atomic-target-size={self.target_size}',
            f'--atomic-num-updates={self.num_updates}',
            f'--atomic-contention-range={self.contention_range}',
        ])
    
    @performance_function('Mops/s')
    def atomic_ops(self):
        """Extract atomic operations performance.
        
        Output format: [GPU{id} Atomic Contention] Performance: {min} / {mean} / {max} Mops/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Atomic Contention\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+Mops/s',
            self.stdout, 1, float
        )


# ============================================================================
# SPARSE MM BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerSparse(TorchHammerBase):
    """Sparse matrix multiplication benchmark."""
    
    descr = 'Torch Hammer Sparse Matrix Multiplication Benchmark'
    tags = {'gpu', 'compute', 'sparse', 'spgemm'}
    
    precision = parameter(['float32', 'float64'])
    sparse_m = variable(int, value=8192)
    sparse_n = variable(int, value=8192)
    sparse_k = variable(int, value=8192)
    density = variable(float, value=0.01)
    
    @run_before('run')
    def set_sparse_options(self):
        """Add sparse MM-specific options."""
        self.executable_opts.extend([
            '--sparse-mm',
            f'--precision-sparse={self.precision}',
            f'--sparse-m={self.sparse_m}',
            f'--sparse-n={self.sparse_n}',
            f'--sparse-k={self.sparse_k}',
            f'--sparse-density={self.density}',
        ])
    
    @performance_function('GFLOP/s')
    def sparse_gflops(self):
        """Extract sparse MM performance in GFLOP/s.
        
        Output format: [GPU{id} Sparse MM] Performance: {min} / {mean} / {max} GFLOP/s
        """
        return sn.extractsingle(
            r'\[GPU\d+\s+Sparse MM\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )


# ============================================================================
# FULL BENCHMARK SUITE
# ============================================================================
@rfm.simple_test
class TorchHammerFullSuite(TorchHammerBase):
    """Run all benchmarks in sequence."""
    
    descr = 'Torch Hammer Full Benchmark Suite'
    tags = {'gpu', 'full', 'suite'}
    
    time_limit = '2h'
    
    @run_before('run')
    def set_full_options(self):
        """Enable all benchmarks."""
        self.executable_opts.extend([
            '--batched-gemm',
            '--convolution',
            '--fft',
            '--einsum',
            '--memory-traffic',
            '--heat-equation',
            '--schrodinger',
        ])
    
    @sanity_function
    def validate_full_run(self):
        """Validate that all benchmarks completed."""
        return sn.all([
            sn.assert_found(r'Batched GEMM', self.stdout),
            sn.assert_found(r'Convolution', self.stdout),
            sn.assert_found(r'3D FFT', self.stdout),
            sn.assert_found(r'Einsum Attention', self.stdout),
            sn.assert_found(r'Memory Traffic', self.stdout),
            sn.assert_found(r'Heat Equation', self.stdout),
            sn.assert_found(r'Schr.dinger Equation', self.stdout),
        ])
    
    @performance_function('GFLOP/s')
    def gemm_perf(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Batched GEMM\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )
    
    @performance_function('img/s')
    def conv_perf(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Convolution\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+img/s',
            self.stdout, 1, float
        )
    
    @performance_function('GFLOP/s')
    def fft_perf(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+3D FFT\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GFLOP/s',
            self.stdout, 1, float
        )
    
    @performance_function('GB/s')
    def mem_perf(self):
        return sn.extractsingle(
            r'\[GPU\d+\s+Memory Traffic\]\s+Performance:\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+\s+GB/s',
            self.stdout, 1, float
        )


# ============================================================================
# MULTI-GPU BENCHMARK
# ============================================================================
@rfm.simple_test
class TorchHammerMultiGPU(TorchHammerBase):
    """Multi-GPU parallel benchmark."""
    
    descr = 'Torch Hammer Multi-GPU Benchmark'
    tags = {'gpu', 'multi-gpu', 'parallel'}
    
    num_gpus = variable(int, value=4)
    time_limit = '1h'
    
    @run_before('run')
    def set_multi_gpu_options(self):
        """Configure for multi-GPU execution."""
        # Remove single device index
        self.executable_opts = [
            opt for opt in self.executable_opts 
            if not opt.startswith('--device-index')
        ]
        
        # Add multi-GPU options
        gpu_list = ','.join(str(i) for i in range(self.num_gpus))
        self.executable_opts.extend([
            f'--gpu-list={gpu_list}',
            '--batched-gemm',
            '--cpu-affinity',
        ])
    
    @sanity_function
    def validate_multi_gpu(self):
        """Validate all GPUs completed."""
        # Check that we see output from all GPUs
        # device_label() outputs "GPU0", "GPU1" etc. (no space)
        checks = [
            sn.assert_found(rf'\[GPU{i}\b', self.stdout) 
            for i in range(self.num_gpus)
        ]
        return sn.all(checks)
