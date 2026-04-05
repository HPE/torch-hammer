# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Functional smoke tests for torch-hammer benchmarks.

These tests run each benchmark with minimal iterations on CPU to verify:
1. Benchmarks execute without crashing
2. Output format is correct
3. Results are valid (not NaN, positive values)

Note: These tests do NOT validate performance numbers (hardware-dependent).
"""
import logging
import pytest
import torch
from unittest.mock import MagicMock


# Skip all tests if torch isn't available (shouldn't happen, but be safe)
pytestmark = pytest.mark.skipif(
    not torch.__version__,
    reason="PyTorch not available"
)


class TestBenchmarkExecution:
    """Smoke tests that run benchmarks with minimal iterations."""
    
    @pytest.fixture
    def cpu_device(self):
        """Provide CPU device for testing."""
        return torch.device("cpu")
    
    @pytest.fixture
    def mock_telemetry(self, th):
        """Provide mock telemetry for testing."""
        tel = th.CpuTelemetry(0)
        return tel
    
    @pytest.fixture
    def mock_telemetry_thread(self, th, mock_telemetry, cpu_device):
        """Provide mock telemetry thread for testing."""
        thread = th.TelemetryThread(mock_telemetry, cpu_device, sample_interval_ms=50)
        # Pre-populate with one reading
        thread.latest_reading = mock_telemetry.read()
        # Mock iteration telemetry
        def mock_get_iteration_telemetry(i):
            return mock_telemetry.read()
        thread.get_iteration_telemetry = mock_get_iteration_telemetry
        return thread
    
    @pytest.fixture
    def mock_logger(self):
        """Provide mock logger for testing."""
        logger = logging.getLogger("test_smoke")
        logger.setLevel(logging.WARNING)  # Suppress INFO during tests
        return logger
    
    @pytest.fixture
    def mock_printer(self, th, mock_logger):
        """Provide mock verbose printer for testing."""
        return th.VerbosePrinter(mock_logger, ["vendor", "model", "device_id"], 0)
    
    @pytest.fixture
    def base_args(self, parser):
        """Provide base arguments with minimal iterations."""
        args = parser.parse_args([
            "--warmup", "1",
            "--skip-telemetry-first-n", "0",
        ])
        # Set minimal iterations for all benchmarks
        args.inner_loop_batched_gemm = 2
        args.inner_loop_convolution = 2
        args.inner_loop_fft = 2
        args.inner_loop_einsum = 2
        args.inner_loop_memory_traffic = 2
        args.inner_loop_heat_equation = 2
        args.inner_loop_schrodinger = 2
        args.inner_loop_atomic = 2
        args.inner_loop_sparse = 2
        args.verbose = False
        args.max_iterations = None
        args.duration = None
        # Add attributes expected by benchmark functions
        args.temp_warn_C = 90.0
        args.temp_critical_C = 95.0
        args.power_warn_pct = 98.0
        args._hardware_baselines = None
        return args
    
    def test_batched_gemm_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Batched GEMM benchmark should run without errors on CPU."""
        base_args.batched_gemm = True
        base_args.batch_count_gemm = 2
        base_args.m = 64
        base_args.n = 64
        base_args.k = 64
        base_args.precision_gemm = "float32"
        base_args.batched_gemm_TF32_mode = False
        
        # Reset telemetry stats
        mock_telemetry.reset_stats()
        
        # Run the benchmark
        result = th.batched_gemm_test(
            base_args, cpu_device, mock_logger, 
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        # Verify result structure
        assert result is not None
        assert "name" in result
        assert "min" in result
        assert "mean" in result
        assert "max" in result
        assert result["name"] == "Batched GEMM"
        
        # Verify values are valid
        assert result["min"] > 0
        assert result["mean"] > 0
        assert result["max"] > 0
        assert result["min"] <= result["mean"] <= result["max"]
    
    def test_convolution_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Convolution benchmark should run without errors on CPU."""
        base_args.convolution = True
        base_args.batch_count_convolution = 2
        base_args.in_channels = 3
        base_args.out_channels = 16
        base_args.height = 32
        base_args.width = 32
        base_args.kernel_size = 3
        base_args.precision_convolution = "float32"
        
        mock_telemetry.reset_stats()
        
        result = th.convolution_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
        assert result["mean"] > 0
    
    def test_fft_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """FFT benchmark should run without errors on CPU."""
        base_args.fft = True
        base_args.batch_count_fft = 2
        base_args.nx = 16
        base_args.ny = 16
        base_args.nz = 16
        base_args.precision_fft = "float32"
        
        mock_telemetry.reset_stats()
        
        result = th.fft_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
        assert result["mean"] > 0
    
    def test_einsum_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Einsum attention benchmark should run without errors on CPU."""
        base_args.einsum = True
        base_args.batch_count_einsum = 2
        base_args.heads = 2
        base_args.seq_len = 16
        base_args.d_model = 16
        base_args.precision_einsum = "float32"
        
        mock_telemetry.reset_stats()
        
        result = th.einsum_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
        assert result["mean"] > 0
    
    def test_memory_traffic_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Memory traffic benchmark should run without errors on CPU."""
        base_args.memory_traffic = True
        base_args.memory_size = 64  # Small size for testing
        base_args.memory_iterations = 2
        base_args.memory_pattern = "random"
        base_args.precision_memory = "float32"
        
        mock_telemetry.reset_stats()
        
        result = th.memory_traffic_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
        assert result["mean"] > 0
    
    def test_memory_traffic_streaming(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Memory traffic with streaming pattern should run."""
        base_args.memory_traffic = True
        base_args.memory_size = 256  # Larger size needed for streaming pattern chunk calculation
        base_args.memory_iterations = 2
        base_args.memory_pattern = "streaming"
        base_args.precision_memory = "float32"
        
        mock_telemetry.reset_stats()
        
        result = th.memory_traffic_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
    
    def test_heat_equation_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Heat equation benchmark should run without errors on CPU."""
        base_args.heat_equation = True
        base_args.heat_grid_size = 32
        base_args.heat_time_steps = 5
        base_args.alpha = 0.01
        base_args.delta_t = 0.01
        base_args.precision_heat = "float32"
        
        mock_telemetry.reset_stats()
        
        result = th.laplacian_heat_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
        assert result["mean"] > 0
    
    def test_schrodinger_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Schrödinger equation benchmark should run without errors on CPU."""
        base_args.schrodinger = True
        base_args.schrodinger_grid_size = 32
        base_args.schrodinger_time_steps = 5
        base_args.schrodinger_delta_x = 0.1
        base_args.schrodinger_delta_t = 0.01
        base_args.schrodinger_hbar = 1.0
        base_args.schrodinger_mass = 1.0
        base_args.schrodinger_potential = "harmonic"
        base_args.precision_schrodinger = "complex64"
        
        mock_telemetry.reset_stats()
        
        result = th.schrodinger_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0
        assert result["mean"] > 0
    
    def test_schrodinger_barrier_potential(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Schrödinger with barrier potential should run."""
        base_args.schrodinger = True
        base_args.schrodinger_grid_size = 32
        base_args.schrodinger_time_steps = 5
        base_args.schrodinger_delta_x = 0.1
        base_args.schrodinger_delta_t = 0.01
        base_args.schrodinger_hbar = 1.0
        base_args.schrodinger_mass = 1.0
        base_args.schrodinger_potential = "barrier"
        base_args.precision_schrodinger = "complex64"
        
        mock_telemetry.reset_stats()
        
        result = th.schrodinger_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is not None
        assert result["min"] > 0

    def test_atomic_contention_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Atomic contention benchmark should run without errors on CPU."""
        base_args.atomic_contention = True
        base_args.atomic_target_size = 1000
        base_args.atomic_num_updates = 5000
        base_args.atomic_contention_range = 64
        base_args.precision_atomic = "float32"
        base_args.inner_loop_atomic = 2

        mock_telemetry.reset_stats()

        result = th.atomic_contention_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )

        assert result is not None
        assert result["name"] == "Atomic Contention"
        assert result["min"] > 0
        assert result["mean"] > 0
        assert result["min"] <= result["mean"] <= result["max"]

    def test_sparse_mm_runs(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Sparse MM benchmark should run without errors on CPU."""
        base_args.sparse_mm = True
        base_args.sparse_m = 64
        base_args.sparse_n = 64
        base_args.sparse_k = 64
        base_args.sparse_density = 0.10
        base_args.precision_sparse = "float32"
        base_args.inner_loop_sparse = 2

        mock_telemetry.reset_stats()

        result = th.sparse_mm_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )

        assert result is not None
        assert result["name"] == "Sparse MM"
        assert result["min"] > 0
        assert result["mean"] > 0
        assert result["min"] <= result["mean"] <= result["max"]


class TestPrecisionVariants:
    """Test benchmarks with different precision types.

    Mirrors the precision coverage in reframe/ci_functional_checks.py:
    - Standard benchmarks (GEMM, Conv, FFT, Einsum, Memory, Heat, Schrödinger)
      accept all 6 precisions (bfloat16, float16, float32, float64, complex64,
      complex128).
    - Atomic and Sparse accept only 4 real precisions (float16, bfloat16,
      float32, float64).
    """

    @pytest.fixture
    def cpu_device(self):
        return torch.device("cpu")

    @pytest.fixture
    def mock_telemetry(self, th):
        return th.CpuTelemetry(0)

    @pytest.fixture
    def mock_telemetry_thread(self, th, mock_telemetry, cpu_device):
        thread = th.TelemetryThread(mock_telemetry, cpu_device, sample_interval_ms=50)
        thread.latest_reading = mock_telemetry.read()
        thread.get_iteration_telemetry = lambda i: mock_telemetry.read()
        return thread

    @pytest.fixture
    def mock_logger(self):
        logger = logging.getLogger("test_precision")
        logger.setLevel(logging.WARNING)
        return logger

    @pytest.fixture
    def mock_printer(self, th, mock_logger):
        return th.VerbosePrinter(mock_logger, ["vendor", "model", "device_id"], 0)

    @pytest.fixture
    def base_args(self, parser):
        args = parser.parse_args(["--warmup", "1"])
        args.verbose = False
        args.max_iterations = None
        args.duration = None
        args.skip_telemetry_first_n = 0
        args.temp_warn_C = 90.0
        args.temp_critical_C = 95.0
        args.power_warn_pct = 98.0
        args._hardware_baselines = None
        return args

    # ── GEMM: float32, float64, complex64 ────────────────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64", "complex64"])
    def test_gemm_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """GEMM should work with real and complex precisions."""
        base_args.batch_count_gemm = 2
        base_args.m = 32
        base_args.n = 32
        base_args.k = 32
        base_args.precision_gemm = precision
        base_args.batched_gemm_TF32_mode = False
        base_args.inner_loop_batched_gemm = 2

        mock_telemetry.reset_stats()
        result = th.batched_gemm_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Convolution: float32, float64 ────────────────────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_convolution_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Convolution should work with different precisions."""
        base_args.batch_count_convolution = 2
        base_args.in_channels = 3
        base_args.out_channels = 8
        base_args.height = 16
        base_args.width = 16
        base_args.kernel_size = 3
        base_args.precision_convolution = precision
        base_args.inner_loop_convolution = 2

        mock_telemetry.reset_stats()
        result = th.convolution_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── FFT: float32, float64, complex64, complex128 ────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64", "complex64", "complex128"])
    def test_fft_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """FFT should work with real and complex precisions."""
        base_args.batch_count_fft = 2
        base_args.nx = 16
        base_args.ny = 16
        base_args.nz = 16
        base_args.precision_fft = precision
        base_args.inner_loop_fft = 2

        mock_telemetry.reset_stats()
        result = th.fft_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Einsum: float32, float64, complex64 ─────────────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64", "complex64"])
    def test_einsum_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Einsum attention should work with real and complex precisions."""
        base_args.batch_count_einsum = 2
        base_args.heads = 2
        base_args.seq_len = 16
        base_args.d_model = 16
        base_args.precision_einsum = precision
        base_args.inner_loop_einsum = 2

        mock_telemetry.reset_stats()
        result = th.einsum_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Memory Traffic: float32, float64 ────────────────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_memory_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Memory traffic should work with different precisions."""
        base_args.memory_size = 64
        base_args.memory_iterations = 2
        base_args.memory_pattern = "random"
        base_args.precision_memory = precision
        base_args.inner_loop_memory_traffic = 2

        mock_telemetry.reset_stats()
        result = th.memory_traffic_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Heat Equation: float32, float64 ─────────────────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_heat_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Heat equation should work with different precisions."""
        base_args.heat_grid_size = 32
        base_args.heat_time_steps = 5
        base_args.alpha = 0.01
        base_args.delta_t = 0.01
        base_args.precision_heat = precision
        base_args.inner_loop_heat_equation = 2

        mock_telemetry.reset_stats()
        result = th.laplacian_heat_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Schrödinger: float32, complex64, complex128 ─────────────────
    @pytest.mark.parametrize("precision", ["float32", "complex64", "complex128"])
    def test_schrodinger_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Schrödinger equation should work with real and complex precisions."""
        base_args.schrodinger_grid_size = 32
        base_args.schrodinger_time_steps = 5
        base_args.schrodinger_delta_x = 0.1
        base_args.schrodinger_delta_t = 0.01
        base_args.schrodinger_hbar = 1.0
        base_args.schrodinger_mass = 1.0
        base_args.schrodinger_potential = "harmonic"
        base_args.precision_schrodinger = precision
        base_args.inner_loop_schrodinger = 2

        mock_telemetry.reset_stats()
        result = th.schrodinger_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Atomic Contention: float32, float64 (real-only) ─────────────
    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_atomic_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Atomic contention should work with real precisions."""
        base_args.atomic_target_size = 1000
        base_args.atomic_num_updates = 5000
        base_args.atomic_contention_range = 64
        base_args.precision_atomic = precision
        base_args.inner_loop_atomic = 2

        mock_telemetry.reset_stats()
        result = th.atomic_contention_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0

    # ── Sparse MM: float32, float64 (real-only) ─────────────────────
    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_sparse_precisions(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, precision):
        """Sparse MM should work with real precisions."""
        base_args.sparse_m = 64
        base_args.sparse_n = 64
        base_args.sparse_k = 64
        base_args.sparse_density = 0.10
        base_args.precision_sparse = precision
        base_args.inner_loop_sparse = 2

        mock_telemetry.reset_stats()
        result = th.sparse_mm_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        assert result is not None
        assert result["min"] > 0


class TestTimerIntegration:
    """Test Timer class with actual workloads."""
    
    def test_timer_measures_matmul(self, th):
        """Timer should measure matrix multiplication time."""
        device = torch.device("cpu")
        
        A = torch.rand(100, 100, device=device)
        B = torch.rand(100, 100, device=device)
        
        with th.Timer(device) as t:
            for _ in range(10):
                torch.matmul(A, B)
        
        assert t.elapsed > 0
        # Should take at least some measurable time
        assert t.elapsed < 10.0  # But not too long
    
    def test_timer_cuda_fallback(self, th):
        """Timer should work on CPU even in 'cuda' branch logic."""
        # This tests that Timer handles CPU correctly
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        assert timer.cuda is False
        
        timer.__enter__()
        torch.rand(100, 100)
        timer.__exit__(None, None, None)
        
        assert timer.elapsed > 0


class TestResultStructure:
    """Test that benchmark results have correct structure."""
    
    @pytest.fixture
    def cpu_device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def mock_telemetry(self, th):
        return th.CpuTelemetry(0)
    
    @pytest.fixture
    def mock_telemetry_thread(self, th, mock_telemetry, cpu_device):
        thread = th.TelemetryThread(mock_telemetry, cpu_device, sample_interval_ms=50)
        thread.latest_reading = mock_telemetry.read()
        thread.get_iteration_telemetry = lambda i: mock_telemetry.read()
        return thread
    
    @pytest.fixture
    def mock_logger(self):
        logger = logging.getLogger("test_result")
        logger.setLevel(logging.WARNING)
        return logger
    
    @pytest.fixture
    def mock_printer(self, th, mock_logger):
        return th.VerbosePrinter(mock_logger, ["vendor", "model", "device_id"], 0)
    
    @pytest.fixture
    def base_args(self, parser):
        args = parser.parse_args(["--warmup", "1"])
        args.inner_loop_batched_gemm = 3
        args.verbose = False
        args.max_iterations = None
        args.duration = None
        args.skip_telemetry_first_n = 0
        args.temp_warn_C = 90.0
        args.temp_critical_C = 95.0
        args.power_warn_pct = 98.0
        args._hardware_baselines = None
        return args
    
    def test_result_contains_params(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Results should include test parameters."""
        base_args.batched_gemm = True
        base_args.batch_count_gemm = 4
        base_args.m = 64
        base_args.n = 64
        base_args.k = 64
        base_args.precision_gemm = "float32"
        base_args.batched_gemm_TF32_mode = False
        
        mock_telemetry.reset_stats()
        
        result = th.batched_gemm_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert "params" in result
        assert "batch" in result["params"]
        assert "m" in result["params"]
        assert "n" in result["params"]
        assert "k" in result["params"]
        assert result["params"]["batch"] == 4
        assert result["params"]["m"] == 64
    
    def test_result_contains_unit(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        """Results should include measurement unit."""
        base_args.batched_gemm = True
        base_args.batch_count_gemm = 2
        base_args.m = 32
        base_args.n = 32
        base_args.k = 32
        base_args.precision_gemm = "float32"
        base_args.batched_gemm_TF32_mode = False
        
        mock_telemetry.reset_stats()
        
        result = th.batched_gemm_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert "unit" in result
        assert result["unit"] == "GFLOP/s"
