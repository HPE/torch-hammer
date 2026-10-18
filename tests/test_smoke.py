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


def _raise_bcomplex32(*args, **kwargs):
    """Stand-in for a torch op whose kernel is missing for a promoted complex dtype.

    Reproduces the torch 2.14 CUDA failure from GitHub issue #43, where
    bfloat16 promotes to the experimental BComplex32 dtype.
    """
    raise NotImplementedError('"mul_cuda" not implemented for \'BComplex32\'')


def _raise_unsupported_fft_dtype(*args, **kwargs):
    """Stand-in for torch.fft.fftn on a backend without half-precision FFT kernels.

    Reproduces the CPU and ROCm failure from GitHub issue #45, where
    torch.fft has no bfloat16/float16 kernels (cuFFT on CUDA has them).
    """
    raise RuntimeError("Unsupported dtype BFloat16")


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

    def test_fft_unsupported_dtype_skips_gracefully(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, monkeypatch):
        """FFT must skip with a warning, not crash, when torch.fft has no kernels for the dtype."""
        base_args.fft = True
        base_args.batch_count_fft = 1
        base_args.nx = 8
        base_args.ny = 8
        base_args.nz = 8
        base_args.precision_fft = "bfloat16"
        
        monkeypatch.setattr(th.torch.fft, "fftn", _raise_unsupported_fft_dtype)
        monkeypatch.setattr(mock_logger, "warning", MagicMock())
        monkeypatch.setattr(mock_logger, "error", MagicMock())
        
        mock_telemetry.reset_stats()
        
        result = th.fft_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is None
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args.args[0]
        assert "not supported" in warning_message
        assert "skipping" in warning_message
        mock_logger.error.assert_not_called()

    def test_fft_bfloat16_on_cpu_does_not_crash(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, monkeypatch):
        """bfloat16 FFT must either run (cuFFT) or skip gracefully (no CPU/ROCm kernels), never crash."""
        base_args.fft = True
        base_args.batch_count_fft = 1
        base_args.nx = 8
        base_args.ny = 8
        base_args.nz = 8
        base_args.precision_fft = "bfloat16"
        
        monkeypatch.setattr(mock_logger, "warning", MagicMock())
        monkeypatch.setattr(mock_logger, "error", MagicMock())
        
        mock_telemetry.reset_stats()
        
        result = th.fft_test(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        mock_logger.error.assert_not_called()
        assert result is None or result["unit"] == "GFLOP/s"
        if result is None:
            mock_logger.warning.assert_called_once()
            warning_message = mock_logger.warning.call_args.args[0]
            assert "not supported" in warning_message
            assert "skipping" in warning_message
    
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

    def test_schrodinger_unsupported_dtype_skips_gracefully(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, monkeypatch):
        """Schrödinger must skip with a warning, not crash, when the promoted complex dtype has no kernels."""
        base_args.schrodinger = True
        base_args.schrodinger_grid_size = 32
        base_args.schrodinger_time_steps = 5
        base_args.schrodinger_delta_x = 0.1
        base_args.schrodinger_delta_t = 0.01
        base_args.schrodinger_hbar = 1.0
        base_args.schrodinger_mass = 1.0
        base_args.schrodinger_potential = "harmonic"
        base_args.precision_schrodinger = "bfloat16"
        
        monkeypatch.setattr(th.torch, "exp", _raise_bcomplex32)
        monkeypatch.setattr(mock_logger, "warning", MagicMock())
        monkeypatch.setattr(mock_logger, "error", MagicMock())
        
        mock_telemetry.reset_stats()
        
        result = th.schrodinger_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        assert result is None
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args.args[0]
        assert "not supported" in warning_message
        assert "skipping" in warning_message
        mock_logger.error.assert_not_called()

    def test_schrodinger_float16_on_cpu_does_not_crash(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, monkeypatch):
        """float16 promotes to ComplexHalf, which lacks CPU exp kernels: either skip gracefully or run, never crash."""
        base_args.schrodinger = True
        base_args.schrodinger_grid_size = 32
        base_args.schrodinger_time_steps = 5
        base_args.schrodinger_delta_x = 0.1
        base_args.schrodinger_delta_t = 0.01
        base_args.schrodinger_hbar = 1.0
        base_args.schrodinger_mass = 1.0
        base_args.schrodinger_potential = "harmonic"
        base_args.precision_schrodinger = "float16"
        
        monkeypatch.setattr(mock_logger, "warning", MagicMock())
        monkeypatch.setattr(mock_logger, "error", MagicMock())
        
        mock_telemetry.reset_stats()
        
        result = th.schrodinger_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        mock_logger.error.assert_not_called()
        assert result is None or result["unit"] == "iter/s"
        if result is None:
            mock_logger.warning.assert_called_once()
            warning_message = mock_logger.warning.call_args.args[0]
            assert "not supported" in warning_message
            assert "skipping" in warning_message

    def test_schrodinger_bfloat16_on_cpu_does_not_crash(self, th, base_args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer, monkeypatch):
        """bfloat16 must either run (complex64 promotion) or skip gracefully (BComplex32 without kernels), never crash."""
        base_args.schrodinger = True
        base_args.schrodinger_grid_size = 32
        base_args.schrodinger_time_steps = 5
        base_args.schrodinger_delta_x = 0.1
        base_args.schrodinger_delta_t = 0.01
        base_args.schrodinger_hbar = 1.0
        base_args.schrodinger_mass = 1.0
        base_args.schrodinger_potential = "harmonic"
        base_args.precision_schrodinger = "bfloat16"
        
        monkeypatch.setattr(mock_logger, "info", MagicMock())
        monkeypatch.setattr(mock_logger, "error", MagicMock())
        
        mock_telemetry.reset_stats()
        
        result = th.schrodinger_equation(
            base_args, cpu_device, mock_logger,
            mock_telemetry, mock_telemetry_thread, mock_printer
        )
        
        mock_logger.error.assert_not_called()
        assert result is None or result["unit"] == "iter/s"
        if result is not None:
            info_messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
            assert any("complex dtype" in message for message in info_messages)

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


class TestMeltExecution:
    """CPU smoke tests for melt mode (TIM settling) — tiny matrices, short caps."""

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
        return thread

    @pytest.fixture
    def mock_logger(self):
        logger = logging.getLogger("test_melt_smoke")
        logger.setLevel(logging.WARNING)
        return logger

    @pytest.fixture
    def mock_printer(self, th, mock_logger):
        return th.VerbosePrinter(mock_logger, ["vendor", "model", "device_id"], 0)

    @pytest.fixture
    def melt_args(self, parser):
        args = parser.parse_args([
            "--melt", "--warmup", "1", "--skip-telemetry-first-n", "0",
            "--melt-batch-count", "2", "--melt-m", "32", "--melt-n", "32", "--melt-k", "32",
            "--melt-burst-ms", "5", "--melt-cool-ms", "5", "--melt-min-duration", "0.2", "--melt-window-s", "0.2",
            "--melt-bin-s", "0.05",
            "--melt-max-duration", "2", "--melt-plateau-cv-pct", "100", "--melt-settled-pct", "0",
        ])
        args._hardware_baselines = None
        return args

    def _run(self, th, args, cpu_device, mock_telemetry, mock_telemetry_thread, mock_logger, mock_printer):
        mock_telemetry.reset_stats()
        return th.melt_test(args, cpu_device, mock_logger, mock_telemetry, mock_telemetry_thread, mock_printer)

    def test_melt_releases_cool_gap_when_done(self, th, melt_args, cpu_device, mock_telemetry,
                                              mock_telemetry_thread, mock_logger, mock_printer, monkeypatch):
        released = []
        real_factory = th._make_melt_cool_gap

        def spying_factory(*a, **kw):
            gap = real_factory(*a, **kw)
            real_release = gap.release
            gap.release = lambda: (released.append(True), real_release())
            return gap

        monkeypatch.setattr(th, "_make_melt_cool_gap", spying_factory)
        result = self._run(th, melt_args, cpu_device, mock_telemetry, mock_telemetry_thread,
                           mock_logger, mock_printer)
        assert result is not None
        assert released == [True]

    def test_melt_runs_and_settles_on_cpu(self, th, melt_args, cpu_device, mock_telemetry,
                                          mock_telemetry_thread, mock_logger, mock_printer):
        result = self._run(th, melt_args, cpu_device, mock_telemetry, mock_telemetry_thread,
                           mock_logger, mock_printer)
        assert result is not None
        assert result["name"] == "Melt"
        assert result["unit"] == "GFLOP/s"
        assert result["reload_path"] == "uniform_"
        assert result["cycles"] > 0
        assert result["iterations"] == result["cycles"]
        assert result["ops_per_burst"] >= 1
        assert result["burst_ms_measured"] > 0
        assert result["settled"] is True
        assert 0 < result["settle_s"] <= 2.5
        assert result["start_gflops"] > 0
        assert result["plateau_gflops"] > 0
        assert result["thermal_throttle_cycles"] == 0
        assert result["last_throttle_s"] is None
        assert result["params"]["dtype"] == "float32"
        assert (result["params"]["batch"], result["params"]["m"]) == (2, 32)
        assert result["params"]["burst_ms"] == 5

    def test_melt_not_settled_when_capped_before_min_duration(self, th, melt_args, cpu_device, mock_telemetry,
                                                              mock_telemetry_thread, mock_logger, mock_printer):
        melt_args.melt_max_duration = 0.3
        melt_args.melt_min_duration = 100
        result = self._run(th, melt_args, cpu_device, mock_telemetry, mock_telemetry_thread,
                           mock_logger, mock_printer)
        assert result is not None
        assert result["settled"] is False
        assert result["settle_s"] is None
        assert result["cycles"] > 0
        assert result["runtime_s"] >= 0.3

    def test_melt_trajectory_in_iteration_telemetry(self, th, melt_args, cpu_device, mock_telemetry,
                                                    mock_telemetry_thread, mock_logger, mock_printer):
        # The idle poll interval is 500 ms: run long enough for the thread to
        # wake up and take samples inside burst windows.
        melt_args.melt_min_duration = 1.0
        melt_args.melt_window_s = 0.5
        melt_args.melt_max_duration = 3.0
        mock_telemetry_thread.start()
        try:
            result = self._run(th, melt_args, cpu_device, mock_telemetry, mock_telemetry_thread,
                               mock_logger, mock_printer)
        finally:
            mock_telemetry_thread.stop()
        trajectory = result["iteration_telemetry"]
        assert trajectory, "expected per-cycle records when the telemetry thread is sampling"
        for entry in trajectory:
            assert entry["thermal_throttled"] is False
            assert "power_min_W" in entry and "power_max_W" in entry
        # melt leaves no per-iteration samples behind for the next benchmark
        assert mock_telemetry_thread.iteration_samples == {}

    def test_melt_ignores_characterization_dims(self, th, parser, cpu_device, mock_telemetry,
                                                mock_telemetry_thread, mock_logger, mock_printer):
        args = parser.parse_args([
            "--melt", "--warmup", "1", "--m", "999", "--n", "999", "--k", "999", "--batch-count-gemm", "99",
            "--melt-m", "16", "--melt-n", "16", "--melt-k", "16", "--melt-batch-count", "2",
            "--melt-burst-ms", "2", "--melt-cool-ms", "1", "--melt-min-duration", "0", "--melt-window-s", "0.2",
            "--melt-max-duration", "0.5", "--melt-plateau-cv-pct", "100", "--melt-settled-pct", "0",
        ])
        args._hardware_baselines = None
        result = self._run(th, args, cpu_device, mock_telemetry, mock_telemetry_thread,
                           mock_logger, mock_printer)
        assert (result["params"]["batch"], result["params"]["m"], result["params"]["n"], result["params"]["k"]) == (2, 16, 16, 16)
        assert result["params"]["cool_ms"] == 1

    def test_melt_result_reports_cool_gap_measurements(self, th, melt_args, cpu_device, mock_telemetry,
                                                        mock_telemetry_thread, mock_logger, mock_printer):
        result = self._run(th, melt_args, cpu_device, mock_telemetry, mock_telemetry_thread,
                           mock_logger, mock_printer)
        assert result["reload_ms_measured"] >= 0
        assert result["params"]["bin_s"] == melt_args.melt_bin_s
        assert result["plateau_cv_pct"] is not None and result["cycle_cv_pct"] is not None
        assert 0 < result["duty_cycle_pct"] < 100
        assert result["duty_cycle_pct"] <= 5 / (5 + 5) * 100 + 15   # 5 ms burst / 5 ms floor, loose bound
