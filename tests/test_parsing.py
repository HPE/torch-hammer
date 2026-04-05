# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for argument parsing and configuration.
"""
import pytest


class TestArgumentParser:
    """Tests for the CLI argument parser."""
    
    def test_parser_creates_successfully(self, parser):
        """Parser should be created without errors."""
        assert parser is not None
        assert parser.prog == "TORCH-HAMMER"
    
    def test_default_values(self, default_args):
        """Default values should be set correctly."""
        # Global defaults
        assert default_args.device_index == 0
        assert default_args.warmup == 10
        assert default_args.verbose is False
        assert default_args.no_log is False
        assert default_args.repeats == 1
        
        # Benchmark enable flags default to False
        assert default_args.batched_gemm is False
        assert default_args.convolution is False
        assert default_args.fft is False
        assert default_args.einsum is False
        assert default_args.memory_traffic is False
        assert default_args.heat_equation is False
        assert default_args.schrodinger is False
    
    def test_precision_defaults(self, default_args):
        """Precision defaults should all be float32."""
        assert default_args.precision_gemm == "float32"
        assert default_args.precision_convolution == "float32"
        assert default_args.precision_fft == "float32"
        assert default_args.precision_einsum == "float32"
        assert default_args.precision_memory == "float32"
        assert default_args.precision_heat == "float32"
        assert default_args.precision_schrodinger == "float32"
        assert default_args.precision_atomic == "float32"
        assert default_args.precision_sparse == "float32"
    
    def test_gemm_args(self, parser):
        """GEMM-specific arguments should parse correctly."""
        args = parser.parse_args([
            "--batched-gemm",
            "--m", "1024",
            "--n", "2048",
            "--k", "512",
            "--batch-count-gemm", "64",
            "--precision-gemm", "float64",
            "--batched-gemm-TF32-mode",
        ])
        assert args.batched_gemm is True
        assert args.m == 1024
        assert args.n == 2048
        assert args.k == 512
        assert args.batch_count_gemm == 64
        assert args.precision_gemm == "float64"
        assert args.batched_gemm_TF32_mode is True
    
    def test_convolution_args(self, parser):
        """Convolution-specific arguments should parse correctly."""
        args = parser.parse_args([
            "--convolution",
            "--batch-count-convolution", "32",
            "--in-channels", "64",
            "--out-channels", "128",
            "--height", "256",
            "--width", "256",
            "--kernel-size", "5",
            "--precision-convolution", "float16",
        ])
        assert args.convolution is True
        assert args.batch_count_convolution == 32
        assert args.in_channels == 64
        assert args.out_channels == 128
        assert args.height == 256
        assert args.width == 256
        assert args.kernel_size == 5
        assert args.precision_convolution == "float16"
    
    def test_fft_args(self, parser):
        """FFT-specific arguments should parse correctly."""
        args = parser.parse_args([
            "--fft",
            "--batch-count-fft", "16",
            "--nx", "256",
            "--ny", "256",
            "--nz", "256",
            "--precision-fft", "complex64",
        ])
        assert args.fft is True
        assert args.batch_count_fft == 16
        assert args.nx == 256
        assert args.ny == 256
        assert args.nz == 256
        assert args.precision_fft == "complex64"
    
    def test_einsum_args(self, parser):
        """Einsum-specific arguments should parse correctly."""
        args = parser.parse_args([
            "--einsum",
            "--batch-count-einsum", "32",
            "--heads", "16",
            "--seq-len", "256",
            "--d-model", "128",
        ])
        assert args.einsum is True
        assert args.batch_count_einsum == 32
        assert args.heads == 16
        assert args.seq_len == 256
        assert args.d_model == 128
    
    def test_memory_args(self, parser):
        """Memory traffic arguments should parse correctly."""
        args = parser.parse_args([
            "--memory-traffic",
            "--memory-size", "2048",
            "--memory-iterations", "20",
            "--memory-pattern", "streaming",
        ])
        assert args.memory_traffic is True
        assert args.memory_size == 2048
        assert args.memory_iterations == 20
        assert args.memory_pattern == "streaming"
    
    def test_heat_equation_args(self, parser):
        """Heat equation arguments should parse correctly."""
        args = parser.parse_args([
            "--heat-equation",
            "--heat-grid-size", "256",
            "--heat-time-steps", "200",
            "--alpha", "0.02",
            "--delta-t", "0.005",
        ])
        assert args.heat_equation is True
        assert args.heat_grid_size == 256
        assert args.heat_time_steps == 200
        assert args.alpha == 0.02
        assert args.delta_t == 0.005
    
    def test_schrodinger_args(self, parser):
        """Schrödinger equation arguments should parse correctly."""
        args = parser.parse_args([
            "--schrodinger",
            "--schrodinger-grid-size", "256",
            "--schrodinger-time-steps", "200",
            "--schrodinger-delta-x", "0.05",
            "--schrodinger-delta-t", "0.005",
            "--schrodinger-potential", "barrier",
            "--precision-schrodinger", "complex128",
        ])
        assert args.schrodinger is True
        assert args.schrodinger_grid_size == 256
        assert args.schrodinger_time_steps == 200
        assert args.schrodinger_delta_x == 0.05
        assert args.schrodinger_delta_t == 0.005
        assert args.schrodinger_potential == "barrier"
        assert args.precision_schrodinger == "complex128"
    
    def test_gpu_selection_args(self, parser):
        """GPU selection arguments should parse correctly."""
        # Single GPU
        args = parser.parse_args(["--device-index", "2"])
        assert args.device_index == 2
        
        # All GPUs
        args = parser.parse_args(["--all-gpus"])
        assert args.all_gpus is True
        
        # GPU list
        args = parser.parse_args(["--gpu-list", "0,2,3"])
        assert args.gpu_list == "0,2,3"
    
    def test_cpu_affinity_args(self, parser):
        """CPU affinity arguments should parse correctly."""
        args = parser.parse_args([
            "--cpu-affinity",
            "--cpu-gpu-map", "0:0-15,1:16-31",
            "--parent-cpu", "63",
        ])
        assert args.cpu_affinity is True
        assert args.cpu_gpu_map == "0:0-15,1:16-31"
        assert args.parent_cpu == 63
    
    def test_cpu_list_arg(self, parser):
        """CPU list argument for CPU-only mode should parse correctly."""
        args = parser.parse_args(["--cpu-list", "0-23,48-71"])
        assert args.cpu_list == "0-23,48-71"
        
        args = parser.parse_args(["--cpu-list", "all"])
        assert args.cpu_list == "all"
    
    def test_duration_mode_args(self, parser):
        """Duration mode arguments should parse correctly."""
        args = parser.parse_args([
            "--duration", "60.0",
            "--min-iterations", "5",
            "--max-iterations", "1000",
        ])
        assert args.duration == 60.0
        assert args.min_iterations == 5
        assert args.max_iterations == 1000
    
    def test_telemetry_args(self, parser):
        """Telemetry arguments should parse correctly."""
        args = parser.parse_args([
            "--skip-telemetry-first-n", "5",
            "--telemetry-interval-ms", "200",
            "--no-telemetry-thread",
        ])
        assert args.skip_telemetry_first_n == 5
        assert args.telemetry_interval_ms == 200
        assert args.no_telemetry_thread is True
    
    def test_output_args(self, parser):
        """Output arguments should parse correctly."""
        args = parser.parse_args([
            "--log-file", "/tmp/test.log",
            "--json-output", "/tmp/results.json",
            "--summary-csv", "/tmp/summary.csv",
            "--verbose",
            "--verbose-file-only",
        ])
        assert args.log_file == "/tmp/test.log"
        assert args.json_output == "/tmp/results.json"
        assert args.summary_csv == "/tmp/summary.csv"
        assert args.verbose is True
        assert args.verbose_file_only is True
    
    def test_invalid_precision_rejected(self, parser):
        """Invalid precision values should be rejected."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--precision-gemm", "invalid"])
    
    def test_invalid_memory_pattern_rejected(self, parser):
        """Invalid memory pattern values should be rejected."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--memory-pattern", "invalid"])
    
    def test_invalid_potential_rejected(self, parser):
        """Invalid potential values should be rejected."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--schrodinger-potential", "invalid"])
    
    def test_atomic_contention_args(self, parser):
        """Atomic contention arguments should parse correctly."""
        args = parser.parse_args([
            "--atomic-contention",
            "--atomic-target-size", "500000",
            "--atomic-num-updates", "5000000",
            "--atomic-contention-range", "512",
            "--precision-atomic", "float64",
        ])
        assert args.atomic_contention is True
        assert args.atomic_target_size == 500000
        assert args.atomic_num_updates == 5000000
        assert args.atomic_contention_range == 512
        assert args.precision_atomic == "float64"

    def test_sparse_mm_args(self, parser):
        """Sparse MM arguments should parse correctly."""
        args = parser.parse_args([
            "--sparse-mm",
            "--sparse-m", "4096",
            "--sparse-n", "4096",
            "--sparse-k", "4096",
            "--sparse-density", "0.05",
            "--precision-sparse", "float16",
        ])
        assert args.sparse_mm is True
        assert args.sparse_m == 4096
        assert args.sparse_n == 4096
        assert args.sparse_k == 4096
        assert args.sparse_density == 0.05
        assert args.precision_sparse == "float16"

    def test_multiple_benchmarks(self, parser):
        """Multiple benchmarks can be enabled simultaneously."""
        args = parser.parse_args([
            "--batched-gemm",
            "--convolution",
            "--fft",
            "--einsum",
        ])
        assert args.batched_gemm is True
        assert args.convolution is True
        assert args.fft is True
        assert args.einsum is True


class TestCpuGpuMapParsing:
    """Tests for CPU-GPU mapping string parsing."""
    
    def test_simple_mapping(self, th):
        """Simple CPU-GPU mapping should parse correctly."""
        result = th.parse_cpu_gpu_map("0:0-15")
        assert result == {0: list(range(0, 16))}
    
    def test_multiple_gpus(self, th):
        """Multiple GPU mappings should parse correctly."""
        result = th.parse_cpu_gpu_map("0:0-15,1:16-31")
        assert result == {
            0: list(range(0, 16)),
            1: list(range(16, 32)),
        }
    
    def test_non_contiguous_ranges(self, th):
        """Non-contiguous GPU indices should parse correctly."""
        result = th.parse_cpu_gpu_map("0:0-7,2:16-23,3:24-31")
        assert result == {
            0: list(range(0, 8)),
            2: list(range(16, 24)),
            3: list(range(24, 32)),
        }
    
    def test_single_cpu_mapping(self, th):
        """Single CPU per GPU should parse correctly."""
        # This format might need adjustment based on actual implementation
        result = th.parse_cpu_gpu_map("0:5")
        assert result == {0: [5]}
    
    def test_empty_string(self, th):
        """Empty string should return empty mapping."""
        result = th.parse_cpu_gpu_map("")
        assert result == {}
    
    def test_whitespace_handling(self, th):
        """Whitespace should be handled gracefully."""
        result = th.parse_cpu_gpu_map("0: 0-15 , 1: 16-31")
        assert 0 in result
        assert 1 in result


class TestConfigValidation:
    """Tests for configuration validation."""

    # ── Standard benchmarks accept all 6 precisions ───────────────
    STANDARD_PRECISION_FLAGS = [
        "--precision-gemm",
        "--precision-convolution",
        "--precision-fft",
        "--precision-einsum",
        "--precision-memory",
        "--precision-heat",
        "--precision-schrodinger",
    ]
    STANDARD_PRECISION_ATTR = [
        "precision_gemm",
        "precision_convolution",
        "precision_fft",
        "precision_einsum",
        "precision_memory",
        "precision_heat",
        "precision_schrodinger",
    ]
    PRECISION_ALL = ["bfloat16", "float16", "float32", "float64", "complex64", "complex128"]
    PRECISION_REAL = ["float16", "bfloat16", "float32", "float64"]

    @pytest.mark.parametrize("flag,attr", list(zip(STANDARD_PRECISION_FLAGS, STANDARD_PRECISION_ATTR)))
    @pytest.mark.parametrize("precision", PRECISION_ALL)
    def test_standard_precision_choices(self, parser, flag, attr, precision):
        """All 7 standard benchmark precision flags accept all 6 precisions."""
        args = parser.parse_args([flag, precision])
        assert getattr(args, attr) == precision

    @pytest.mark.parametrize("precision", PRECISION_REAL)
    def test_atomic_precision_choices(self, parser, precision):
        """Atomic precision flag accepts all 4 real precisions."""
        args = parser.parse_args(["--precision-atomic", precision])
        assert args.precision_atomic == precision

    @pytest.mark.parametrize("precision", PRECISION_REAL)
    def test_sparse_precision_choices(self, parser, precision):
        """Sparse precision flag accepts all 4 real precisions."""
        args = parser.parse_args(["--precision-sparse", precision])
        assert args.precision_sparse == precision

    @pytest.mark.parametrize("complex_dtype", ["complex64", "complex128"])
    def test_atomic_rejects_complex(self, parser, complex_dtype):
        """Atomic precision flag must reject complex types."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--precision-atomic", complex_dtype])

    @pytest.mark.parametrize("complex_dtype", ["complex64", "complex128"])
    def test_sparse_rejects_complex(self, parser, complex_dtype):
        """Sparse precision flag must reject complex types."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--precision-sparse", complex_dtype])

    def test_memory_pattern_choices(self, parser):
        """All valid memory patterns should be accepted."""
        valid_patterns = ["random", "streaming", "unit"]
        for pattern in valid_patterns:
            args = parser.parse_args(["--memory-pattern", pattern])
            assert args.memory_pattern == pattern
    
    def test_potential_choices(self, parser):
        """All valid potential types should be accepted."""
        valid_potentials = ["harmonic", "barrier"]
        for potential in valid_potentials:
            args = parser.parse_args(["--schrodinger-potential", potential])
            assert args.schrodinger_potential == potential


class TestCpuListParsing:
    """Tests for CPU list string parsing."""
    
    def test_simple_range(self, th):
        """Simple CPU range should parse correctly."""
        result = th.parse_cpu_list("0-7")
        assert result == list(range(8))
    
    def test_multiple_ranges(self, th):
        """Multiple CPU ranges should parse correctly."""
        result = th.parse_cpu_list("0-3,8-11")
        assert result == [0, 1, 2, 3, 8, 9, 10, 11]
    
    def test_single_cpus(self, th):
        """Single CPU IDs should parse correctly."""
        result = th.parse_cpu_list("0,4,8,12")
        assert result == [0, 4, 8, 12]
    
    def test_mixed_format(self, th):
        """Mixed ranges and single CPUs should parse correctly."""
        result = th.parse_cpu_list("0-3,7,10-12")
        assert result == [0, 1, 2, 3, 7, 10, 11, 12]
    
    def test_all_keyword(self, th):
        """'all' keyword should return all CPUs."""
        import os
        result = th.parse_cpu_list("all")
        assert result == list(range(os.cpu_count() or 1))
    
    def test_deduplication(self, th):
        """Duplicate CPUs should be removed."""
        result = th.parse_cpu_list("0-3,2-5")
        assert result == [0, 1, 2, 3, 4, 5]  # Sorted and deduplicated


class TestApplyConfigToArgs:
    """Tests for apply_config_to_args YAML global settings."""

    def _apply(self, th, parser, config):
        """Helper: parse empty CLI, apply config, return args."""
        import sys
        orig = sys.argv
        sys.argv = ["torch-hammer.py"]  # no CLI flags
        try:
            args = parser.parse_args([])
            return th.apply_config_to_args(args, config)
        finally:
            sys.argv = orig

    def test_syslog_from_yaml(self, th, parser):
        """syslog: true in YAML global should set args.syslog."""
        args = self._apply(th, parser, {"global": {"syslog": True}})
        assert args.syslog is True

    def test_syslog_dmesg_from_yaml(self, th, parser):
        """syslog_dmesg: true in YAML global should set args.syslog_dmesg."""
        args = self._apply(th, parser, {"global": {"syslog_dmesg": True}})
        assert args.syslog_dmesg is True

    def test_compact_from_yaml(self, th, parser):
        """compact: true in YAML global should set args.compact."""
        args = self._apply(th, parser, {"global": {"compact": True}})
        assert args.compact is True

    def test_all_output_modes_together(self, th, parser):
        """All three output flags can coexist from YAML."""
        config = {"global": {"compact": True, "syslog": True, "syslog_dmesg": True}}
        args = self._apply(th, parser, config)
        assert args.compact is True
        assert args.syslog is True
        assert args.syslog_dmesg is True

    def test_cli_overrides_yaml_syslog(self, th, parser):
        """CLI --syslog should take precedence over YAML syslog: false."""
        import sys
        orig = sys.argv
        sys.argv = ["torch-hammer.py", "--syslog"]
        try:
            args = parser.parse_args(["--syslog"])
            args = th.apply_config_to_args(args, {"global": {"syslog": False}})
            assert args.syslog is True
        finally:
            sys.argv = orig

    def test_yaml_false_does_not_override_default(self, th, parser):
        """syslog: false in YAML should leave default False unchanged."""
        args = self._apply(th, parser, {"global": {"syslog": False}})
        assert args.syslog is False

    def test_hyphen_key_normalised(self, th, parser):
        """YAML key 'syslog-dmesg' (hyphenated) should map to args.syslog_dmesg."""
        args = self._apply(th, parser, {"global": {"syslog-dmesg": True}})
        assert args.syslog_dmesg is True
