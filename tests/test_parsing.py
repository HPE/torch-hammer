# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for argument parsing and configuration.
"""
import logging

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

    def _apply_with_cli(self, th, parser, cli_flags, config):
        """Helper: parse the given CLI flags (mirrored into sys.argv), apply config, return args."""
        import sys
        orig = sys.argv
        sys.argv = ["torch-hammer.py"] + cli_flags
        try:
            args = parser.parse_args(cli_flags)
            return th.apply_config_to_args(args, config)
        finally:
            sys.argv = orig

    # ── Booleans whose parser default is True (#29) ──────────────────────

    def test_yaml_false_overrides_true_default(self, th, parser):
        """cpu_affinity: false in YAML global should disable the True-by-default flag."""
        args = self._apply(th, parser, {"global": {"cpu_affinity": False}})
        assert args.cpu_affinity is False

    def test_negated_key_true_disables(self, th, parser):
        """no_cpu_affinity: true (either spelling) should map to args.cpu_affinity = False."""
        args = self._apply(th, parser, {"global": {"no_cpu_affinity": True}})
        assert args.cpu_affinity is False
        args = self._apply(th, parser, {"global": {"no-cpu-affinity": True}})
        assert args.cpu_affinity is False

    def test_negated_key_false_keeps_enabled(self, th, parser):
        """no_cpu_affinity: false should leave affinity enabled."""
        args = self._apply(th, parser, {"global": {"no_cpu_affinity": False}})
        assert args.cpu_affinity is True

    def test_cli_negated_flag_beats_yaml(self, th, parser):
        """CLI --no-cpu-affinity must win over YAML cpu_affinity: true."""
        args = self._apply_with_cli(th, parser, ["--no-cpu-affinity"], {"global": {"cpu_affinity": True}})
        assert args.cpu_affinity is False

    def test_cli_positive_flag_beats_yaml_false(self, th, parser):
        """CLI --cpu-affinity must win over YAML cpu_affinity: false."""
        args = self._apply_with_cli(th, parser, ["--cpu-affinity"], {"global": {"cpu_affinity": False}})
        assert args.cpu_affinity is True

    def test_yaml_false_disables_verbose(self, th, parser):
        """verbose: false in YAML global should leave verbose off."""
        args = self._apply(th, parser, {"global": {"verbose": False}})
        assert args.verbose is False

    # ── duration under global: (#30) ─────────────────────────────────────

    def test_duration_in_global_section(self, th, parser):
        """duration under global: should be applied, not silently ignored."""
        args = self._apply(th, parser, {"global": {"duration": 30}})
        assert args.duration == 30

    def test_duration_in_runtime_section_still_works(self, th, parser):
        """duration under runtime: must keep working."""
        args = self._apply(th, parser, {"runtime": {"duration": 30}})
        assert args.duration == 30

    def test_cli_duration_beats_yaml(self, th, parser):
        """CLI --duration must win over YAML global duration."""
        args = self._apply_with_cli(th, parser, ["--duration", "5"], {"global": {"duration": 30}})
        assert args.duration == 5.0


class TestConfigGet:
    """Tests for the _config_get helper function."""

    def test_first_key_found(self, th):
        """Should return value from the first matching key."""
        config = {'precision': 'float32', 'precision_gemm': 'float64'}
        assert th._config_get(config, 'precision', 'precision_gemm', default='bfloat16') == 'float32'

    def test_second_key_found(self, th):
        """Should return value from the second key when first is missing."""
        config = {'precision_gemm': 'float64'}
        assert th._config_get(config, 'precision', 'precision_gemm', default='float32') == 'float64'

    def test_default_when_no_match(self, th):
        """Should return default when no keys match."""
        config = {'other_key': 'value'}
        assert th._config_get(config, 'precision', 'precision_gemm', default='float32') == 'float32'

    def test_empty_config(self, th):
        """Should return default for empty config dict."""
        assert th._config_get({}, 'precision', default='float32') == 'float32'

    def test_none_default(self, th):
        """Should return None when no keys match and no default given."""
        assert th._config_get({'x': 1}, 'y') is None


class TestConfigDispatchKeys:
    """Tests for config dispatch accepting both short and full attribute name keys."""

    def _apply(self, th, parser, config):
        """Helper: parse empty CLI, apply config, return args."""
        import sys
        orig = sys.argv
        sys.argv = ["torch-hammer.py"]
        try:
            args = parser.parse_args([])
            return th.apply_config_to_args(args, config)
        finally:
            sys.argv = orig

    def test_gemm_short_keys(self, th, parser):
        """Short keys (precision, batch_count) should work for batched_gemm."""
        config = {
            "benchmarks": [
                {"name": "batched_gemm", "precision": "float64", "batch_count": 32}
            ]
        }
        args = self._apply(th, parser, config)
        assert args.benchmark_list is not None
        assert len(args.benchmark_list) == 1
        bench = args.benchmark_list[0]
        assert bench['precision'] == 'float64'
        assert bench['batch_count'] == 32

    def test_gemm_full_keys(self, th, parser):
        """Full attribute keys (precision_gemm, batch_count_gemm) should work."""
        config = {
            "benchmarks": [
                {"name": "batched_gemm", "precision_gemm": "float64", "batch_count_gemm": 32}
            ]
        }
        args = self._apply(th, parser, config)
        assert args.benchmark_list is not None
        bench = args.benchmark_list[0]
        assert bench['precision_gemm'] == 'float64'
        assert bench['batch_count_gemm'] == 32

    def test_dual_gemm_entries_preserved(self, th, parser):
        """Two GEMM entries with different precisions should both be in benchmark_list."""
        config = {
            "benchmarks": [
                {"name": "batched_gemm", "precision_gemm": "float32"},
                {"name": "batched_gemm", "precision_gemm": "float64"},
            ]
        }
        args = self._apply(th, parser, config)
        assert len(args.benchmark_list) == 2
        assert args.benchmark_list[0]['precision_gemm'] == 'float32'
        assert args.benchmark_list[1]['precision_gemm'] == 'float64'

    def test_memory_traffic_pattern_in_config(self, th, parser):
        """memory_pattern key should be preserved in benchmark_list."""
        config = {
            "benchmarks": [
                {"name": "memory_traffic", "memory_pattern": "streaming"},
                {"name": "memory_traffic", "memory_pattern": "random"},
            ]
        }
        args = self._apply(th, parser, config)
        assert len(args.benchmark_list) == 2
        assert args.benchmark_list[0]['memory_pattern'] == 'streaming'
        assert args.benchmark_list[1]['memory_pattern'] == 'random'

    def test_disabled_benchmark_excluded(self, th, parser):
        """Benchmarks with enabled: false should not appear in benchmark_list."""
        config = {
            "benchmarks": [
                {"name": "batched_gemm", "enabled": True},
                {"name": "fft", "enabled": False},
            ]
        }
        args = self._apply(th, parser, config)
        assert len(args.benchmark_list) == 1
        assert args.benchmark_list[0]['name'] == 'batched_gemm'

    def test_platform_stress_config_loads(self, th, parser):
        """The platform-stress.yaml config should load and parse correctly."""
        import os
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config-examples', 'platform-stress.yaml')
        if not os.path.exists(config_path):
            pytest.skip("platform-stress.yaml not found")
        config = th.load_config(config_path)
        args = self._apply(th, parser, config)
        assert args.benchmark_list is not None
        # Should have 47 benchmarks (6 GEMM + 6 conv + 5 fft + 6 einsum +
        # 4 memory + 6 heat + 6 schrodinger + 4 atomic + 4 sparse)
        assert len(args.benchmark_list) == 47
        # First entries should be batched_gemm across precisions
        assert args.benchmark_list[0]['name'] == 'batched_gemm'
        assert args.benchmark_list[0].get('precision_gemm') == 'bfloat16'
        assert args.benchmark_list[2]['name'] == 'batched_gemm'
        assert args.benchmark_list[2].get('precision_gemm') == 'float32'


class TestStressFromConfig:
    """Tests for stress-test sizing reaching benchmarks that come from a config file (#30)."""

    def _apply(self, th, parser, config):
        """Helper: parse empty CLI, apply config, return args."""
        import sys
        orig = sys.argv
        sys.argv = ["torch-hammer.py"]
        try:
            args = parser.parse_args([])
            return th.apply_config_to_args(args, config)
        finally:
            sys.argv = orig

    def test_apply_stress_params_heat_enlarges_grid(self, th, parser):
        """apply_stress_params('heat') should grow the grid beyond the 128 default."""
        from unittest.mock import MagicMock
        args = parser.parse_args([])
        args.precision_heat = "float32"
        th.apply_stress_params(args, "heat", 8000.0, MagicMock())
        assert args.heat_grid_size > 128

    def test_apply_stress_params_gemm_sets_dims(self, th, parser):
        """apply_stress_params('gemm') should replace the default M/N/K and batch count."""
        from unittest.mock import MagicMock
        default_dims = (parser.get_default("m"), parser.get_default("n"), parser.get_default("k"))
        args = parser.parse_args([])
        args.precision_gemm = "float32"
        th.apply_stress_params(args, "gemm", 8000.0, MagicMock())
        assert args.m > 0
        assert args.batch_count_gemm > 0
        assert (args.m, args.n, args.k) != default_dims

    def test_stress_test_from_yaml_global_survives_with_benchmarks(self, th, parser):
        """stress_test: true under global: must survive when a benchmarks: list is present."""
        config = {
            "global": {"stress_test": True},
            "benchmarks": [{"name": "heat_equation", "precision": "float32"}],
        }
        args = self._apply(th, parser, config)
        assert args.stress_test is True
        assert len(args.benchmark_list) == 1

    def test_platform_stress_yaml_enables_stress_and_duration(self, th, parser):
        """platform-stress.yaml should turn on stress sizing and the 30s duration by itself."""
        import os
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config-examples', 'platform-stress.yaml')
        if not os.path.exists(config_path):
            pytest.skip("platform-stress.yaml not found")
        config = th.load_config(config_path)
        args = self._apply(th, parser, config)
        assert args.stress_test is True
        assert args.duration == 30


class TestMeltParsing:
    """Melt mode (TIM settling) flags and YAML keys."""

    def _apply(self, th, parser, config, cli=()):
        import sys
        orig = sys.argv
        sys.argv = ["torch-hammer.py", *cli]
        try:
            args = parser.parse_args(list(cli))
            return th.apply_config_to_args(args, config)
        finally:
            sys.argv = orig

    def test_melt_defaults(self, th, default_args):
        assert default_args.melt is False
        assert default_args.melt_only is False
        assert default_args.precision_melt == "float32"
        assert default_args.melt_burst_ms == th.MELT_BURST_MS == 140
        assert default_args.melt_cool_ms == th.MELT_COOL_MS == 100
        assert default_args.melt_max_duration == th.MELT_MAX_DURATION_S == 3600
        assert default_args.melt_min_duration == th.MELT_MIN_DURATION_S == 3600   # fixed dose: min == max
        assert default_args.melt_window_s == th.MELT_WINDOW_S == 60
        assert default_args.melt_bin_s == th.MELT_BIN_S == 10.0
        assert default_args.melt_plateau_cv_pct == th.MELT_PLATEAU_CV_PCT == 1.0
        assert default_args.melt_settled_pct == th.MELT_SETTLED_PCT == 98.0
        assert default_args.melt_target_gflops is None
        assert default_args.melt_temp_slope_c_per_min == th.MELT_TEMP_SLOPE_C_PER_MIN == 0.5
        assert default_args.melt_temp_window_s == th.MELT_TEMP_WINDOW_S == 300.0
        assert default_args.melt_temp_sibling_warn_c == th.MELT_TEMP_SIBLING_WARN_C == 5.0

    def test_melt_sizes_have_their_own_defaults(self, th, default_args):
        """Melt must heat a die; it does not inherit the characterization --m/--n/--k."""
        assert default_args.melt_m == default_args.melt_n == default_args.melt_k == th.MELT_MNK == 4096
        assert default_args.melt_batch_count == th.MELT_BATCH == 8

    def test_melt_pause_ms_is_gone(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--melt", "--melt-pause-ms", "10"])

    def test_melt_help_text_mentions_minutes_and_independence(self, parser):
        text = " ".join(parser.format_help().split())
        assert text.count("3600 = 60 min") == 2
        assert "independent of --m" in text

    def test_melt_and_tim_settle_share_a_destination(self, parser):
        assert parser.parse_args(["--melt"]).melt is True
        assert parser.parse_args(["--tim-settle"]).melt is True

    def test_melt_only(self, parser):
        args = parser.parse_args(["--melt", "--melt-only", "--batched-gemm"])
        assert args.melt_only is True
        assert args.batched_gemm is True

    def test_melt_tuning_flags(self, parser):
        args = parser.parse_args([
            "--melt", "--precision-melt", "float64",
            "--melt-m", "4096", "--melt-n", "2048", "--melt-k", "1024",
            "--melt-batch-count", "8", "--melt-burst-ms", "200",
            "--melt-cool-ms", "50", "--melt-max-duration", "600",
            "--melt-min-duration", "120", "--melt-window-s", "30",
            "--melt-plateau-cv-pct", "2.5", "--melt-settled-pct", "95",
            "--melt-target-gflops", "47000",
            "--melt-temp-slope-c-per-min", "0.3", "--melt-temp-window-s", "300",
            "--melt-temp-sibling-warn-c", "8",
        ])
        assert args.precision_melt == "float64"
        assert (args.melt_m, args.melt_n, args.melt_k, args.melt_batch_count) == (4096, 2048, 1024, 8)
        assert args.melt_burst_ms == 200
        assert args.melt_cool_ms == 50
        assert args.melt_max_duration == 600
        assert args.melt_min_duration == 120
        assert args.melt_window_s == 30
        assert args.melt_plateau_cv_pct == 2.5
        assert args.melt_settled_pct == 95
        assert args.melt_target_gflops == 47000
        assert args.melt_temp_slope_c_per_min == 0.3
        assert args.melt_temp_window_s == 300.0
        assert args.melt_temp_sibling_warn_c == 8.0

    def test_precision_melt_rejects_complex(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--precision-melt", "complex64"])

    def test_melt_dims_ignore_characterization_sizes(self, th, parser):
        args = parser.parse_args(["--m", "256", "--n", "128", "--k", "64", "--batch-count-gemm", "16"])
        assert th._melt_dims(args) == (8, 4096, 4096, 4096)
        args = parser.parse_args(["--m", "256", "--melt-m", "32", "--melt-batch-count", "2"])
        assert th._melt_dims(args) == (2, 32, 4096, 4096)

    def test_yaml_cool_ms_key(self, th):
        assert th._MELT_CONFIG_KEYS["cool_ms"] == "melt_cool_ms"
        assert th._MELT_CONFIG_KEYS["bin_s"] == "melt_bin_s"
        assert "pause_ms" not in th._MELT_CONFIG_KEYS

    def test_yaml_temperature_keys(self, th):
        assert th._MELT_CONFIG_KEYS["temp_slope_c_per_min"] == "melt_temp_slope_c_per_min"
        assert th._MELT_CONFIG_KEYS["temp_window_s"] == "melt_temp_window_s"
        assert th._MELT_CONFIG_KEYS["temp_sibling_warn_c"] == "melt_temp_sibling_warn_c"

    def test_yaml_melt_entry_sets_temperature_knobs(self, th, parser):
        args = parser.parse_args(["--melt"])
        th._apply_melt_config(args, {"name": "melt", "temp_slope_c_per_min": 0, "temp_window_s": 300,
                                     "temp_sibling_warn_c": 6})
        assert args.melt_temp_slope_c_per_min == 0
        assert args.melt_temp_window_s == 300
        assert args.melt_temp_sibling_warn_c == 6

    def test_yaml_melt_entry_lands_in_benchmark_list(self, th, parser):
        config = {"benchmarks": [
            {"name": "melt", "precision": "float64", "burst_ms": 200, "window_s": 30,
             "m": 4096, "batch_count": 4, "max_duration": 1200},
            {"name": "batched_gemm"},
        ]}
        args = self._apply(th, parser, config)
        names = [b["name"] for b in args.benchmark_list]
        assert names == ["melt", "batched_gemm"]
        melt = args.benchmark_list[0]
        assert melt["precision"] == "float64"
        assert melt["burst_ms"] == 200
        assert melt["window_s"] == 30

    def test_yaml_melt_name_is_valid(self, th, parser, capsys):
        self._apply(th, parser, {"benchmarks": [{"name": "melt"}]})
        assert "invalid name 'melt'" not in capsys.readouterr().out

    def test_yaml_global_melt_true_sets_flag(self, th, parser):
        args = self._apply(th, parser, {"global": {"melt": True}})
        assert args.melt is True

    def test_yaml_global_tim_settle_true_sets_flag(self, th, parser):
        args = self._apply(th, parser, {"global": {"tim_settle": True}})
        assert args.melt is True

    def test_yaml_global_melt_only_sets_flag(self, th, parser):
        args = self._apply(th, parser, {"global": {"melt_only": True}})
        assert args.melt_only is True


class TestMeltArgValidation:
    """Fail fast on contradictory melt flags (Don Norman H1/M5)."""

    class _ErrorParser:
        def error(self, message):
            raise ValueError(message)

    def _validate(self, th, parser, argv, caplog=None):
        args = parser.parse_args(argv)
        log = logging.getLogger("test_melt_validate")
        th._validate_melt_args(self._ErrorParser(), args, log)
        return args

    def test_min_duration_must_not_exceed_max(self, th, parser):
        with pytest.raises(ValueError) as exc:
            self._validate(th, parser, ["--melt", "--melt-min-duration", "5", "--melt-max-duration", "2"])
        assert "5" in str(exc.value) and "2" in str(exc.value)
        assert "--melt-min-duration" in str(exc.value) and "--melt-max-duration" in str(exc.value)

    def test_min_duration_equal_to_max_is_a_fixed_dose(self, th, parser):
        args = self._validate(th, parser, ["--melt", "--melt-min-duration", "2", "--melt-max-duration", "2"])
        assert args.melt_min_duration == args.melt_max_duration == 2
        args = self._validate(th, parser, ["--melt"])
        assert args.melt_min_duration == args.melt_max_duration == 3600

    @pytest.mark.parametrize("flag", ["--melt-window-s", "--melt-burst-ms", "--melt-max-duration"])
    def test_positive_only_flags(self, th, parser, flag):
        with pytest.raises(ValueError, match="must be > 0"):
            self._validate(th, parser, ["--melt", flag, "0"])

    def test_cool_ms_may_be_zero_but_not_negative(self, th, parser):
        self._validate(th, parser, ["--melt", "--melt-cool-ms", "0"])
        with pytest.raises(ValueError, match="--melt-cool-ms"):
            self._validate(th, parser, ["--melt", "--melt-cool-ms", "-1"])

    @pytest.mark.parametrize("value", ["0", "100.5", "-3"])
    def test_settled_pct_range(self, th, parser, value):
        with pytest.raises(ValueError, match="--melt-settled-pct"):
            self._validate(th, parser, ["--melt", "--melt-settled-pct", value])

    def test_bin_s_must_be_positive_and_fit_three_times_in_the_window(self, th, parser):
        with pytest.raises(ValueError, match="--melt-bin-s"):
            self._validate(th, parser, ["--melt", "--melt-bin-s", "0"])
        with pytest.raises(ValueError, match="--melt-bin-s"):
            self._validate(th, parser, ["--melt", "--melt-bin-s", "25", "--melt-window-s", "60"])
        self._validate(th, parser, ["--melt", "--melt-bin-s", "20", "--melt-window-s", "60"])

    def test_bin_s_fit_tolerates_floating_point_rounding(self, th, parser):
        # 0.2 * 3 == 0.6000000000000001 in binary floating point; the user typed exactly three bins.
        self._validate(th, parser, ["--melt", "--melt-bin-s", "0.2", "--melt-window-s", "0.6"])
        self._validate(th, parser, ["--melt", "--melt-bin-s", "0.1", "--melt-window-s", "0.3"])

    def test_plateau_cv_pct_must_not_be_negative(self, th, parser):
        with pytest.raises(ValueError, match="--melt-plateau-cv-pct"):
            self._validate(th, parser, ["--melt", "--melt-plateau-cv-pct", "-0.1"])

    @pytest.mark.parametrize("flag", ["--melt-temp-slope-c-per-min", "--melt-temp-sibling-warn-c"])
    def test_temperature_thresholds_may_be_zero_but_not_negative(self, th, parser, flag):
        self._validate(th, parser, ["--melt", flag, "0"])
        with pytest.raises(ValueError, match=f"{flag} must be >= 0"):
            self._validate(th, parser, ["--melt", flag, "-1"])

    def test_temp_window_must_be_positive(self, th, parser):
        with pytest.raises(ValueError, match="--melt-temp-window-s must be > 0"):
            self._validate(th, parser, ["--melt", "--melt-temp-window-s", "0"])

    @pytest.mark.parametrize("argv, expected", [
        (["--melt", "--melt-temp-window-s", "30"], 60.0),                 # explicit horizon shorter than the window
        (["--melt", "--melt-window-s", "400"], 400.0),                    # default 300 s horizon vs a longer window
    ])
    def test_temp_window_shorter_than_the_window_is_raised_to_it(self, th, parser, argv, expected, caplog):
        with caplog.at_level(logging.INFO, logger="test_melt_validate"):
            args = self._validate(th, parser, argv)
        assert args.melt_temp_window_s == expected
        assert any(r.message == f"--melt-temp-window-s raised to {expected:g} s to cover --melt-window-s"
                   for r in caplog.records)

    def test_temp_window_at_least_the_window_is_left_alone(self, th, parser, caplog):
        with caplog.at_level(logging.INFO, logger="test_melt_validate"):
            args = self._validate(th, parser, ["--melt", "--melt-temp-window-s", "60"])
        assert args.melt_temp_window_s == 60.0
        assert not any("raised" in r.message for r in caplog.records)

    def test_window_longer_than_min_duration_only_warns(self, th, parser, caplog):
        with caplog.at_level(logging.WARNING, logger="test_melt_validate"):
            self._validate(th, parser, ["--melt", "--melt-window-s", "400", "--melt-min-duration", "300"])
        assert any("window extends before the min-duration gate" in r.message for r in caplog.records)

    def test_melt_only_implies_melt(self, th, parser, caplog):
        with caplog.at_level(logging.INFO, logger="test_melt_validate"):
            args = self._validate(th, parser, ["--melt-only"])
        assert args.melt is True
        assert any("--melt-only implies --melt" in r.message for r in caplog.records)

    def test_duration_does_not_apply_to_melt(self, th, parser, caplog):
        with caplog.at_level(logging.INFO, logger="test_melt_validate"):
            self._validate(th, parser, ["--melt", "--duration", "60"])
        assert any("--duration does not apply to melt" in r.message for r in caplog.records)

    def test_valid_defaults_pass_silently(self, th, parser, caplog):
        with caplog.at_level(logging.INFO, logger="test_melt_validate"):
            self._validate(th, parser, ["--melt"])
        assert caplog.records == []

    def test_no_melt_is_a_no_op(self, th, parser):
        self._validate(th, parser, ["--batched-gemm", "--melt-min-duration", "5", "--melt-max-duration", "2"])
