# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for utility functions: Timer, VerbosePrinter, helper functions.
"""
import time
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestGflopsCalculation:
    """Tests for GFLOP/s calculation helper."""
    
    def test_simple_calculation(self, th):
        """Basic GFLOP/s calculation should work correctly."""
        # 1e12 flops in 1 second = 1000 GFLOP/s
        result = th.gflops(1e12, 1.0)
        assert result == 1000.0
    
    def test_scaled_calculation(self, th):
        """Scaled values should calculate correctly."""
        # 2e12 flops in 2 seconds = 1000 GFLOP/s
        result = th.gflops(2e12, 2.0)
        assert result == 1000.0
    
    def test_small_time(self, th):
        """Small time values should produce high GFLOP/s."""
        # 1e12 flops in 0.1 seconds = 10000 GFLOP/s
        result = th.gflops(1e12, 0.1)
        assert abs(result - 10000.0) < 0.01
    
    def test_zero_time_raises(self, th):
        """Zero time should raise ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            th.gflops(1e12, 0.0)


class TestTimer:
    """Tests for the Timer context manager."""
    
    def test_cpu_timing(self, th):
        """Test Timer measures elapsed time on CPU."""
        import torch
        # Pass a torch.device object, not a string
        timer = th.Timer(device=torch.device('cpu'))
        with timer:
            time.sleep(0.1)
        # Allow generous tolerance for CI runners (sleep(0.1) + overhead)
        assert timer.elapsed >= 0.09  # At least ~90ms
        assert timer.elapsed < 0.3    # But less than 300ms (CI can be slow)
 
    def test_cpu_timer_attributes(self, th):
        """CPU Timer should have correct attributes."""
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        assert timer.cuda is False
    
    def test_cuda_timer_attributes(self, th):
        """CUDA Timer should have correct attributes when CUDA available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda:0")
        timer = th.Timer(device)
        
        assert timer.cuda is True
        assert hasattr(timer, 's')
        assert hasattr(timer, 'e')
    
    def test_context_manager_protocol(self, th):
        """Timer should implement context manager protocol."""
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        # Should have __enter__ and __exit__
        assert hasattr(timer, '__enter__')
        assert hasattr(timer, '__exit__')
    
    def test_elapsed_only_available_after_exit(self, th):
        """elapsed attribute should only be set after context exit."""
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        # Before entering, no elapsed
        assert not hasattr(timer, 'elapsed') or timer.elapsed is None
        
        timer.__enter__()
        # During, might not have elapsed yet
        time.sleep(0.01)
        timer.__exit__(None, None, None)
        
        # After exit, should have elapsed
        assert hasattr(timer, 'elapsed')
        assert timer.elapsed > 0


class TestVerbosePrinter:
    """Tests for the VerbosePrinter class."""
    
    def test_initialization(self, th):
        """VerbosePrinter should initialize correctly."""
        logger = logging.getLogger("test")
        schema = ["vendor", "model", "device_id", "sm_util", "power_W"]
        
        printer = th.VerbosePrinter(logger, schema, gpu_id=0)
        
        assert printer.log == logger
        assert printer.gpu_id == 0
        assert "vendor" in printer.schema
        assert "model" in printer.schema
    
    def test_schema_reordering(self, th):
        """VerbosePrinter should put vendor/model first in schema."""
        logger = logging.getLogger("test")
        schema = ["sm_util", "power_W", "vendor", "model", "device_id"]
        
        printer = th.VerbosePrinter(logger, schema, gpu_id=0)
        
        # vendor and model should be first
        assert printer.schema[0] == "vendor"
        assert printer.schema[1] == "model"
    
    def test_emit_logs_header_first_time(self, th):
        """emit() should log header on first call."""
        mock_logger = MagicMock()
        schema = ["vendor", "model", "sm_util"]
        
        printer = th.VerbosePrinter(mock_logger, schema, gpu_id=0)
        
        tel_data = {"vendor": "TEST", "model": "TestGPU", "sm_util": 95}
        printer.emit(0, "gemm", "float32", "gflops", 1000.0, tel_data)
        
        # Should have called info twice (header + data)
        assert mock_logger.info.call_count == 2
    
    def test_emit_includes_all_fields(self, th):
        """emit() should include all schema fields in output."""
        mock_logger = MagicMock()
        schema = ["vendor", "model", "sm_util", "power_W"]
        
        printer = th.VerbosePrinter(mock_logger, schema, gpu_id=0)
        
        tel_data = {"vendor": "NVIDIA", "model": "A100", "sm_util": 95, "power_W": 300.5}
        printer.emit(0, "gemm", "float32", "gflops", 1000.0, tel_data)
        
        # Get the data line (second call)
        data_call = mock_logger.info.call_args_list[1]
        data_line = data_call[0][0]
        
        assert "NVIDIA" in data_line
        assert "A100" in data_line
        assert "95" in data_line


class TestFormatTelemetryCompact:
    """Tests for telemetry formatting helper."""
    
    def test_basic_formatting(self, th):
        """Basic telemetry data should format correctly."""
        tel_data = {
            "device_id": 0,
            "vendor": "NVIDIA",  # Indicates GPU device
            "sm_util": 95,
            "mem_bw_util": 80,
            "temp_gpu_C": 65,
            "power_W": 250.5,
            "gpu_clock": 1500,
        }
        
        result = th._format_telemetry_compact(tel_data)
        
        assert "GPU0" in result
        assert "SM:95%" in result
        assert "Temp:65°C" in result
    
    def test_handles_missing_fields(self, th):
        """Formatter should handle missing fields gracefully."""
        tel_data = {"device_id": 1, "vendor": "NVIDIA"}
        
        # Should not raise
        result = th._format_telemetry_compact(tel_data)
        
        assert "GPU1" in result
    
    def test_cpu_device_formatting(self, th):
        """CPU-only telemetry should not say GPU."""
        tel_data = {
            "device_id": 0,
            "vendor": "CPU",
            "model": "x86_64",
        }
        
        result = th._format_telemetry_compact(tel_data)
        
        # Should show model name, not GPU0
        assert "x86_64" in result
        assert "GPU" not in result
    
    def test_memory_formatting(self, th):
        """Memory usage should format with GB and percentage."""
        tel_data = {
            "device_id": 0,
            "mem_used_MB": 20480,  # 20 GB
            "mem_total_MB": 40960,  # 40 GB
        }
        
        result = th._format_telemetry_compact(tel_data)
        
        assert "Mem:" in result
        assert "GB" in result


class TestValidatePerformance:
    """Tests for hardware performance validation."""
    
    def test_no_baselines_returns_valid(self, th):
        """With no baselines, validation should return valid."""
        result = th.validate_performance(
            model_name="Unknown GPU",
            benchmark="batched_gemm",
            dtype="float32",
            measured_value=50000.0,  # 50 TFLOP/s
            baselines={}
        )
        
        assert result['valid'] is True
        assert result['baseline'] is None
    
    def test_with_target_baselines(self, th):
        """Target-based baselines should validate correctly."""
        baselines = {
            "TestGPU": {
                "benchmarks": {
                    "batched_gemm": {
                        "float32": {
                            "target_gflops": 40000.0,
                            "min_efficiency": 80.0
                        }
                    }
                }
            }
        }
        
        result = th.validate_performance(
            model_name="TestGPU 80GB",
            benchmark="batched_gemm",
            dtype="float32",
            measured_value=35000.0,  # 35 TFLOP/s = 87.5% of target
            unit="gflop/s",
            baselines=baselines
        )
        
        assert result['valid'] is True
        assert result['efficiency'] is not None
        assert 85 < result['efficiency'] < 90
    
    def test_low_efficiency_triggers_warning(self, th):
        """Low efficiency should trigger warning."""
        baselines = {
            "TestGPU": {
                "benchmarks": {
                    "batched_gemm": {
                        "float32": {
                            "target_gflops": 40000.0,
                            "min_efficiency": 80.0
                        }
                    }
                }
            }
        }
        
        result = th.validate_performance(
            model_name="TestGPU 80GB",
            benchmark="batched_gemm",
            dtype="float32",
            measured_value=20000.0,  # 20 TFLOP/s = 50% of target
            unit="gflop/s",
            baselines=baselines
        )
        
        assert result['valid'] is False
        assert result['warning'] is not None
        assert "below target" in result['warning'].lower()


class TestLoadHardwareBaselines:
    """Tests for baseline file loading."""
    
    def test_load_nonexistent_file(self, th):
        """Loading nonexistent file should return empty dict."""
        result = th.load_hardware_baselines("/nonexistent/path/file.json")
        assert result == {}
    
    def test_load_unknown_format(self, th):
        """Unknown file format should return empty dict."""
        result = th.load_hardware_baselines("/some/path/file.txt")
        assert result == {}


class TestSummaryTableGrouping:
    """Tests for the multi-GPU summary table grouping logic.
    
    Verifies that benchmark results are grouped correctly when
    multiple config entries share the same name+dtype (e.g.,
    memory_traffic streaming vs random, both float32).
    """

    def _make_gpu_result(self, gpu_index, benchmarks, serial='SN123'):
        """Helper to create a mock GPU result dict."""
        return {
            'gpu_index': gpu_index,
            'serial': serial,
            'telemetry_stats': {},
            'benchmarks': benchmarks,
        }

    def _make_bench(self, name, dtype, mean, unit='GFLOP/s', params=None):
        """Helper to create a mock benchmark result dict."""
        p = {'dtype': dtype}
        if params:
            p.update(params)
        return {
            'name': name,
            'mean': mean,
            'min': mean * 0.95,
            'max': mean * 1.05,
            'unit': unit,
            'params': p,
            'telemetry': {},
        }

    def test_same_name_dtype_different_index_separate_groups(self):
        """Two benchmarks with same name+dtype but different positions should be separate groups."""
        # Simulate two memory_traffic runs (streaming vs random), both float32
        bench_streaming = self._make_bench('Memory Traffic', 'float32', 500.0, 'GB/s',
                                           params={'pattern': 'streaming'})
        bench_random = self._make_bench('Memory Traffic', 'float32', 300.0, 'GB/s',
                                        params={'pattern': 'random'})
        
        # Group by bench_idx as the fix does
        benchmark_groups = {}
        results = [
            self._make_gpu_result(0, [bench_streaming, bench_random]),
            self._make_gpu_result(1, [bench_streaming, bench_random]),
        ]
        
        for result in results:
            gpu_idx = result['gpu_index']
            for bench_idx, bench in enumerate(result.get('benchmarks', [])):
                if not bench:
                    continue
                dtype = bench.get('params', {}).get('dtype', 'unknown')
                test_key = f"{bench_idx:03d}_{bench['name']}_{dtype}"
                if test_key not in benchmark_groups:
                    benchmark_groups[test_key] = {'results': []}
                benchmark_groups[test_key]['results'].append({'gpu': gpu_idx})
        
        # Should have 2 groups, each with 2 GPU entries
        assert len(benchmark_groups) == 2
        for group in benchmark_groups.values():
            assert len(group['results']) == 2

    def test_different_dtype_separate_groups(self):
        """Two GEMM entries with different dtypes should be separate groups."""
        bench_fp32 = self._make_bench('Batched GEMM', 'float32', 15000.0)
        bench_fp64 = self._make_bench('Batched GEMM', 'float64', 7500.0)
        
        benchmark_groups = {}
        results = [self._make_gpu_result(0, [bench_fp32, bench_fp64])]
        
        for result in results:
            for bench_idx, bench in enumerate(result.get('benchmarks', [])):
                if not bench:
                    continue
                dtype = bench.get('params', {}).get('dtype', 'unknown')
                test_key = f"{bench_idx:03d}_{bench['name']}_{dtype}"
                if test_key not in benchmark_groups:
                    benchmark_groups[test_key] = {'results': []}
                benchmark_groups[test_key]['results'].append({'gpu': result['gpu_index']})
        
        assert len(benchmark_groups) == 2

    def test_unique_gpu_count(self):
        """Aggregate should count unique GPUs, not total result entries."""
        results_list = [
            {'gpu': 0, 'performance': 100.0},
            {'gpu': 1, 'performance': 100.0},
        ]
        unique_gpus = len(set(r['gpu'] for r in results_list))
        assert unique_gpus == 2

    def test_display_label_includes_pattern(self):
        """Display label for memory_traffic should include pattern info."""
        bench = self._make_bench('Memory Traffic', 'float32', 500.0, 'GB/s',
                                 params={'pattern': 'streaming'})
        params = bench.get('params', {})
        dtype = params.get('dtype', 'unknown')
        extra_parts = []
        if 'pattern' in params:
            extra_parts.append(params['pattern'])
        display_label = f"{bench['name']} ({dtype})" if not extra_parts else f"{bench['name']} ({dtype}, {', '.join(extra_parts)})"
        
        assert display_label == "Memory Traffic (float32, streaming)"

    def test_display_label_without_pattern(self):
        """Display label for GEMM should just show name and dtype."""
        bench = self._make_bench('Batched GEMM', 'float64', 7500.0)
        params = bench.get('params', {})
        dtype = params.get('dtype', 'unknown')
        extra_parts = []
        if 'pattern' in params:
            extra_parts.append(params['pattern'])
        display_label = f"{bench['name']} ({dtype})" if not extra_parts else f"{bench['name']} ({dtype}, {', '.join(extra_parts)})"
        
        assert display_label == "Batched GEMM (float64)"


class TestPhysicalCoreCount:
    """Thread count must follow the CPUs the process may use, not the whole node."""

    @staticmethod
    def _node(root, sockets=2, cores=4):
        n = sockets * cores
        for cpu in range(2 * n):  # SMT: cpu and cpu+n share a core
            core = cpu % n
            topo = root / f"cpu{cpu}" / "topology"
            topo.mkdir(parents=True)
            (topo / "physical_package_id").write_text(f"{core // cores}\n")
            (topo / "core_id").write_text(f"{core % cores}\n")
        return str(root)

    def test_whole_node(self, th, tmp_path):
        cpu_dir = self._node(tmp_path)
        assert th._get_physical_core_count_early(cpu_dir, allowed=set(range(16))) == 8

    def test_one_cpu_slurm_step_gets_one_core(self, th, tmp_path):
        cpu_dir = self._node(tmp_path)
        assert th._get_physical_core_count_early(cpu_dir, allowed={0, 8}) == 1

    def test_partial_allocation_counts_cores_not_threads(self, th, tmp_path):
        cpu_dir = self._node(tmp_path)
        assert th._get_physical_core_count_early(cpu_dir, allowed={0, 1, 2, 8, 9, 10, 4}) == 4

    def test_import_time_count_is_whole_node(self, th, tmp_path, monkeypatch):
        """OMP_NUM_THREADS is set at import for every run, GPU runs included: keep the node count."""
        cpu_dir = self._node(tmp_path)
        monkeypatch.setattr(th.os, "sched_getaffinity", lambda pid: {3, 11}, raising=False)
        monkeypatch.setattr(th, "_STARTUP_CPU_AFFINITY", frozenset({3, 11}))
        assert th._get_physical_core_count_early(cpu_dir) == 8

    def test_cpu_runs_use_startup_affinity_not_current_thread_mask(self, th, tmp_path, monkeypatch):
        cpu_dir = self._node(tmp_path)
        # OpenMP has pinned the main thread to one core; the startup mask still covers the node
        monkeypatch.setattr(th.os, "sched_getaffinity", lambda pid: {0, 8}, raising=False)
        monkeypatch.setattr(th, "_STARTUP_CPU_AFFINITY", frozenset(range(16)))
        assert th.get_physical_core_count(cpu_dir) == 8

    def test_cpu_runs_respect_restricted_startup_affinity(self, th, tmp_path, monkeypatch):
        cpu_dir = self._node(tmp_path)
        monkeypatch.setattr(th, "_STARTUP_CPU_AFFINITY", frozenset({3, 11}))
        assert th.get_physical_core_count(cpu_dir) == 1
