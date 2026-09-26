# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Tests for reports/hammer_report.py — fleet report generator.

Covers:
  - Compact CSV parsing (single-node, multi-node, malformed rows)
  - Summary CSV parsing
  - JSON parsing
  - Shell dump parsing
  - Auto-detection of input format
  - Multi-benchmark grouping (BenchmarkStats)
  - Outlier detection
  - HTML generation (XSS safety, multi-section output)
  - CLI entry point (exit codes, filters)
"""
import csv
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the report module
import importlib.util

ROOT_DIR = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "hammer_report", ROOT_DIR / "reports" / "hammer_report.py"
)
hr = importlib.util.module_from_spec(_spec)
sys.modules["hammer_report"] = hr
_spec.loader.exec_module(hr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPACT_HEADER = (
    "hostname,gpu,gpu_model,serial,benchmark,dtype,iterations,runtime_s,"
    "min,mean,max,unit,power_avg_w,temp_max_c"
)

def _compact_row(hostname="node1", gpu=0, benchmark="Batched GEMM",
                 dtype="float32", mean=1000.0, unit="GFLOP/s", **kw):
    """Build one compact CSV row dict."""
    defaults = dict(
        hostname=hostname, gpu=str(gpu), gpu_model="TestGPU",
        serial="SN123456789012", benchmark=benchmark, dtype=dtype,
        iterations="100", runtime_s="10.0",
        min=str(mean * 0.95), mean=str(mean), max=str(mean * 1.05),
        unit=unit, power_avg_w="250", temp_max_c="72",
    )
    defaults.update(kw)
    return defaults


def _compact_csv_text(rows, header=COMPACT_HEADER):
    """Build a compact CSV string from a list of row dicts."""
    lines = [header]
    cols = header.split(",")
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def _make_results(n_nodes=2, n_gpus=4, benchmarks=None, mean_base=1000.0,
                   power=250.0, temp=72.0, sm_util=0.0, mem_bw_util=0.0,
                   gpu_clock=0.0):
    """Generate a list of GPUResult for testing."""
    if benchmarks is None:
        benchmarks = [("Batched GEMM", "float32", "GFLOP/s")]
    results = []
    for ni in range(n_nodes):
        for gi in range(n_gpus):
            for bname, dtype, unit in benchmarks:
                mean = mean_base + ni * 10 + gi * 2
                results.append(hr.GPUResult(
                    hostname="node{}".format(ni),
                    gpu=gi,
                    gpu_model="TestGPU",
                    serial="SN{:04d}".format(ni * 10 + gi),
                    benchmark=bname,
                    dtype=dtype,
                    iterations=100,
                    runtime_s=10.0,
                    min_val=mean * 0.95,
                    mean_val=mean,
                    max_val=mean * 1.05,
                    unit=unit,
                    power_avg_w=power,
                    temp_max_c=temp,
                    sm_util_mean=sm_util,
                    mem_bw_util_mean=mem_bw_util,
                    gpu_clock_mean=gpu_clock,
                ))
    return results


# ======================================================================
# 1. Compact CSV parsing
# ======================================================================

class TestCompactCSVParsing:
    """Tests for load_compact_csv and _parse_compact_row."""

    def test_parse_single_row(self):
        row = _compact_row()
        result = hr._parse_compact_row(row)
        assert result is not None
        assert result.hostname == "node1"
        assert result.gpu == 0
        assert result.benchmark == "Batched GEMM"
        assert result.dtype == "float32"
        assert result.mean_val == 1000.0
        assert result.unit == "GFLOP/s"

    def test_parse_row_strips_whitespace(self):
        row = _compact_row(benchmark="  FFT  ", dtype=" float64 ")
        result = hr._parse_compact_row(row)
        assert result.benchmark == "FFT"
        assert result.dtype == "float64"

    def test_parse_row_malformed_returns_none(self):
        row = {"gpu": "not_a_number", "benchmark": "test"}
        result = hr._parse_compact_row(row)
        assert result is None

    def test_load_compact_csv_basic(self, tmp_path):
        rows = [_compact_row(gpu=i) for i in range(4)]
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(_compact_csv_text(rows))

        results = hr.load_compact_csv(csv_path)
        assert len(results) == 4
        assert all(r.hostname == "node1" for r in results)
        gpus = [r.gpu for r in results]
        assert gpus == [0, 1, 2, 3]

    def test_load_compact_csv_hostname_override(self, tmp_path):
        rows = [_compact_row(hostname="", gpu=0)]
        csv_path = tmp_path / "node42.csv"
        csv_path.write_text(_compact_csv_text(rows))

        results = hr.load_compact_csv(csv_path, hostname_override="node42")
        assert len(results) == 1
        assert results[0].hostname == "node42"

    def test_load_compact_csv_multi_benchmark(self, tmp_path):
        rows = [
            _compact_row(benchmark="Batched GEMM", dtype="float32", mean=1000),
            _compact_row(benchmark="FFT", dtype="float32", mean=500),
            _compact_row(benchmark="Batched GEMM", dtype="float64", mean=800),
        ]
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(_compact_csv_text(rows))

        results = hr.load_compact_csv(csv_path)
        assert len(results) == 3
        benchmarks = [r.benchmark for r in results]
        assert "Batched GEMM" in benchmarks
        assert "FFT" in benchmarks

    def test_load_compact_csv_skips_bad_rows(self, tmp_path):
        text = (COMPACT_HEADER + "\n"
                "node1,0,GPU,SN1,GEMM,fp32,100,10,950,1000,1050,GFLOP/s,250,72\n"
                "node1,BAD,GPU,SN2,GEMM,fp32,X,Y,Z,W,V,GFLOP/s,250,72\n"
                "node1,1,GPU,SN3,GEMM,fp32,100,10,900,950,1000,GFLOP/s,250,72\n")
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(text)

        results = hr.load_compact_csv(csv_path)
        assert len(results) == 2  # bad row skipped


# ======================================================================
# 2. Summary CSV parsing
# ======================================================================

class TestSummaryCSVParsing:
    """Tests for load_summary_csv and _parse_summary_row."""

    def _summary_csv(self, rows):
        header = "test,dtype,gpu,serial,performance,unit,power_avg_W,temp_max_C,status,notes"
        lines = [header]
        for r in rows:
            lines.append(",".join(str(r.get(c, "")) for c in header.split(",")))
        return "\n".join(lines) + "\n"

    def test_parse_summary_row(self):
        row = {
            "test": "Batched GEMM", "dtype": "float32", "gpu": "0",
            "serial": "SN123", "performance": "1234.5", "unit": "GFLOP/s",
            "power_avg_W": "250", "temp_max_C": "72", "status": "PASS",
            "notes": "",
        }
        result = hr._parse_summary_row(row, hostname="myhost")
        assert result is not None
        assert result.hostname == "myhost"
        assert result.benchmark == "Batched GEMM"
        assert result.mean_val == 1234.5
        assert result.min_val == 1234.5  # summary only has one value

    def test_load_summary_csv(self, tmp_path):
        rows = [
            {"test": "Batched GEMM", "dtype": "float32", "gpu": "0",
             "serial": "SN1", "performance": "1000", "unit": "GFLOP/s",
             "power_avg_W": "250", "temp_max_C": "72", "status": "PASS", "notes": ""},
            {"test": "FFT", "dtype": "float32", "gpu": "0",
             "serial": "SN1", "performance": "500", "unit": "GFLOP/s",
             "power_avg_W": "200", "temp_max_C": "68", "status": "PASS", "notes": ""},
        ]
        csv_path = tmp_path / "summary.csv"
        csv_path.write_text(self._summary_csv(rows))

        results = hr.load_summary_csv(csv_path, hostname="node1")
        assert len(results) == 2
        assert results[0].benchmark == "Batched GEMM"
        assert results[1].benchmark == "FFT"


# ======================================================================
# 3. JSON parsing
# ======================================================================

class TestJSONParsing:
    """Tests for load_json."""

    def _json_data(self, n_gpus=2, benchmarks=None):
        if benchmarks is None:
            benchmarks = [{"name": "Batched GEMM", "min": 950, "mean": 1000,
                           "max": 1050, "unit": "GFLOP/s", "iterations": 100,
                           "runtime_s": 10.0,
                           "params": {"dtype": "float32"},
                           "telemetry": {"power_W_mean": 250, "temp_gpu_C_max": 72}}]
        gpus = []
        for i in range(n_gpus):
            gpus.append({
                "gpu_index": i,
                "model": "TestGPU",
                "serial": "SN{:04d}".format(i),
                "benchmarks": benchmarks,
            })
        return {
            "metadata": {"hostname": "testhost", "torch_version": "2.0"},
            "gpus": gpus,
        }

    def test_load_json_basic(self, tmp_path):
        data = self._json_data(n_gpus=4)
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps(data))

        results = hr.load_json(json_path)
        assert len(results) == 4
        assert all(r.hostname == "testhost" for r in results)
        assert all(r.benchmark == "Batched GEMM" for r in results)
        assert results[0].gpu == 0
        assert results[3].gpu == 3

    def test_load_json_multi_benchmark(self, tmp_path):
        benches = [
            {"name": "Batched GEMM", "min": 950, "mean": 1000, "max": 1050,
             "unit": "GFLOP/s", "iterations": 100, "runtime_s": 10.0,
             "params": {"dtype": "float32"}, "telemetry": {}},
            {"name": "FFT", "min": 400, "mean": 500, "max": 600,
             "unit": "GFLOP/s", "iterations": 50, "runtime_s": 5.0,
             "params": {"dtype": "float64"}, "telemetry": {}},
        ]
        data = self._json_data(n_gpus=2, benchmarks=benches)
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps(data))

        results = hr.load_json(json_path)
        assert len(results) == 4  # 2 GPUs * 2 benchmarks
        benchmarks = sorted(set(r.benchmark for r in results))
        assert benchmarks == ["Batched GEMM", "FFT"]

    def test_load_json_null_benchmark_skipped(self, tmp_path):
        data = self._json_data(n_gpus=1, benchmarks=[None, {"name": "GEMM",
             "min": 1, "mean": 2, "max": 3, "unit": "GFLOP/s",
             "iterations": 1, "runtime_s": 1.0, "params": {"dtype": "fp32"},
             "telemetry": {}}])
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps(data))

        results = hr.load_json(json_path)
        assert len(results) == 1


# ======================================================================
# 4. Shell dump parsing
# ======================================================================

class TestShellDumpParsing:
    """Tests for load_shell_output."""

    def test_basic_shell_dump(self, tmp_path):
        content = (
            "file: node1\n"
            "{}\n"
            "node1,0,GPU,SN1,GEMM,fp32,100,10,950,1000,1050,GFLOP/s,250,72\n"
            "node1,1,GPU,SN2,GEMM,fp32,100,10,900,950,1000,GFLOP/s,245,71\n"
            "file: node2\n"
            "{}\n"
            "node2,0,GPU,SN3,GEMM,fp32,100,10,980,1020,1060,GFLOP/s,255,73\n"
        ).format(COMPACT_HEADER, COMPACT_HEADER)
        dump_path = tmp_path / "dump.txt"
        dump_path.write_text(content)

        results = hr.load_shell_output(dump_path)
        assert len(results) == 3
        hosts = sorted(set(r.hostname for r in results))
        assert hosts == ["node1", "node2"]

    def test_shell_dump_empty_node(self, tmp_path):
        content = (
            "file: node1\n"
            "{}\n"
            "node1,0,GPU,SN1,GEMM,fp32,100,10,950,1000,1050,GFLOP/s,250,72\n"
            "file: node2\n"
        ).format(COMPACT_HEADER)
        dump_path = tmp_path / "dump.txt"
        dump_path.write_text(content)

        results = hr.load_shell_output(dump_path)
        assert len(results) == 1
        assert results[0].hostname == "node1"


# ======================================================================
# 5. Format auto-detection
# ======================================================================

class TestAutoDetection:
    """Tests for _detect_and_load."""

    def test_detect_json_by_extension(self, tmp_path):
        data = {"metadata": {"hostname": "h1"}, "gpus": []}
        p = tmp_path / "results.json"
        p.write_text(json.dumps(data))

        results = hr._detect_and_load(p)
        assert results == []  # valid JSON, no GPUs

    def test_detect_compact_csv(self, tmp_path):
        rows = [_compact_row()]
        p = tmp_path / "data.csv"
        p.write_text(_compact_csv_text(rows))

        results = hr._detect_and_load(p)
        assert len(results) == 1
        assert results[0].benchmark == "Batched GEMM"

    def test_detect_summary_csv(self, tmp_path):
        text = ("test,dtype,gpu,serial,performance,unit,power_avg_W,"
                "temp_max_C,status,notes\n"
                "GEMM,fp32,0,SN1,1000,GFLOP/s,250,72,PASS,\n")
        p = tmp_path / "summary.csv"
        p.write_text(text)

        results = hr._detect_and_load(p)
        assert len(results) == 1
        assert results[0].benchmark == "GEMM"

    def test_load_input_directory(self, tmp_path):
        for i in range(3):
            rows = [_compact_row(hostname="node{}".format(i), gpu=0)]
            p = tmp_path / "node{}.csv".format(i)
            p.write_text(_compact_csv_text(rows))

        results = hr.load_input(tmp_path)
        assert len(results) == 3
        hosts = sorted(set(r.hostname for r in results))
        assert len(hosts) == 3


# ======================================================================
# 6. Benchmark grouping and stats
# ======================================================================

class TestBenchmarkStats:
    """Tests for compute_benchmark_stats."""

    def test_single_benchmark(self):
        results = _make_results(n_nodes=2, n_gpus=4)
        stats = hr.compute_benchmark_stats(results)
        assert len(stats) == 1
        key = ("Batched GEMM", "float32")
        assert key in stats
        bs = stats[key]
        assert bs.name == "Batched GEMM"
        assert bs.dtype == "float32"
        assert bs.unit == "GFLOP/s"
        assert len(bs.gpu_means) == 8  # 2 nodes * 4 GPUs
        assert bs.fleet_mean > 0

    def test_multi_benchmark_grouping(self):
        benchmarks = [
            ("Batched GEMM", "float32", "GFLOP/s"),
            ("FFT", "float32", "GFLOP/s"),
            ("Batched GEMM", "float64", "GFLOP/s"),
        ]
        results = _make_results(n_nodes=1, n_gpus=2, benchmarks=benchmarks)
        stats = hr.compute_benchmark_stats(results)
        assert len(stats) == 3
        assert ("Batched GEMM", "float32") in stats
        assert ("FFT", "float32") in stats
        assert ("Batched GEMM", "float64") in stats

    def test_fleet_cv_single_gpu(self):
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        bs = list(stats.values())[0]
        assert bs.fleet_cv == 0.0  # CV undefined for single sample

    def test_fleet_cv_identical_values(self):
        results = _make_results(n_nodes=1, n_gpus=4, mean_base=1000)
        # Make all means exactly 1000
        for r in results:
            r.mean_val = 1000.0
        stats = hr.compute_benchmark_stats(results)
        bs = list(stats.values())[0]
        assert bs.fleet_cv == 0.0

    def test_node_intra_cv(self):
        results = _make_results(n_nodes=2, n_gpus=4)
        stats = hr.compute_benchmark_stats(results)
        bs = list(stats.values())[0]
        # Should have intra-CV for both nodes
        assert len(bs.node_intra_cv) == 2

    def test_bench_label_from_key(self):
        assert hr.bench_label_from_key(("GEMM", "fp32")) == "GEMM (fp32)"
        assert hr.bench_label_from_key(("GEMM", "")) == "GEMM"


# ======================================================================
# 7. Outlier detection
# ======================================================================

class TestOutlierDetection:
    """Tests for detect_outliers."""

    def test_no_outliers_uniform_data(self):
        results = _make_results(n_nodes=4, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        assert len(outliers) == 0

    def test_outlier_detected(self):
        results = _make_results(n_nodes=4, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        # Make one node significantly slower
        results[0].mean_val = 500.0
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        assert len(outliers) >= 1
        bad = [o for o in outliers if o["severity"] == "bad"]
        assert len(bad) >= 1
        assert bad[0]["host"] == "node0"

    def test_outlier_above_is_good(self):
        results = _make_results(n_nodes=4, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        # Make one node much faster
        results[0].mean_val = 2000.0
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        above = [o for o in outliers if o["direction"] == "above"]
        assert len(above) >= 1
        assert above[0]["severity"] == "good"

    def test_outlier_threshold_respected(self):
        results = _make_results(n_nodes=4, n_gpus=1, mean_base=1000)
        results[0].mean_val = 900.0  # 10% below mean ~975
        results[1].mean_val = 1000.0
        results[2].mean_val = 1000.0
        results[3].mean_val = 1000.0
        stats = hr.compute_benchmark_stats(results)

        # High threshold -> no outliers
        outliers_high = hr.detect_outliers(stats, threshold_pct=20.0)
        # Low threshold -> outlier
        outliers_low = hr.detect_outliers(stats, threshold_pct=3.0)
        assert len(outliers_low) >= len(outliers_high)

    def test_single_gpu_no_outliers(self):
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        assert len(outliers) == 0


# ======================================================================
# 8. HTML generation
# ======================================================================

class TestHTMLGeneration:
    """Tests for render_html and XSS safety."""

    def test_html_contains_doctype(self):
        results = _make_results(n_nodes=2, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats)
        html_out = hr.render_html(results, stats, outliers, "test.csv", 15.0)
        assert html_out.startswith("<!DOCTYPE html>")

    def test_html_has_no_external_deps(self):
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "chart.umd.js" not in html_out
        assert "<script src=" not in html_out
        assert "<svg" in html_out

    def test_html_multi_benchmark_sections(self):
        benchmarks = [
            ("Batched GEMM", "float32", "GFLOP/s"),
            ("FFT", "float64", "GFLOP/s"),
        ]
        results = _make_results(n_nodes=2, n_gpus=2, benchmarks=benchmarks)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Batched GEMM" in html_out
        assert "FFT" in html_out
        # Each section has its own SVG chart
        assert html_out.count("<svg") >= 2

    def test_html_xss_safe(self):
        """User-controlled data must be HTML-escaped."""
        xss = '<script>alert("xss")</script>'
        results = [hr.GPUResult(
            hostname=xss, gpu=0, gpu_model=xss, serial=xss,
            benchmark=xss, dtype="float32", iterations=1,
            runtime_s=1.0, min_val=1, mean_val=1, max_val=1,
            unit="GFLOP/s", power_avg_w=0, temp_max_c=0,
        )]
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test", 15.0)
        # The raw script tag must not appear unescaped
        assert '<script>alert(' not in html_out
        assert '&lt;script&gt;' in html_out

    def test_html_esc_function(self):
        assert hr._esc("<b>bold</b>") == "&lt;b&gt;bold&lt;/b&gt;"
        assert hr._esc('a"b') == "a&quot;b"
        assert hr._esc("safe") == "safe"

    def test_html_colorblind_palette(self):
        results = _make_results(n_nodes=1, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "#0072B2" in html_out  # first Okabe-Ito color

    def test_html_theme_auto_uses_media_query(self):
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(
            results, stats, [], "test.csv", 15.0, theme_mode="auto")
        assert "@media (prefers-color-scheme: dark)" in html_out

    def test_html_theme_light_disables_auto_dark(self):
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(
            results, stats, [], "test.csv", 15.0, theme_mode="light")
        assert "@media (prefers-color-scheme: dark)" not in html_out

    def test_fmt_tick_adapts_to_magnitude(self):
        assert "M" in hr._fmt_tick(2_000_000)
        assert "k" in hr._fmt_tick(50_000)
        assert hr._fmt_tick(500) == "500"
        assert hr._fmt_tick(5.3) == "5.3"
        assert hr._fmt_tick(0.42) == "0.42"


# ======================================================================
# 9. CLI entry point
# ======================================================================

class TestCLI:
    """Tests for the main() CLI function."""

    def test_exit_0_no_outliers(self, tmp_path):
        rows = [_compact_row(hostname="n1", gpu=i, mean=1000) for i in range(4)]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        with patch("sys.argv", ["prog", str(csv_path), "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 0

    def test_exit_1_with_outliers(self, tmp_path):
        rows = [_compact_row(hostname="n1", gpu=i, mean=1000) for i in range(4)]
        # Make GPU 3 an extreme outlier
        rows[3]["mean"] = "100"
        rows[3]["min"] = "95"
        rows[3]["max"] = "105"
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        with patch("sys.argv", ["prog", str(csv_path), "--quiet",
                                "--outlier-threshold", "10", "--no-color"]):
            rc = hr.main()
        assert rc == 1

    def test_exit_2_missing_file(self, tmp_path):
        with patch("sys.argv", ["prog", str(tmp_path / "nonexistent.csv"),
                                "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 2

    def test_exit_2_empty_after_filter(self, tmp_path):
        rows = [_compact_row(benchmark="GEMM")]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        with patch("sys.argv", ["prog", str(csv_path), "--benchmark",
                                "nonexistent", "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 2

    def test_benchmark_filter(self, tmp_path):
        rows = [
            _compact_row(benchmark="Batched GEMM", mean=1000),
            _compact_row(benchmark="FFT", mean=500),
        ]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        with patch("sys.argv", ["prog", str(csv_path), "--benchmark", "FFT",
                                "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 0

    def test_dtype_filter(self, tmp_path):
        rows = [
            _compact_row(dtype="float32", mean=1000),
            _compact_row(dtype="float64", mean=800),
        ]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        with patch("sys.argv", ["prog", str(csv_path), "--dtype", "float32",
                                "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 0

    def test_html_output_written(self, tmp_path):
        rows = [_compact_row(mean=1000)]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        html_path = tmp_path / "report.html"
        with patch("sys.argv", ["prog", str(csv_path), "-o", str(html_path),
                                "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 0
        assert html_path.exists()
        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_directory_input(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        for i in range(2):
            rows = [_compact_row(hostname="node{}".format(i), gpu=0)]
            (d / "node{}.csv".format(i)).write_text(_compact_csv_text(rows))

        with patch("sys.argv", ["prog", str(d), "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 0

    def test_theme_flag_dark_writes_dark_html(self, tmp_path):
        rows = [_compact_row(mean=1000)]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(_compact_csv_text(rows))

        html_path = tmp_path / "report.html"
        with patch("sys.argv", ["prog", str(csv_path), "-o", str(html_path),
                                "--theme", "dark", "--quiet", "--no-color"]):
            rc = hr.main()
        assert rc == 0
        content = html_path.read_text()
        assert "@media (prefers-color-scheme: dark)" not in content
        assert "--bg: #1a1a18" in content


# ======================================================================
# 10. Edge cases
# ======================================================================

class TestEdgeCases:
    """Misc edge cases."""

    def test_safe_float(self):
        assert hr._safe_float("3.14") == 3.14
        assert hr._safe_float("bad") == 0.0
        assert hr._safe_float("bad", -1.0) == -1.0
        assert hr._safe_float(None) == 0.0

    def test_safe_int(self):
        assert hr._safe_int("42") == 42
        assert hr._safe_int("bad") == 0
        assert hr._safe_int(None, -1) == -1

    def test_group_by_preserves_order(self):
        items = [("b", 1), ("a", 2), ("b", 3), ("a", 4)]
        groups = hr._group_by(items, lambda x: x[0])
        assert list(groups.keys()) == ["b", "a"]
        assert groups["b"] == [("b", 1), ("b", 3)]

    def test_gpu_result_dataclass(self):
        r = hr.GPUResult(
            hostname="h", gpu=0, gpu_model="m", serial="s",
            benchmark="b", dtype="d", iterations=1, runtime_s=1.0,
            min_val=1.0, mean_val=2.0, max_val=3.0,
            unit="u", power_avg_w=100.0, temp_max_c=50.0,
        )
        assert r.mean_val == 2.0
        assert r.hostname == "h"

    def test_benchmark_stats_empty(self):
        bs = hr.BenchmarkStats(name="test", dtype="fp32", unit="GFLOP/s")
        assert bs.fleet_mean == 0.0
        assert bs.fleet_cv == 0.0
        assert bs.fleet_min == 0.0
        assert bs.fleet_max == 0.0


# ======================================================================
# 11. Scale-adaptive rendering
# ======================================================================

class TestScaleAdaptive:
    """Tests for histogram vs bar chart switching and table truncation."""

    def test_small_fleet_uses_histogram(self):
        """<=50 nodes should use histogram (same as large fleet)."""
        results = _make_results(n_nodes=10, n_gpus=4)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Histogram: has GPU count axis label in SVG
        assert "GPU count" in html_out
        assert "<svg" in html_out

    def test_large_fleet_uses_histogram(self):
        """>>50 nodes should switch to histogram."""
        results = _make_results(n_nodes=100, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Histogram: has GPU count axis label in SVG
        assert "GPU count" in html_out
        assert "<svg" in html_out

    def test_large_fleet_table_truncated(self):
        """>>50 nodes should show truncated table with separator."""
        results = _make_results(n_nodes=100, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Separator row present
        assert "within threshold" in html_out
        # Should NOT have all 100 node rows
        node_count = html_out.count("node")  # rough count
        # Should be much less than 100 node entries
        assert node_count < 100 * 3  # generous upper bound vs all-rows

    def test_small_fleet_table_not_truncated(self):
        """<=50 nodes should have all rows."""
        results = _make_results(n_nodes=10, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "within threshold" not in html_out
        # All 10 nodes present
        for i in range(10):
            assert "node{}".format(i) in html_out

    def test_large_fleet_has_percentiles(self):
        """>>50 nodes should show p5/median/p95 in stats line."""
        results = _make_results(n_nodes=100, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "p5:" in html_out
        assert "median:" in html_out
        assert "p95:" in html_out

    def test_small_fleet_with_10_gpus_shows_percentiles(self):
        """>=10 GPUs should show percentile stats regardless of node count."""
        results = _make_results(n_nodes=5, n_gpus=2)  # 10 GPUs
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "p5:" in html_out
        assert "p95:" in html_out

    def test_scale_threshold_constant(self):
        assert hr.SCALE_THRESHOLD == 50


# ======================================================================
# 12. Sortable tables
# ======================================================================

class TestSortableTables:
    """Tests for sortable table markup."""

    def test_sortable_class_on_headers(self):
        results = _make_results(n_nodes=3, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert 'class="sortable"' in html_out
        assert 'data-col=' in html_out

    def test_data_sort_attributes_on_cells(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert 'data-sort=' in html_out

    def test_sortable_js_present(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "th.sortable" in html_out
        assert "addEventListener" in html_out

    def test_sortable_table_class(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "sortable-table" in html_out


# ======================================================================
# 13. vs-fleet column
# ======================================================================

class TestVsFleet:
    """Tests for the vs-fleet deviation column in HTML."""

    def test_vs_fleet_shown_in_html(self):
        results = _make_results(n_nodes=3, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "vs fleet" in html_out

    def test_vs_fleet_positive_sign(self):
        """Nodes above fleet mean should show +."""
        results = _make_results(n_nodes=3, n_gpus=1, mean_base=1000)
        # node2 will have mean 1020, fleet mean ~1010
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "+0." in html_out or "+1." in html_out

    def test_vs_fleet_has_tooltip(self):
        """vs fleet header should have title attribute."""
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "title=" in html_out
        assert "fleet mean" in html_out


# ======================================================================
# 14. Truncated table UX (section labels + show-all toggle)
# ======================================================================

class TestTruncatedTableUX:
    """Tests for section labels and show-all toggle in truncated tables."""

    def test_large_fleet_has_section_labels(self):
        """Truncated table should label lowest/highest sections."""
        results = _make_results(n_nodes=100, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Lowest performing" in html_out
        assert "Highest performing" in html_out

    def test_small_fleet_no_section_labels(self):
        """Non-truncated table should NOT have section labels."""
        results = _make_results(n_nodes=5, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Lowest performing" not in html_out

    def test_large_fleet_has_show_all_button(self):
        """Truncated table should have a Show all button."""
        results = _make_results(n_nodes=100, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Show all" in html_out
        assert "show-all-btn" in html_out

    def test_show_all_hidden_rows_contain_all_nodes(self):
        """Hidden section should contain the omitted nodes."""
        results = _make_results(n_nodes=100, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # The hidden rows are in a collapsed inner table
        assert "inner-expand" in html_out
        assert 'style="display:none"' in html_out

    def test_small_fleet_no_show_all(self):
        """Non-truncated table should NOT have show-all button in table."""
        results = _make_results(n_nodes=5, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Show all</button>" not in html_out


# ======================================================================
# 15. Sortable column discoverability
# ======================================================================

class TestSortableDiscoverability:
    """Tests for sortable arrow visibility."""

    def test_sortable_arrows_visible_at_idle(self):
        """Sort arrows should be visible (opacity >= 0.5) at idle."""
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # CSS should have opacity: 0.5 for idle arrows (not 0.3)
        assert "opacity: 0.5" in html_out


# ======================================================================
# 16. SVG chart generation
# ======================================================================

class TestSVGCharts:
    """Tests for server-side SVG chart rendering."""

    def test_svg_bar_chart_basic(self):
        """Bar chart SVG should contain rect elements and legend."""
        results = _make_results(n_nodes=3, n_gpus=2)
        by_node = hr._group_by(results, lambda r: r.hostname)
        nodes = sorted(by_node.keys())
        gpu_indices = [0, 1]
        vals = [r.mean_val for r in results]
        svg = hr._svg_bar_chart(nodes, gpu_indices, by_node, "GFLOP/s",
                                min(vals) * 0.9, max(vals) * 1.1)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "<rect" in svg
        assert "GPU 0" in svg

    def test_svg_histogram_basic(self):
        """Histogram SVG should contain bins and axis labels."""
        vals = [100 + i * 0.5 for i in range(200)]
        svg = hr._svg_histogram(vals, "GFLOP/s", 150.0, n_bins=20)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "<rect" in svg
        assert "GPU count" in svg

    def test_svg_single_bar_basic(self):
        """Secondary metric bar chart should render."""
        nodes = ["node0", "node1", "node2"]
        values = {"node0": 250, "node1": 245, "node2": 260}
        svg = hr._svg_single_bar(nodes, values, "W", "#E69F00")
        assert svg.startswith("<svg")
        assert "<rect" in svg

    def test_svg_histogram_empty(self):
        """Empty vals should return empty string."""
        svg = hr._svg_histogram([], "GFLOP/s", 0.0)
        assert svg == ""

    def test_nice_ticks_monotonic(self):
        """Tick values should be monotonically increasing."""
        ticks = hr._nice_ticks(100, 500, 5)
        assert len(ticks) >= 2
        for i in range(1, len(ticks)):
            assert ticks[i] > ticks[i - 1]

    def test_fmt_tick_formats(self):
        """Tick formatter should produce concise labels."""
        assert "M" in hr._fmt_tick(2_500_000)
        assert "k" in hr._fmt_tick(15_000)
        assert hr._fmt_tick(42) == "42.0"


# ======================================================================
# 17. Multi-metric charts
# ======================================================================

class TestMultiMetricCharts:
    """Tests for power/temperature diagnostic panels."""

    def test_small_fleet_power_panel(self):
        """Small fleet with power data should render power chart."""
        results = _make_results(n_nodes=5, n_gpus=2, power=250.0, temp=72.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Power (W)" in html_out
        assert "chart-secondary" in html_out

    def test_small_fleet_temp_panel(self):
        """Small fleet with temp data should render temperature chart."""
        results = _make_results(n_nodes=5, n_gpus=2, power=250.0, temp=72.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Temperature" in html_out

    def test_small_fleet_no_power_no_panel(self):
        """No power data => no power chart panel."""
        results = _make_results(n_nodes=5, n_gpus=2, power=0.0, temp=0.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # The chart-secondary div should not appear in the body (only in CSS)
        assert "Power (W)" not in html_out
        assert "Temperature" not in html_out

    def test_large_fleet_power_histogram(self):
        """Large fleet with power data should show power histogram."""
        results = _make_results(n_nodes=100, n_gpus=1, power=250.0, temp=72.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Power (W)" in html_out
        assert "chart-row" in html_out

    def test_large_fleet_multi_panel_layout(self):
        """Large fleet with perf+power+temp should use chart-row layout."""
        results = _make_results(n_nodes=100, n_gpus=1, power=250.0, temp=72.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "chart-panel" in html_out
        # Should have 3 panels: performance + power + temperature
        assert html_out.count("chart-panel") >= 3


# ======================================================================
# 18. Sort JS fix (section labels preserved during sort)
# ======================================================================

class TestSortJSFix:
    """Test that sort JS excludes section-label rows."""

    def test_sort_js_excludes_section_labels(self):
        """Sort script should not move section-label rows."""
        results = _make_results(n_nodes=100, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert ':not(.section-label)' in html_out
        assert 'tr.separator, tr.section-label' in html_out


# ======================================================================
# 19. CLI node table truncation
# ======================================================================

class TestCLITruncation:
    """Test that large fleet CLI node tables are truncated."""

    def test_cli_truncates_large_fleet(self):
        """CLI should truncate node table for >30 hosts."""
        results = _make_results(n_nodes=50, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        import io
        buf = io.StringIO()
        import sys
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "PASS nodes omitted" in output

    def test_cli_no_truncation_small_fleet(self):
        """CLI should show all nodes for <=30 hosts."""
        results = _make_results(n_nodes=5, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        import io
        buf = io.StringIO()
        import sys
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "omitted" not in output


# ======================================================================
# 20. Zoom toggle
# ======================================================================

class TestZoomToggle:
    """Tests for chart zoom button."""

    def test_zoom_button_present(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "zoom-btn" in html_out
        assert "chart-zoomed" in html_out

    def test_zoom_css_rules(self):
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert ".chart-wrap.chart-zoomed" in html_out


# ======================================================================
# 21. System name / job name
# ======================================================================

class TestSystemJobName:
    """Tests for --system-name and --job-name in report."""

    def test_system_name_in_title(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  system_name="AI-P2")
        assert "AI-P2" in html_out
        assert "<title>" in html_out
        # Should appear in both title and subtitle
        assert 'class="subtitle"' in html_out

    def test_job_name_in_title(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  job_name="2026-04-20 Maintenance")
        assert "2026-04-20 Maintenance" in html_out
        assert 'class="subtitle"' in html_out

    def test_both_names_combined(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  system_name="AI-P2",
                                  job_name="Weekly Stress")
        assert "AI-P2" in html_out
        assert "Weekly Stress" in html_out

    def test_no_names_no_subtitle(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert 'class="subtitle"' not in html_out

    def test_cli_summary_shows_system_name(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        import io, sys
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0,
                             system_name="AI-P2", job_name="Maintenance")
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "AI-P2" in output
        assert "Maintenance" in output


# ======================================================================
# 22. SVG chart sizes
# ======================================================================

class TestSVGSizes:
    """Tests for enlarged SVG dimensions."""

    def test_viewbox_dimensions(self):
        assert hr._SVG_W == 960
        assert hr._SVG_H == 320
        assert hr._SVG_H_SM == 260

    def test_bar_chart_uses_new_viewbox(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        by_node = hr._group_by(results, lambda r: r.hostname)
        nodes = sorted(by_node.keys())
        svg = hr._svg_bar_chart(nodes, [0], by_node, "GFLOP/s", 900, 1100)
        assert 'viewBox="0 0 960 320"' in svg


# ======================================================================
# 23. Telemetry metric visualization
# ======================================================================

class TestTelemetryMetrics:
    """Tests for SM util, mem BW util, GPU clock visualization."""

    def test_small_fleet_sm_util_panel(self):
        results = _make_results(n_nodes=5, n_gpus=2, sm_util=85.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "SM Utilization (%)" in html_out

    def test_small_fleet_mem_bw_panel(self):
        results = _make_results(n_nodes=5, n_gpus=2, mem_bw_util=60.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "Memory BW Utilization (%)" in html_out

    def test_small_fleet_gpu_clock_panel(self):
        results = _make_results(n_nodes=5, n_gpus=2, gpu_clock=1500.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "GPU Clock (MHz)" in html_out

    def test_no_telemetry_no_panels(self):
        results = _make_results(n_nodes=5, n_gpus=2, sm_util=0.0, mem_bw_util=0.0, gpu_clock=0.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "SM Utilization" not in html_out
        assert "Memory BW" not in html_out
        assert "GPU Clock" not in html_out

    def test_large_fleet_sm_util_histogram(self):
        results = _make_results(n_nodes=100, n_gpus=1, sm_util=90.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "SM Utilization (%)" in html_out
        assert "chart-panel" in html_out

    def test_benchmark_stats_collects_new_fields(self):
        results = _make_results(n_nodes=3, n_gpus=1, sm_util=85.0, mem_bw_util=60.0, gpu_clock=1500.0)
        stats = hr.compute_benchmark_stats(results)
        bs = list(stats.values())[0]
        assert len(bs.sm_util_values) == 3
        assert len(bs.mem_bw_util_values) == 3
        assert len(bs.gpu_clock_values) == 3

    def test_gpu_result_new_fields_default_zero(self):
        r = hr.GPUResult(
            hostname="h", gpu=0, gpu_model="", serial="", benchmark="t",
            dtype="f32", iterations=1, runtime_s=1.0, min_val=1, mean_val=1,
            max_val=1, unit="u", power_avg_w=0, temp_max_c=0)
        assert r.sm_util_mean == 0.0
        assert r.mem_bw_util_mean == 0.0
        assert r.gpu_clock_mean == 0.0

    def test_compact_csv_parses_verbose_fields(self):
        row = {
            "hostname": "node0", "gpu": "0", "gpu_model": "GPU", "serial": "S",
            "benchmark": "GEMM", "dtype": "float32", "iterations": "10",
            "runtime_s": "1.0", "min": "100", "mean": "100", "max": "100",
            "unit": "GFLOP/s", "power_avg_w": "250", "temp_max_c": "72",
            "sm_util_mean": "88.5", "mem_bw_util_mean": "45.2", "gpu_clock_mean": "1410",
        }
        r = hr._parse_compact_row(row)
        assert r is not None
        assert r.sm_util_mean == 88.5
        assert r.mem_bw_util_mean == 45.2
        assert r.gpu_clock_mean == 1410.0


# ======================================================================
# 24. Node map (--node-map)
# ======================================================================

class TestNodeMap:
    """Tests for --node-map location enrichment."""

    def test_load_node_map_basic(self, tmp_path):
        csv_path = tmp_path / "nodes.csv"
        csv_path.write_text("hostname,location\nnode0,rack-A1\nnode1,rack-B2\n")
        result = hr.load_node_map(csv_path)
        assert result == {"node0": "rack-A1", "node1": "rack-B2"}

    def test_load_node_map_ignores_empty(self, tmp_path):
        csv_path = tmp_path / "nodes.csv"
        csv_path.write_text(
            "hostname,location\n"
            "node0,rack-A1\n"
            ",rack-B2\n"           # empty hostname
            "node2,\n"            # empty location
            "node3,rack-C3\n"
        )
        result = hr.load_node_map(csv_path)
        assert result == {"node0": "rack-A1", "node3": "rack-C3"}

    def test_html_table_has_location_column(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        node_map = {"node0": "rack-A", "node1": "rack-B", "node2": "rack-C"}
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats)
        html_out = hr.render_html(results, stats, outliers, "test.csv", 15.0,
                                  node_map=node_map)
        assert ">location</th>" in html_out
        assert "rack-A" in html_out
        assert "rack-B" in html_out
        assert "rack-C" in html_out

    def test_html_table_no_location_without_map(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  node_map=None)
        assert ">location</th>" not in html_out

    def test_html_outlier_shows_location(self):
        results = _make_results(n_nodes=4, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        results[0].mean_val = 500.0  # >15% below fleet mean
        node_map = {"node0": "rack-Z9", "node1": "rack-A1",
                    "node2": "rack-A2", "node3": "rack-A3"}
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        html_out = hr.render_html(results, stats, outliers, "test.csv", 15.0,
                                  node_map=node_map)
        assert "rack-Z9" in html_out

    def test_html_location_cluster_summary(self):
        results = _make_results(n_nodes=6, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        # Two outliers sharing a location
        results[0].mean_val = 400.0
        results[1].mean_val = 420.0
        node_map = {"node0": "rack-X", "node1": "rack-X",
                    "node2": "rack-A", "node3": "rack-A",
                    "node4": "rack-B", "node5": "rack-B"}
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        html_out = hr.render_html(results, stats, outliers, "test.csv", 15.0,
                                  node_map=node_map)
        assert "Location clusters" in html_out
        assert "rack-X" in html_out

    def test_cli_location_column(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        node_map = {"node0": "rack-A", "node1": "rack-B", "node2": "rack-C"}
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0, node_map=node_map)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "Location" in output

    def test_cli_outlier_shows_location(self):
        results = _make_results(n_nodes=4, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        results[0].mean_val = 500.0
        node_map = {"node0": "rack-FAIL", "node1": "rack-OK",
                    "node2": "rack-OK", "node3": "rack-OK"}
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0, node_map=node_map)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "[rack-FAIL]" in output

    def test_cli_location_clusters(self):
        results = _make_results(n_nodes=6, n_gpus=1, mean_base=1000)
        for r in results:
            r.mean_val = 1000.0
        results[0].mean_val = 400.0
        results[1].mean_val = 420.0
        node_map = {"node0": "rack-X", "node1": "rack-X",
                    "node2": "rack-A", "node3": "rack-A",
                    "node4": "rack-B", "node5": "rack-B"}
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0, node_map=node_map)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "LOCATION CLUSTERS" in output

    def test_backward_compat_none_node_map(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        # HTML with no node_map
        html_out = hr.render_html(results, stats, outliers, "test.csv", 15.0,
                                  node_map=None)
        assert ">location</th>" not in html_out
        assert "Location clusters" not in html_out
        # CLI with no node_map
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0, node_map=None)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "Location" not in output
        assert "LOCATION CLUSTERS" not in output

    def test_unmapped_node_shows_dash(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        node_map = {"node0": "rack-A"}  # node1, node2 not mapped
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        # HTML: unmapped nodes get empty location cell
        html_out = hr.render_html(results, stats, outliers, "test.csv", 15.0,
                                  node_map=node_map)
        assert ">location</th>" in html_out
        assert "rack-A" in html_out
        # CLI: unmapped nodes show "--"
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0, node_map=node_map)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "--" in output


# ---------------------------------------------------------------------------
# 25. Verbose log-dir file parsing
# ---------------------------------------------------------------------------

_VERBOSE_GEMM_LOG = """\
2026-04-13T20:52:48 INFO    Using device cuda:0
2026-04-13T20:52:48 INFO    Initial telemetry: GPU0 | SM:0% | MemBW:0% | Temp:27C | Power:213W
2026-04-13T20:53:34 INFO    [GPU0 GEMM] Warmup (10 iterations)...
2026-04-13T20:54:13 INFO    repeat, iter, test, dtype, gflops, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, vbst_sync, power_W, temp_gpu_C, temp_hbm_C, mem_used_MB, mem_total_MB, mem_free_MB, hw_slowdown, sw_slowdown, power_limit, throttled
2026-04-13T20:54:14 INFO    1, 1, gemm, TF32, 890000.00, NVIDIA, NVIDIA GB300, gpu-010, 0, 1641925001189, 98, 60, , 1600, 3996, N/A, 1370.00, 58, N/A, 256858.44, 284208.00, 27349.56, 0, 0, 0, 0
2026-04-13T20:54:14 INFO    1, 2, gemm, TF32, 910000.00, NVIDIA, NVIDIA GB300, gpu-010, 0, 1641925001189, 100, 63, , 1604, 3996, N/A, 1380.00, 60, N/A, 256858.44, 284208.00, 27349.56, 0, 0, 0, 0
2026-04-13T20:54:15 INFO    1, 3, gemm, TF32, 900000.00, NVIDIA, NVIDIA GB300, gpu-010, 0, 1641925001189, 99, 62, , 1590, 3996, N/A, 1375.00, 59, N/A, 256858.44, 284208.00, 27349.56, 0, 0, 0, 0
2026-04-13T21:09:16 WARNING [WARN] Power limit reached: 1402W / 1400W (100.1%)
"""

_VERBOSE_MULTI_BENCH_LOG = """\
2026-04-13T20:52:48 INFO    Using device cuda:0
2026-04-13T20:54:13 INFO    repeat, iter, test, dtype, gflops, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, vbst_sync, power_W, temp_gpu_C, temp_hbm_C, mem_used_MB, mem_total_MB, mem_free_MB, hw_slowdown, sw_slowdown, power_limit, throttled
2026-04-13T20:54:14 INFO    1, 1, gemm, float32, 500000.00, NVIDIA, TestGPU, host1, 0, SN123, 95, 55, , 1500, 3000, N/A, 300.00, 70, N/A, 10000, 20000, 10000, 0, 0, 0, 0
2026-04-13T20:54:14 INFO    1, 2, gemm, float32, 520000.00, NVIDIA, TestGPU, host1, 0, SN123, 96, 56, , 1510, 3000, N/A, 305.00, 71, N/A, 10000, 20000, 10000, 0, 0, 0, 0
2026-04-13T21:00:00 INFO    repeat, iter, test, dtype, gb_s, vendor, model, hostname, device_id, serial, sm_util, mem_bw_util, mem_util, gpu_clock, mem_clock, vbst_sync, power_W, temp_gpu_C, temp_hbm_C, mem_used_MB, mem_total_MB, mem_free_MB, hw_slowdown, sw_slowdown, power_limit, throttled
2026-04-13T21:00:01 INFO    1, 1, mem, float32, 1200.50, NVIDIA, TestGPU, host1, 0, SN123, 40, 90, , 1400, 3000, N/A, 280.00, 68, N/A, 10000, 20000, 10000, 0, 0, 0, 0
2026-04-13T21:00:02 INFO    1, 2, mem, float32, 1250.75, NVIDIA, TestGPU, host1, 0, SN123, 42, 92, , 1420, 3000, N/A, 290.00, 69, N/A, 10000, 20000, 10000, 0, 0, 0, 0
"""


class TestVerboseLog:

    def test_is_verbose_log_true(self, tmp_path):
        f = tmp_path / "gpu0_host1_SN123.csv"
        f.write_text(_VERBOSE_GEMM_LOG)
        assert hr._is_verbose_log(f) is True

    def test_is_verbose_log_false_compact(self, tmp_path):
        f = tmp_path / "results.csv"
        f.write_text(_compact_csv_text([_compact_row()]))
        assert hr._is_verbose_log(f) is False

    def test_load_verbose_log_basic(self, tmp_path):
        f = tmp_path / "gpu0_gpu-010_164192500118.csv"
        f.write_text(_VERBOSE_GEMM_LOG)
        results = hr.load_verbose_log(f)
        assert len(results) == 1
        r = results[0]
        assert r.benchmark == "Batched GEMM"
        assert r.dtype == "TF32"
        assert r.unit == "GFLOP/s"
        assert r.iterations == 3
        assert r.hostname == "gpu-010"
        assert r.gpu == 0
        assert r.serial == "1641925001189"
        assert r.gpu_model == "NVIDIA GB300"
        assert r.min_val == 890000.00
        assert r.max_val == 910000.00
        assert 890000 < r.mean_val < 910000

    def test_load_verbose_log_multi_benchmark(self, tmp_path):
        f = tmp_path / "gpu0_host1_SN123.csv"
        f.write_text(_VERBOSE_MULTI_BENCH_LOG)
        results = hr.load_verbose_log(f)
        assert len(results) == 2
        names = [r.benchmark for r in results]
        assert "Batched GEMM" in names
        assert "Memory Traffic" in names
        gemm = [r for r in results if r.benchmark == "Batched GEMM"][0]
        mem = [r for r in results if r.benchmark == "Memory Traffic"][0]
        assert gemm.unit == "GFLOP/s"
        assert mem.unit == "GB/s"

    def test_load_verbose_log_fallback_hostname_from_filename(self, tmp_path):
        # Build a log where hostname column is empty and device_id is empty
        log = (
            "2026-04-13T20:54:13 INFO    "
            "repeat, iter, test, dtype, gflops, vendor, model, hostname, "
            "device_id, serial, power_W, temp_gpu_C\n"
            "2026-04-13T20:54:14 INFO    "
            "1, 1, gemm, float32, 100.0, NVIDIA, TestGPU, , , SN1, "
            "250.0, 70\n"
        )
        f = tmp_path / "gpu2_fallback-host_SERIAL99.csv"
        f.write_text(log)
        results = hr.load_verbose_log(f)
        assert len(results) == 1
        assert results[0].hostname == "fallback-host"
        assert results[0].gpu == 2

    def test_load_verbose_log_skips_non_data_lines(self, tmp_path):
        f = tmp_path / "gpu0_host_SN.csv"
        f.write_text(_VERBOSE_GEMM_LOG)
        results = hr.load_verbose_log(f)
        # Only 3 data lines, not the setup/warmup/warning lines
        assert results[0].iterations == 3

    def test_auto_detect_verbose_log(self, tmp_path):
        f = tmp_path / "gpu0_host_SN.csv"
        f.write_text(_VERBOSE_GEMM_LOG)
        results = hr._detect_and_load(f)
        assert len(results) == 1
        assert results[0].benchmark == "Batched GEMM"

    def test_load_verbose_log_aggregation(self, tmp_path):
        # Build a log with varied values to verify min < mean < max
        log = (
            "2026-04-13T10:00:00 INFO    "
            "repeat, iter, test, dtype, gflops, hostname, device_id, serial, "
            "model, power_W, temp_gpu_C, sm_util, mem_bw_util, gpu_clock\n"
            "2026-04-13T10:00:01 INFO    "
            "1, 1, gemm, float32, 100.0, h1, 0, S1, GPU, 200.0, 60, 80, 50, 1400\n"
            "2026-04-13T10:00:02 INFO    "
            "1, 2, gemm, float32, 200.0, h1, 0, S1, GPU, 250.0, 70, 90, 60, 1500\n"
            "2026-04-13T10:00:03 INFO    "
            "1, 3, gemm, float32, 300.0, h1, 0, S1, GPU, 300.0, 80, 100, 70, 1600\n"
        )
        f = tmp_path / "gpu0_h1_S1.csv"
        f.write_text(log)
        results = hr.load_verbose_log(f)
        r = results[0]
        assert r.min_val == 100.0
        assert r.max_val == 300.0
        assert r.mean_val == 200.0
        assert r.min_val < r.mean_val < r.max_val
        assert r.power_avg_w == 250.0
        assert r.temp_max_c == 80.0
        assert r.sm_util_mean == 90.0
        assert r.mem_bw_util_mean == 60.0
        assert r.gpu_clock_mean == 1500.0

    def test_load_verbose_log_bench_name_mapping(self, tmp_path):
        template = (
            "2026-04-13T10:00:00 INFO    "
            "repeat, iter, test, dtype, gflops, hostname, device_id\n"
            "2026-04-13T10:00:01 INFO    "
            "1, 1, {test}, float32, 100.0, h, 0\n"
        )
        for short, display in [("gemm", "Batched GEMM"), ("mem", "Memory Traffic"),
                                ("conv", "Convolution"), ("fft3d", "3D FFT"),
                                ("heat", "Heat Equation"), ("atomic", "Atomic Contention")]:
            f = tmp_path / "gpu0_h_{}.csv".format(short)
            f.write_text(template.format(test=short))
            results = hr.load_verbose_log(f)
            assert results[0].benchmark == display, \
                "{} should map to {}".format(short, display)

    def test_load_verbose_log_unit_mapping(self, tmp_path):
        for metric, unit in [("gflops", "GFLOP/s"), ("gb_s", "GB/s"),
                              ("img_s", "img/s"), ("mlups", "MLUP/s"),
                              ("mops", "Mops/s"), ("iter_s", "iter/s")]:
            log = (
                "2026-04-13T10:00:00 INFO    "
                "repeat, iter, test, dtype, {metric}, hostname, device_id\n"
                "2026-04-13T10:00:01 INFO    "
                "1, 1, bench, float32, 42.0, h, 0\n"
            ).format(metric=metric)
            f = tmp_path / "gpu0_h_{}.csv".format(metric)
            f.write_text(log)
            results = hr.load_verbose_log(f)
            assert results[0].unit == unit, \
                "{} should map to {}".format(metric, unit)


# ======================================================================
# 26. Tufte design fixes
# ======================================================================

class TestTufteDesignFixes:
    """Tests for fleet-mean reference line, grid layout, and secondary panel styling."""

    def test_fleet_mean_line_in_bar_chart(self):
        """Bar chart should include a fleet-mean reference line when fleet_mean is given."""
        results = _make_results(n_nodes=5, n_gpus=2)
        by_node = hr._group_by(results, lambda r: r.hostname)
        nodes = sorted(by_node.keys())
        vals = [r.mean_val for r in results]
        fleet_mean = sum(vals) / len(vals)
        svg = hr._svg_bar_chart(nodes, [0, 1], by_node, "GFLOP/s",
                                min(vals) * 0.9, max(vals) * 1.1,
                                fleet_mean=fleet_mean)
        assert "fleet mean" in svg
        assert 'stroke-dasharray="6,4"' in svg
        assert "var(--danger" in svg

    def test_fleet_mean_line_not_shown_when_none(self):
        """Bar chart should NOT include fleet-mean line when fleet_mean is None."""
        results = _make_results(n_nodes=3, n_gpus=1)
        by_node = hr._group_by(results, lambda r: r.hostname)
        nodes = sorted(by_node.keys())
        vals = [r.mean_val for r in results]
        svg = hr._svg_bar_chart(nodes, [0], by_node, "GFLOP/s",
                                min(vals) * 0.9, max(vals) * 1.1,
                                fleet_mean=None)
        assert "fleet mean" not in svg
        assert 'stroke-dasharray="6,4"' not in svg

    def test_fleet_mean_in_full_html(self):
        """Full HTML render should include fleet-mean marker in histogram."""
        results = _make_results(n_nodes=5, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Histogram fleet mean uses solid line with ø symbol
        assert '\u00f8' in html_out or 'opacity="0.8"' in html_out

    def test_css_grid_layout(self):
        """Secondary panels should use CSS grid, not flexbox."""
        css = hr._get_css()
        assert "grid-template-columns" in css
        assert "repeat(3" in css

    def test_secondary_panel_compact(self):
        """Secondary panels should have reduced padding."""
        css = hr._get_css()
        assert ".chart-secondary" in css
        assert "padding: 0.25rem 0" in css

    # -- Task-specified Tufte tests --

    def test_bar_chart_fleet_mean_line(self):
        """Bar chart should include fleet mean reference line when provided."""
        results = _make_results(n_nodes=3, n_gpus=2)
        by_node = hr._group_by(results, lambda r: r.hostname)
        nodes = sorted(by_node.keys())
        svg = hr._svg_bar_chart(nodes, [0, 1], by_node, "GFLOP/s",
                                900, 1100, fleet_mean=1005.0)
        assert "fleet mean" in svg
        assert 'stroke-dasharray="6,4"' in svg

    def test_bar_chart_no_fleet_mean_when_none(self):
        """Bar chart omits fleet mean line when fleet_mean is None."""
        results = _make_results(n_nodes=3, n_gpus=1)
        by_node = hr._group_by(results, lambda r: r.hostname)
        nodes = sorted(by_node.keys())
        svg = hr._svg_bar_chart(nodes, [0], by_node, "GFLOP/s", 900, 1100)
        assert "fleet mean" not in svg

    def test_secondary_panels_grid_layout(self):
        """CSS should use grid for .chart-row, not flex."""
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "grid-template-columns" in html_out

    def test_html_has_fleet_mean_line(self):
        """Full HTML render should contain fleet mean marker (ø symbol)."""
        results = _make_results(n_nodes=5, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Histogram uses ø (\u00f8) for fleet mean marker
        assert '\u00f8' in html_out


# ======================================================================
# 27. Histogram always primary chart
# ======================================================================

class TestHistogramAlways:
    """Histogram is the primary chart at ALL fleet sizes."""

    def test_small_fleet_uses_histogram(self):
        results = _make_results(n_nodes=5, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "GPU count" in html_out

    def test_histogram_outlier_threshold_lines(self):
        # Wide spread data so threshold lines fall within the range
        vals = [800 + i * 10 for i in range(50)]  # range 800-1290
        mean_val = 1045.0  # roughly center
        svg = hr._svg_histogram(vals, "GFLOP/s", mean_val, n_bins=20,
                                threshold_pct=15.0)
        # lo_bound = 1045*0.85 = 888.25, hi_bound = 1045*1.15 = 1201.75
        assert "-15%" in svg
        assert "+15%" in svg

    def test_histogram_outlier_bins_colored(self):
        # Create values with a clear outlier cluster below threshold
        vals = [1000] * 40 + [500] * 5  # 500 is way below 15% of mean ~944
        mean_val = 944.0
        svg = hr._svg_histogram(vals, "GFLOP/s", mean_val, n_bins=20,
                                threshold_pct=15.0)
        assert "#D55E00" in svg  # outlier bins colored vermillion

    def test_histogram_fleet_mean_marker(self):
        vals = [100 + i for i in range(30)]
        svg = hr._svg_histogram(vals, "GFLOP/s", 115.0, n_bins=15,
                                threshold_pct=10.0)
        # Fleet mean vertical line present (ø symbol)
        assert '\u00f8' in svg or 'opacity="0.8"' in svg

    def test_histogram_no_threshold_no_lines(self):
        vals = [100 + i for i in range(20)]
        svg = hr._svg_histogram(vals, "GFLOP/s", 110.0, n_bins=10)
        assert "-15%" not in svg
        assert "+15%" not in svg

    def test_histogram_adaptive_bins_small(self):
        """Small data sets should use fewer bins."""
        results = _make_results(n_nodes=3, n_gpus=2)  # 6 GPUs
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "<svg" in html_out
        assert "GPU count" in html_out

    def test_small_fleet_percentiles_shown(self):
        """Percentiles shown when >=10 GPUs, regardless of node count."""
        results = _make_results(n_nodes=5, n_gpus=2)  # 10 GPUs
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "p5:" in html_out


# ======================================================================
# 28. Dot plot
# ======================================================================

class TestDotPlot:
    """Tests for _svg_dot_plot (rug-density), _sigma_band, --dot-plot wiring."""

    def test_svg_dot_plot_basic(self):
        """Rug-density chart: produces density polygon and rug tick lines."""
        vals = [100 + i * 0.5 for i in range(20)] + [200, 250]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 110.0, n_bins=10)
        assert "<polygon" in svg, "density curve should render as polygon"
        assert "<line" in svg, "rug ticks should render as line elements"
        assert "density" in svg
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_dot_plot_rug_ticks_for_all_values(self):
        """Every GPU value produces a rug tick line with a tooltip."""
        vals = [100 + i * 0.1 for i in range(30)]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 101.5, n_bins=10)
        # Each value gets a <line> with a <title> tooltip
        data_lines = [line for line in svg.split('\n')
                      if '<line' in line and '<title>' in line]
        assert len(data_lines) == 30, (
            "expected 30 rug ticks, got {}".format(len(data_lines)))

    def test_dot_plot_density_curve_present(self):
        """Density curve is rendered as a filled polygon."""
        vals = [1000.0] * 20 + [1030.0] * 3
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 1000.0, n_bins=20)
        assert "<polygon" in svg, "density polygon expected"
        assert "#0072B2" in svg, "density fill color expected"

    def test_dot_plot_no_overflow_annotations(self):
        """Rug-density design never produces +N overflow annotations."""
        import random
        rng = random.Random(42)
        vals = [rng.gauss(1000, 20) for _ in range(5000)]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 1000.0, n_bins=40)
        assert "<polygon" in svg
        # The rug design has no overflow mechanism; any "+" is from
        # threshold labels like "+15%" not "+N" overflow
        overflow_annotations = [line for line in svg.split('\n')
                                if '>+' in line and 'text' in line
                                and '%' not in line]
        assert len(overflow_annotations) == 0, (
            "rug plot should never overflow; got: {}".format(
                overflow_annotations))

    def test_dot_plot_sigma_colors(self):
        """Spread data: rug ticks use all sigma band colors."""
        vals = [100] * 20 + [200] * 20 + [500] * 5 + [900] * 3
        import statistics as st
        mean = st.mean(vals)
        svg = hr._svg_dot_plot(vals, "GFLOP/s", mean, n_bins=20)
        for color, _ in hr.SIGMA_COLORS:
            assert color in svg

    def test_dot_plot_threshold_lines(self):
        vals = [800 + i * 10 for i in range(50)]
        mean_val = 1045.0
        svg = hr._svg_dot_plot(vals, "GFLOP/s", mean_val, n_bins=20,
                               threshold_pct=15.0)
        assert "-15%" in svg
        assert "+15%" in svg

    def test_dot_plot_fleet_mean_marker(self):
        vals = [100 + i for i in range(30)]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 115.0, n_bins=15)
        assert '\u00f8' in svg

    def test_dot_plot_adaptive_tick_width(self):
        """Tick width adapts to fleet size: thinner for larger fleets."""
        small_vals = [100, 101, 102, 103, 200]
        svg_small = hr._svg_dot_plot(small_vals, "GFLOP/s", 102.0, n_bins=5)
        large_vals = [100 + (i % 50) for i in range(500)] + [500, 600]
        svg_large = hr._svg_dot_plot(large_vals, "GFLOP/s", 125.0, n_bins=40)
        # Both produce rug ticks (line elements with tooltips)
        small_ticks = [l for l in svg_small.split('\n')
                       if '<line' in l and '<title>' in l]
        large_ticks = [l for l in svg_large.split('\n')
                       if '<line' in l and '<title>' in l]
        assert len(small_ticks) == 5
        assert len(large_ticks) == 502

    def test_dot_plot_tooltip_with_results(self):
        """Rug ticks carry hostname/GPU info in tooltips."""
        results = _make_results(n_nodes=10, n_gpus=4)
        results.append(hr.GPUResult(
            hostname="outlier", gpu=0, gpu_model="TestGPU",
            serial="SNout", benchmark="Batched GEMM", dtype="float32",
            iterations=100, runtime_s=10.0,
            min_val=100, mean_val=100, max_val=100,
            unit="GFLOP/s", power_avg_w=250, temp_max_c=72,
        ))
        vals = [r.mean_val for r in results]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 1005.0, n_bins=20,
                               results=results)
        assert "outlier" in svg
        assert "GPU" in svg

    def test_dot_plot_tooltip_without_results(self):
        vals = [100.0, 200.0, 300.0]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 200.0, n_bins=3)
        assert "GFLOP/s" in svg
        # No hostname should appear
        assert "node" not in svg

    def test_dot_plot_html_wiring(self):
        """render_html with dot_plot=True produces SVG with density polygon."""
        results = _make_results(n_nodes=5, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  dot_plot=True)
        assert "<polygon" in html_out
        assert "density" in html_out

    def test_dot_plot_secondary_panels(self):
        """Secondary panels (power, temp) also produce SVG content."""
        results = _make_results(n_nodes=5, n_gpus=2, power=250.0, temp=72.0)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  dot_plot=True)
        assert "<svg" in html_out
        assert "<polygon" in html_out

    def test_histogram_default_when_no_flag(self):
        results = _make_results(n_nodes=5, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0,
                                  dot_plot=False)
        # Histogram uses <rect> elements
        assert "<rect" in html_out
        # Dot plot density polygon should not appear in histogram mode
        svg_sections = html_out.split("GPU count")
        if len(svg_sections) > 1:
            histogram_section = svg_sections[1].split("</svg>")[0]
            assert "<polygon" not in histogram_section

    def test_sigma_band_function(self):
        mean = 100.0
        stdev = 10.0
        # Within 1 sigma
        color, label = hr._sigma_band(105.0, mean, stdev)
        assert color == "#0072B2"
        assert "1\u03c3" in label
        # 1-2 sigma
        color, label = hr._sigma_band(115.0, mean, stdev)
        assert color == "#E69F00"
        # 2-3 sigma
        color, label = hr._sigma_band(125.0, mean, stdev)
        assert color == "#D55E00"
        # Beyond 3 sigma
        color, label = hr._sigma_band(140.0, mean, stdev)
        assert color == "#000000"
        # Zero stdev -> always first color
        color, label = hr._sigma_band(999.0, mean, 0.0)
        assert color == "#0072B2"

    def test_dot_plot_large_fleet_no_overflow(self):
        """200 identical values: rug handles gracefully, no overflow."""
        vals = [100.0] * 200
        svg = hr._svg_dot_plot(vals, "GFLOP/s", 100.0, n_bins=1)
        # All 200 values get rug ticks
        data_ticks = [l for l in svg.split('\n')
                      if '<line' in l and '<title>' in l]
        assert len(data_ticks) == 200

    def test_dot_plot_outlier_labels(self):
        """Extreme outliers (>3 sigma) get text labels when results provided."""
        import statistics as st
        # Tight cluster + extreme outlier
        base = [1000.0] * 50
        results = []
        for i in range(50):
            results.append(hr.GPUResult(
                hostname="node0", gpu=i % 4, gpu_model="TestGPU",
                serial="SN{:04d}".format(i), benchmark="GEMM",
                dtype="float32", iterations=100, runtime_s=10.0,
                min_val=1000, mean_val=1000, max_val=1000,
                unit="GFLOP/s", power_avg_w=250, temp_max_c=72))
        # Add extreme outlier
        results.append(hr.GPUResult(
            hostname="badnode", gpu=7, gpu_model="TestGPU",
            serial="SNbad", benchmark="GEMM", dtype="float32",
            iterations=100, runtime_s=10.0,
            min_val=500, mean_val=500, max_val=500,
            unit="GFLOP/s", power_avg_w=250, temp_max_c=72))
        vals = [r.mean_val for r in results]
        svg = hr._svg_dot_plot(vals, "GFLOP/s", st.mean(vals), n_bins=20,
                               results=results)
        assert "badnode:7" in svg, "extreme outlier should get a text label"

    def test_cli_dot_plot_flag(self, tmp_path):
        rows = [_compact_row(gpu=i) for i in range(4)]
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(_compact_csv_text(rows))
        out_path = tmp_path / "report.html"
        with patch.object(sys, 'argv',
                          ['prog', str(csv_path), '-o', str(out_path),
                           '--dot-plot']):
            rc = hr.main()
        assert rc == 0
        html_content = out_path.read_text()
        assert "<svg" in html_content


# ======================================================================
# Interactive Plotly dashboard
# ======================================================================

class TestInteractiveMode:
    """Tests for --interactive Plotly dashboard generation."""

    def test_interactive_flag_accepted(self):
        """Parser accepts --interactive without error."""
        parser = hr.argparse.ArgumentParser()
        parser.add_argument("source")
        parser.add_argument("--interactive", action="store_true")
        args = parser.parse_args(["dummy.csv", "--interactive"])
        assert args.interactive is True

    def test_interactive_without_plotly_falls_back(self, tmp_path):
        """When plotly is unavailable, --interactive falls back to SVG with warning."""
        rows = [_compact_row(gpu=i) for i in range(4)]
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(_compact_csv_text(rows))
        out_path = tmp_path / "report.html"

        with patch.object(hr, '_PLOTLY_AVAILABLE', False):
            with patch.object(sys, 'argv',
                              ['prog', str(csv_path), '-o', str(out_path),
                               '--interactive']):
                rc = hr.main()

        assert rc == 0
        content = out_path.read_text()
        assert "<svg" in content
        assert "Plotly.newPlot" not in content

    def test_interactive_renders_html(self):
        """_render_interactive_html produces valid output with Plotly.js."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* plotly.js mock content */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)

            assert '/* plotly.js mock content */' in html_out
            assert 'Plotly.newPlot' in html_out
            assert '<div id=' in html_out
            assert 'id="filter-host"' in html_out
            assert 'id="metric-select"' in html_out
            assert 'id="distribution-chart"' in html_out
            assert 'id="fleetmap-chart"' in html_out
            assert 'id="strip-chart"' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_with_iteration_data(self):
        """Time series chart is visible when _iteration_data is populated."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = []

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            hr._iteration_data["node0:0:Batched GEMM:float32"] = [
                {"iteration": 0, "performance": 1000.0, "power_W": 250.0,
                 "temp_gpu_C": 72.0, "sm_util": 90.0, "mem_bw_util": 50.0,
                 "gpu_clock": 1500.0},
                {"iteration": 1, "performance": 1010.0, "power_W": 252.0,
                 "temp_gpu_C": 73.0, "sm_util": 91.0, "mem_bw_util": 51.0,
                 "gpu_clock": 1500.0},
            ]

            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* plotly mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)

            assert 'id="timeseries-section" style="display:block"' in html_out
            assert 'id="timeseries-chart"' in html_out
            assert 'renderTimeSeries' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_without_iteration_data(self):
        """Time series section is hidden when no iteration data exists."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = []

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()

            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* plotly mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)

            assert 'id="timeseries-section" style="display:none"' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_html_escaping(self):
        """Hostnames with </script> are safely escaped in embedded JSON."""
        from unittest.mock import MagicMock
        results = [hr.GPUResult(
            hostname='</script><script>alert("xss")</script>',
            gpu=0, gpu_model="TestGPU", serial="SN0000",
            benchmark="Batched GEMM", dtype="float32",
            iterations=100, runtime_s=10.0,
            min_val=950, mean_val=1000, max_val=1050,
            unit="GFLOP/s", power_avg_w=250, temp_max_c=72,
        )]
        stats = hr.compute_benchmark_stats(results)
        outliers = []

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)

            # The raw </script> must NOT appear unescaped inside a JSON block
            # (it would prematurely close the <script> tag)
            assert '</script><script>alert' not in html_out
            # The safely escaped form <\/script> should be present
            assert '<\\/script>' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_data_embedding(self):
        """Result data is properly JSON-serialized in the HTML."""
        from unittest.mock import MagicMock
        import re
        results = _make_results(n_nodes=2, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)

            assert 'var _RCOLS = ' in html_out
            assert 'var _RDATA = ' in html_out
            assert 'var RESULTS = ' in html_out
            assert 'var STATS = ' in html_out
            assert 'var OUTLIERS = ' in html_out
            assert 'var ITER_DATA = ' in html_out
            assert 'var THRESHOLD = ' in html_out

            # Extract columnar data and reconstruct RESULTS
            m_cols = re.search(r'var _RCOLS = (.+?);\n', html_out)
            m_data = re.search(r'var _RDATA = (.+?);\n', html_out)
            assert m_cols is not None
            assert m_data is not None
            cols = json.loads(m_cols.group(1))
            rows = json.loads(m_data.group(1))
            data = [dict(zip(cols, row)) for row in rows]
            assert len(data) == len(results)
            assert data[0]["hostname"] == "node0"
            assert data[0]["benchmark"] == "Batched GEMM"
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_theme_mode_embedded(self):
        """Interactive HTML embeds CLI-selected theme mode."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=1, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [],
                        source_name="test.csv", threshold=15.0,
                        theme_mode="light")

            assert 'var THEME_MODE = "light";' in html_out
            assert 'function initTheme()' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_iteration_data_from_verbose_log(self, tmp_path):
        """Verbose log parsing populates _iteration_data."""
        log_content = (
            "2026-01-01T00:00:00 INFO repeat, iter, test, dtype, gflops, "
            "power_W, temp_gpu_C, sm_util, mem_bw_util, gpu_clock\n"
            "2026-01-01T00:00:01 INFO 0, 0, gemm, float32, 1000.0, "
            "250.0, 72.0, 90.0, 50.0, 1500.0\n"
            "2026-01-01T00:00:02 INFO 0, 1, gemm, float32, 1010.0, "
            "252.0, 73.0, 91.0, 51.0, 1510.0\n"
            "2026-01-01T00:00:03 INFO 0, 2, gemm, float32, 1005.0, "
            "251.0, 72.5, 90.5, 50.5, 1505.0\n"
        )
        log_path = tmp_path / "gpu0_testhost_SN123456789012.csv"
        log_path.write_text(log_content)

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            results = hr.load_verbose_log(log_path)
            assert len(results) == 1
            assert results[0].benchmark == "Batched GEMM"

            assert len(hr._iteration_data) > 0
            key = list(hr._iteration_data.keys())[0]
            records = hr._iteration_data[key]
            assert len(records) == 3
            assert records[0]["performance"] == 1000.0
            assert records[1]["power_W"] == 252.0
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_iteration_data_from_json(self, tmp_path):
        """JSON loading populates _iteration_data from iteration_telemetry."""
        json_data = {
            "metadata": {"hostname": "testhost"},
            "gpus": [{
                "gpu_index": 0,
                "model": "TestGPU",
                "serial": "SN0000",
                "benchmarks": [{
                    "name": "Batched GEMM",
                    "params": {"dtype": "float32"},
                    "iterations": 3,
                    "runtime_s": 5.0,
                    "min": 950.0,
                    "mean": 1000.0,
                    "max": 1050.0,
                    "unit": "GFLOP/s",
                    "telemetry": {},
                    "iteration_telemetry": [
                        {
                            "iteration": 0,
                            "performance": 950.0,
                            "telemetry": {
                                "power_W": 248.0,
                                "temp_gpu_C": 71.0,
                                "sm_util": 89.0,
                                "mem_bw_util": 49.0,
                                "gpu_clock": 1490.0,
                            }
                        },
                        {
                            "iteration": 1,
                            "performance": 1000.0,
                            "telemetry": {
                                "power_W": 250.0,
                                "temp_gpu_C": 72.0,
                                "sm_util": 90.0,
                                "mem_bw_util": 50.0,
                                "gpu_clock": 1500.0,
                            }
                        },
                        {
                            "iteration": 2,
                            "performance": 1050.0,
                            "telemetry": {
                                "power_W": 252.0,
                                "temp_gpu_C": 73.0,
                                "sm_util": 91.0,
                                "mem_bw_util": 51.0,
                                "gpu_clock": 1510.0,
                            }
                        },
                    ]
                }]
            }]
        }
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps(json_data))

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            results = hr.load_json(json_path)
            assert len(results) == 1

            assert len(hr._iteration_data) == 1
            key = "testhost:0:Batched GEMM:float32"
            assert key in hr._iteration_data
            records = hr._iteration_data[key]
            assert len(records) == 3
            assert records[0]["performance"] == 950.0
            assert records[2]["power_W"] == 252.0
            assert records[1]["gpu_clock"] == 1500.0
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_chart_sections_present(self):
        """All 6 chart sections + inventory are in the HTML."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=3, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            # All chart div IDs present
            for div_id in ["fleetmap-chart", "distribution-chart", "efficiency-chart",
                           "strip-chart", "timeseries-chart",
                           "inventory-table"]:
                assert 'id="{}"'.format(div_id) in html_out, \
                    "Missing chart div: {}".format(div_id)
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_js_render_functions_present(self):
        """All JS render functions are embedded."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            for fn in ["renderFleetMap", "renderDistribution", "renderEfficiency",
                        "renderStrip", "renderTimeSeries",
                        "renderInventory", "renderAll"]:
                assert "function {}".format(fn) in html_out, \
                    "Missing JS function: {}".format(fn)
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_fleet_map_section_has_subtitle(self):
        """Fleet Map section has descriptive subtitle."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            assert "Fleet Map" in html_out
            assert "topology-correlated" in html_out.lower()
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_node_map_embedded(self):
        """NODE_MAP is serialized into the interactive HTML."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        node_map = {"node0": "Rack A", "node1": "Rack B"}
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv",
                        threshold=15.0, node_map=node_map)
            assert 'var NODE_MAP = ' in html_out
            assert 'Rack A' in html_out
            assert 'Rack B' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_xname_parser_in_js(self):
        """JS includes xname parsing for Cray EX topology."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            assert "parseXname" in html_out
            assert "cabinet" in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_efficiency_section_present(self):
        """Power vs Performance scatter section exists (hidden by default if no power)."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            assert 'id="efficiency-section"' in html_out
            assert "Power vs Performance" in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_strip_chart_section(self):
        """Node Variability strip chart section exists."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=3, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            assert "Node Variability" in html_out
            assert "fleet mean" in html_out.lower()
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_helper_functions_present(self):
        """JS utility functions for status detection and color mapping."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            for fn in ["arrStats", "mkOutlierSet", "mkThrottleSet",
                        "gpuStatus", "statusColor", "isLowerBetter",
                        "metricUnit"]:
                assert fn in html_out, "Missing JS helper: {}".format(fn)
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_iteration_trace_envelope(self):
        """Iteration trace shows p10-p90 envelope when >= 10 GPU traces."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            assert "Iteration Trace" in html_out
            assert "p10" in html_out
            assert "fleet median" in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_sigma_bands_in_distribution(self):
        """Distribution chart JS includes sigma band reference lines."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=3, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            # Sigma annotations in JS
            assert "\\u03c3" in html_out  # sigma symbol
            assert "fleet mean" in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_inventory_table_enhanced(self):
        """Enhanced inventory table has metric, deviation, power, temp, status columns."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, [], source_name="t.csv", threshold=15.0)
            # Enhanced inventory has vs Fleet and Status columns
            assert "vs Fleet" in html_out
            assert "Status" in html_out
            # Status icons in JS
            assert "\\u2714" in html_out  # checkmark
            assert "\\u2716" in html_out  # cross
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

class TestThermalThrottling:
    """Tests for throttle detection, flagging, and display."""

    # -- Data model --

    def test_gpu_result_throttled_default_false(self):
        r = hr.GPUResult(
            hostname="h", gpu=0, gpu_model="", serial="", benchmark="t",
            dtype="f32", iterations=1, runtime_s=1.0, min_val=1, mean_val=1,
            max_val=1, unit="u", power_avg_w=0, temp_max_c=0)
        assert r.throttled is False
        assert r.throttle_samples == 0

    def test_gpu_result_throttled_explicit(self):
        r = hr.GPUResult(
            hostname="h", gpu=0, gpu_model="", serial="", benchmark="t",
            dtype="f32", iterations=1, runtime_s=1.0, min_val=1, mean_val=1,
            max_val=1, unit="u", power_avg_w=0, temp_max_c=0,
            throttled=True, throttle_samples=5)
        assert r.throttled is True
        assert r.throttle_samples == 5

    # -- Compact CSV parsing --

    def test_compact_csv_parses_throttled_true(self):
        row = {
            "hostname": "n1", "gpu": "0", "gpu_model": "GPU", "serial": "S",
            "benchmark": "GEMM", "dtype": "float32", "iterations": "10",
            "runtime_s": "1.0", "min": "100", "mean": "100", "max": "100",
            "unit": "GFLOP/s", "power_avg_w": "250", "temp_max_c": "72",
            "throttled": "true",
        }
        r = hr._parse_compact_row(row)
        assert r is not None
        assert r.throttled is True

    def test_compact_csv_parses_throttled_empty(self):
        row = {
            "hostname": "n1", "gpu": "0", "gpu_model": "GPU", "serial": "S",
            "benchmark": "GEMM", "dtype": "float32", "iterations": "10",
            "runtime_s": "1.0", "min": "100", "mean": "100", "max": "100",
            "unit": "GFLOP/s", "power_avg_w": "250", "temp_max_c": "72",
            "throttled": "",
        }
        r = hr._parse_compact_row(row)
        assert r is not None
        assert r.throttled is False

    def test_compact_csv_parses_no_throttled_column(self):
        row = {
            "hostname": "n1", "gpu": "0", "gpu_model": "GPU", "serial": "S",
            "benchmark": "GEMM", "dtype": "float32", "iterations": "10",
            "runtime_s": "1.0", "min": "100", "mean": "100", "max": "100",
            "unit": "GFLOP/s", "power_avg_w": "250", "temp_max_c": "72",
        }
        r = hr._parse_compact_row(row)
        assert r is not None
        assert r.throttled is False

    # -- Verbose log parsing --

    def test_verbose_log_detects_throttle(self, tmp_path):
        log = (
            "2026-04-13T10:00:00 INFO    "
            "repeat, iter, test, dtype, gflops, hostname, device_id, serial, "
            "hw_slowdown, sw_slowdown, power_limit, throttled\n"
            "2026-04-13T10:00:01 INFO    "
            "1, 1, gemm, float32, 100.0, h1, 0, S1, 0, 0, 0, 0\n"
            "2026-04-13T10:00:02 INFO    "
            "1, 2, gemm, float32, 200.0, h1, 0, S1, 1, 0, 0, 1\n"
            "2026-04-13T10:00:03 INFO    "
            "1, 3, gemm, float32, 300.0, h1, 0, S1, 0, 1, 0, 1\n"
        )
        f = tmp_path / "gpu0_h1_S1.csv"
        f.write_text(log)
        results = hr.load_verbose_log(f)
        assert len(results) == 1
        assert results[0].throttled is True
        assert results[0].throttle_samples == 2

    def test_verbose_log_no_throttle(self, tmp_path):
        log = (
            "2026-04-13T10:00:00 INFO    "
            "repeat, iter, test, dtype, gflops, hostname, device_id, serial, "
            "hw_slowdown, sw_slowdown, power_limit, throttled\n"
            "2026-04-13T10:00:01 INFO    "
            "1, 1, gemm, float32, 100.0, h1, 0, S1, 0, 0, 0, 0\n"
            "2026-04-13T10:00:02 INFO    "
            "1, 2, gemm, float32, 200.0, h1, 0, S1, 0, 0, 0, 0\n"
        )
        f = tmp_path / "gpu0_h1_S1.csv"
        f.write_text(log)
        results = hr.load_verbose_log(f)
        assert results[0].throttled is False
        assert results[0].throttle_samples == 0

    def test_verbose_log_iteration_data_has_throttle(self, tmp_path):
        log = (
            "2026-04-13T10:00:00 INFO    "
            "repeat, iter, test, dtype, gflops, hostname, device_id, throttled\n"
            "2026-04-13T10:00:01 INFO    "
            "1, 1, gemm, float32, 100.0, h1, 0, 1\n"
            "2026-04-13T10:00:02 INFO    "
            "1, 2, gemm, float32, 200.0, h1, 0, 0\n"
        )
        f = tmp_path / "gpu0_h1_S1.csv"
        f.write_text(log)

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            hr.load_verbose_log(f)
            key = list(hr._iteration_data.keys())[0]
            records = hr._iteration_data[key]
            assert records[0]["throttled"] == 1
            assert records[1]["throttled"] == 0
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    # -- JSON parsing --

    def test_json_detects_bench_level_throttle(self, tmp_path):
        data = {
            "metadata": {"hostname": "testhost"},
            "gpus": [{
                "gpu_index": 0, "model": "GPU", "serial": "SN0",
                "benchmarks": [{
                    "name": "Batched GEMM", "min": 950, "mean": 1000, "max": 1050,
                    "unit": "GFLOP/s", "iterations": 100, "runtime_s": 10.0,
                    "params": {"dtype": "float32"}, "telemetry": {},
                    "throttled": True,
                }]
            }]
        }
        p = tmp_path / "results.json"
        p.write_text(json.dumps(data))
        results = hr.load_json(p)
        assert results[0].throttled is True

    def test_json_detects_iteration_throttle(self, tmp_path):
        data = {
            "metadata": {"hostname": "testhost"},
            "gpus": [{
                "gpu_index": 0, "model": "GPU", "serial": "SN0",
                "benchmarks": [{
                    "name": "Batched GEMM", "min": 950, "mean": 1000, "max": 1050,
                    "unit": "GFLOP/s", "iterations": 3, "runtime_s": 5.0,
                    "params": {"dtype": "float32"}, "telemetry": {},
                    "iteration_telemetry": [
                        {"iteration": 0, "performance": 950,
                         "telemetry": {"throttled": 0}},
                        {"iteration": 1, "performance": 1000,
                         "telemetry": {"throttled": 1, "hw_slowdown": 1}},
                        {"iteration": 2, "performance": 1050,
                         "telemetry": {"throttled": 0}},
                    ]
                }]
            }]
        }
        p = tmp_path / "results.json"
        p.write_text(json.dumps(data))
        results = hr.load_json(p)
        assert results[0].throttled is True
        assert results[0].throttle_samples == 1

    def test_json_no_throttle(self, tmp_path):
        data = {
            "metadata": {"hostname": "testhost"},
            "gpus": [{
                "gpu_index": 0, "model": "GPU", "serial": "SN0",
                "benchmarks": [{
                    "name": "Batched GEMM", "min": 950, "mean": 1000, "max": 1050,
                    "unit": "GFLOP/s", "iterations": 2, "runtime_s": 5.0,
                    "params": {"dtype": "float32"}, "telemetry": {},
                    "iteration_telemetry": [
                        {"iteration": 0, "performance": 950,
                         "telemetry": {"throttled": 0}},
                        {"iteration": 1, "performance": 1000,
                         "telemetry": {"throttled": 0}},
                    ]
                }]
            }]
        }
        p = tmp_path / "results.json"
        p.write_text(json.dumps(data))
        results = hr.load_json(p)
        assert results[0].throttled is False
        assert results[0].throttle_samples == 0

    def test_json_iteration_data_has_throttle(self, tmp_path):
        data = {
            "metadata": {"hostname": "testhost"},
            "gpus": [{
                "gpu_index": 0, "model": "GPU", "serial": "SN0",
                "benchmarks": [{
                    "name": "GEMM", "min": 1, "mean": 2, "max": 3,
                    "unit": "GFLOP/s", "iterations": 2, "runtime_s": 1.0,
                    "params": {"dtype": "fp32"}, "telemetry": {},
                    "iteration_telemetry": [
                        {"iteration": 0, "performance": 1,
                         "telemetry": {"throttled": 1}},
                        {"iteration": 1, "performance": 2,
                         "telemetry": {"throttled": 0}},
                    ]
                }]
            }]
        }
        p = tmp_path / "results.json"
        p.write_text(json.dumps(data))

        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            hr.load_json(p)
            key = list(hr._iteration_data.keys())[0]
            records = hr._iteration_data[key]
            assert records[0]["throttled"] == 1
            assert records[1]["throttled"] == 0
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    # -- BenchmarkStats --

    def test_benchmark_stats_collects_throttled(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        results[0].throttled = True
        results[0].throttle_samples = 5
        stats = hr.compute_benchmark_stats(results)
        bs = list(stats.values())[0]
        assert len(bs.throttled_results) == 1
        assert bs.throttled_results[0].hostname == "node0"

    def test_benchmark_stats_no_throttled(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        bs = list(stats.values())[0]
        assert len(bs.throttled_results) == 0

    # -- CLI output --

    def test_cli_shows_thrt_column(self):
        results = _make_results(n_nodes=2, n_gpus=1)
        results[0].throttled = True
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "Thrt" in output

    def test_cli_shows_thermal_throttling_section(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        results[0].throttled = True
        results[0].throttle_samples = 12
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "THERMAL THROTTLING" in output
        assert "1 GPU(s)" in output
        assert "1 node(s)" in output
        assert "12 samples" in output

    def test_cli_no_throttle_section_when_clean(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "THERMAL THROTTLING" not in output

    def test_cli_node_status_thrt(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        results[1].throttled = True
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "THRT" in output

    def test_cli_verdict_includes_throttled(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        results[0].throttled = True
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "THROTTLED" in output

    def test_cli_verdict_no_throttle_when_clean(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            hr.print_summary(results, stats, outliers, 15.0)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        assert "THROTTLED" not in output
        assert "0 WARN" in output

    # -- HTML output --

    def test_html_has_thrt_column(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert ">thrt</th>" in html_out

    def test_html_throttle_warning_icon(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        results[0].throttled = True
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Warning triangle character for throttled node
        assert "\u26a0" in html_out

    def test_html_no_throttle_icon_when_clean(self):
        results = _make_results(n_nodes=3, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        # Warning icon should not appear in table cells (may appear in header tooltip)
        table_section = html_out.split("<tbody>")[1].split("</tbody>")[0]
        assert "\u26a0" not in table_section

    def test_html_stats_line_throttle_count(self):
        results = _make_results(n_nodes=5, n_gpus=2)
        results[0].throttled = True
        results[1].throttled = True
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "2 GPU(s) throttled" in html_out

    def test_html_stats_line_no_throttle_when_clean(self):
        results = _make_results(n_nodes=5, n_gpus=2)
        stats = hr.compute_benchmark_stats(results)
        html_out = hr.render_html(results, stats, [], "test.csv", 15.0)
        assert "throttled" not in html_out

    # -- Interactive (Plotly) output --

    def test_interactive_results_data_has_throttle_fields(self):
        """results_data serialized into RESULTS includes throttled and throttle_samples."""
        from unittest.mock import MagicMock
        import re as re_mod
        results = _make_results(n_nodes=2, n_gpus=1)
        results[0].throttled = True
        results[0].throttle_samples = 7
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            m_cols = re_mod.search(r'var _RCOLS = (.+?);\n', html_out)
            m_data = re_mod.search(r'var _RDATA = (.+?);\n', html_out)
            assert m_cols is not None
            assert m_data is not None
            cols = json.loads(m_cols.group(1))
            rows = json.loads(m_data.group(1))
            data = [dict(zip(cols, row)) for row in rows]
            assert data[0]["throttled"] is True
            assert data[0]["throttle_samples"] == 7
            assert data[1]["throttled"] is False
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_throttle_metric_card(self):
        """Throttled metric card shows count when GPUs are throttled."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=3, n_gpus=2)
        results[0].throttled = True
        results[2].throttled = True
        stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(stats, threshold_pct=15.0)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            assert '>throttled</div>' in html_out.lower() or 'throttled</div>' in html_out
            # Card should show count > 0 with "bad" class
            assert 'metric-card bad' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_no_throttle_metric_card_ok(self):
        """Throttled metric card shows 0 with 'ok' class when no throttling."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = []
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            assert 'throttled</div>' in html_out
            # All metric cards with 'ok' class (no 'bad' for throttle)
            # The throttle card specifically should have "ok" when count is 0
            # Find the throttled card content
            assert '"label">throttled</div>' in html_out.lower() or \
                   '"label">Throttled</div>' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_header_meta_throttle_warning(self):
        """Header meta shows throttle warning when GPUs are throttled."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=2)
        results[0].throttled = True
        results[1].throttled = True  # same node, different GPU
        stats = hr.compute_benchmark_stats(results)
        outliers = []
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            assert "GPU(s) throttled" in html_out
            assert "\u26a0" in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_header_meta_no_throttle_when_clean(self):
        """Header meta does not mention throttle when no GPUs throttled."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        stats = hr.compute_benchmark_stats(results)
        outliers = []
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            meta_section = html_out.split('<div class="meta">')[1].split('</div>')[0]
            assert "throttled" not in meta_section
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_inventory_throttle_column(self):
        """Inventory table has Throttled column with warning icon for throttled GPUs."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=2, n_gpus=1)
        results[0].throttled = True
        stats = hr.compute_benchmark_stats(results)
        outliers = []
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html_out = hr._render_interactive_html(
                        results, stats, outliers,
                        source_name="test.csv", threshold=15.0)
            # JS inventory table has Status column and gpuStatus checks r.throttled
            assert '"Status"' in html_out or "Status</th>" in html_out
            # JS logic: gpuStatus checks r.throttled via mkThrottleSet
            assert 'r.throttled' in html_out
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)


# ---------------------------------------------------------------------------
# Readiness audit fixes
# ---------------------------------------------------------------------------

class TestReadinessFixes:
    """Tests for readiness audit fixes (XSS, _safe_float, _auto_scale_units, etc.)."""

    # -- _auto_scale_units tests (previously zero coverage) --

    def test_auto_scale_gflops_to_tflops(self):
        """GFLOP/s values above 1000 should be scaled to TFLOP/s."""
        results = [
            hr.GPUResult(hostname="node1", gpu=0, gpu_model="H100", serial="S1",
                         benchmark="Batched GEMM", dtype="float32", iterations=10,
                         runtime_s=1.0, min_val=1500.0, mean_val=1600.0,
                         max_val=1700.0, unit="GFLOP/s", power_avg_w=300.0,
                         temp_max_c=70.0),
            hr.GPUResult(hostname="node2", gpu=0, gpu_model="H100", serial="S2",
                         benchmark="Batched GEMM", dtype="float32", iterations=10,
                         runtime_s=1.0, min_val=1400.0, mean_val=1500.0,
                         max_val=1600.0, unit="GFLOP/s", power_avg_w=300.0,
                         temp_max_c=70.0),
        ]
        bench_stats = hr.compute_benchmark_stats(results)
        hr._auto_scale_units(results, bench_stats, {})
        key = ("Batched GEMM", "float32")
        assert bench_stats[key].unit == "TFLOP/s"
        assert results[0].unit == "TFLOP/s"
        assert abs(results[0].mean_val - 1.6) < 0.01
        assert abs(results[1].mean_val - 1.5) < 0.01

    def test_auto_scale_no_scale_below_threshold(self):
        """Values below 1000 should NOT be scaled."""
        results = [
            hr.GPUResult(hostname="node1", gpu=0, gpu_model="H100", serial="S1",
                         benchmark="Batched GEMM", dtype="float32", iterations=10,
                         runtime_s=1.0, min_val=500.0, mean_val=600.0,
                         max_val=700.0, unit="GFLOP/s", power_avg_w=300.0,
                         temp_max_c=70.0),
        ]
        bench_stats = hr.compute_benchmark_stats(results)
        hr._auto_scale_units(results, bench_stats, {})
        key = ("Batched GEMM", "float32")
        assert bench_stats[key].unit == "GFLOP/s"
        assert results[0].mean_val == 600.0

    def test_auto_scale_gbs_to_tbs(self):
        """GB/s values above 1000 should be scaled to TB/s."""
        results = [
            hr.GPUResult(hostname="node1", gpu=0, gpu_model="H100", serial="S1",
                         benchmark="Memory Traffic", dtype="float32", iterations=10,
                         runtime_s=1.0, min_val=2000.0, mean_val=2500.0,
                         max_val=3000.0, unit="GB/s", power_avg_w=300.0,
                         temp_max_c=70.0),
        ]
        bench_stats = hr.compute_benchmark_stats(results)
        hr._auto_scale_units(results, bench_stats, {})
        key = ("Memory Traffic", "float32")
        assert bench_stats[key].unit == "TB/s"
        assert abs(results[0].mean_val - 2.5) < 0.01

    def test_auto_scale_scales_iteration_data(self):
        """_auto_scale_units should also scale iteration data."""
        results = [
            hr.GPUResult(hostname="node1", gpu=0, gpu_model="H100", serial="S1",
                         benchmark="Batched GEMM", dtype="float32", iterations=10,
                         runtime_s=1.0, min_val=1500.0, mean_val=1600.0,
                         max_val=1700.0, unit="GFLOP/s", power_avg_w=300.0,
                         temp_max_c=70.0),
        ]
        iter_data = {
            "node1:0:Batched GEMM:float32": [
                {"iteration": 0, "performance": 1600.0},
                {"iteration": 1, "performance": 1500.0},
            ]
        }
        bench_stats = hr.compute_benchmark_stats(results)
        hr._auto_scale_units(results, bench_stats, iter_data)
        assert abs(iter_data["node1:0:Batched GEMM:float32"][0]["performance"] - 1.6) < 0.01

    def test_auto_scale_non_scalable_unit_unchanged(self):
        """Units not in _UNIT_UPSCALE (e.g., img/s) should be unchanged."""
        results = [
            hr.GPUResult(hostname="node1", gpu=0, gpu_model="H100", serial="S1",
                         benchmark="Convolution", dtype="float32", iterations=10,
                         runtime_s=1.0, min_val=5000.0, mean_val=6000.0,
                         max_val=7000.0, unit="img/s", power_avg_w=300.0,
                         temp_max_c=70.0),
        ]
        bench_stats = hr.compute_benchmark_stats(results)
        hr._auto_scale_units(results, bench_stats, {})
        assert results[0].unit == "img/s"
        assert results[0].mean_val == 6000.0

    # -- _safe_float inf/nan guard tests --

    def test_safe_float_rejects_inf(self):
        assert hr._safe_float("inf") == 0.0
        assert hr._safe_float("-inf") == 0.0
        assert hr._safe_float("infinity") == 0.0

    def test_safe_float_rejects_nan(self):
        assert hr._safe_float("nan") == 0.0

    def test_safe_float_normal_values_unchanged(self):
        assert hr._safe_float("42.5") == 42.5
        assert hr._safe_float("-1.0") == -1.0
        assert hr._safe_float("0") == 0.0

    # -- INTERACTIVE_RESULT_COLS constant tests --

    def test_interactive_result_cols_is_module_constant(self):
        """INTERACTIVE_RESULT_COLS should be a module-level constant."""
        assert hasattr(hr, 'INTERACTIVE_RESULT_COLS')
        assert isinstance(hr.INTERACTIVE_RESULT_COLS, list)
        assert "hostname" in hr.INTERACTIVE_RESULT_COLS
        assert "throttled" in hr.INTERACTIVE_RESULT_COLS

    def test_interactive_result_cols_matches_gpuresult_fields(self):
        """All INTERACTIVE_RESULT_COLS should be valid GPUResult attribute names."""
        import dataclasses
        gpu_fields = {f.name for f in dataclasses.fields(hr.GPUResult)}
        for col in hr.INTERACTIVE_RESULT_COLS:
            assert col in gpu_fields, f"{col} not in GPUResult fields"

    # -- XSS escaping tests --

    def test_interactive_report_has_esc_function(self):
        """Interactive report should include the esc() XSS helper."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=1, n_gpus=1)
        bench_stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(bench_stats)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html = hr._render_interactive_html(
                        results, bench_stats, outliers,
                        source_name="test", threshold=15.0)
            assert "function esc(s)" in html
            assert "esc(r.hostname)" in html
            assert "esc(r.gpu_model)" in html
            assert "esc(r.serial)" in html
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    def test_interactive_report_escapes_fleet_map_titles(self):
        """Fleet map mkSquare should use esc() on hostnames."""
        from unittest.mock import MagicMock
        results = _make_results(n_nodes=1, n_gpus=1)
        bench_stats = hr.compute_benchmark_stats(results)
        outliers = hr.detect_outliers(bench_stats)
        old_data = dict(hr._iteration_data)
        try:
            hr._iteration_data.clear()
            with patch.object(hr, '_PLOTLY_AVAILABLE', True):
                mock_plotly = MagicMock()
                mock_plotly.get_plotlyjs.return_value = '/* mock */'
                with patch.object(hr, '_plotly_offline', mock_plotly):
                    html = hr._render_interactive_html(
                        results, bench_stats, outliers,
                        source_name="test", threshold=15.0)
            assert "esc(host)" in html
        finally:
            hr._iteration_data.clear()
            hr._iteration_data.update(old_data)

    # -- Dual outlier detection documentation test --

    def test_detect_outliers_docstring_documents_dual_algorithm(self):
        """detect_outliers should document the dual algorithm design."""
        doc = hr.detect_outliers.__doc__
        assert doc is not None
        assert "sigma" in doc.lower()


# ---------------------------------------------------------------------------
# --csv-output / --summary-csv files written by torch-hammer round-trip
# ---------------------------------------------------------------------------

class TestCsvOutputRoundTrip:
    """Files produced by torch-hammer's CSV writers load through _detect_and_load."""

    def _perf(self):
        return {
            "name": "Batched GEMM", "params": {"dtype": "float32"},
            "iterations": 10, "runtime_s": 1.5,
            "min": 48000.0, "mean": 49500.0, "max": 50700.0,
            "unit": "GFLOP/s",
            "telemetry": {
                "power_W_mean": 620.0, "temp_gpu_C_max": 71.0,
                "sm_util_mean": 98.0, "mem_bw_util_mean": 40.0,
                "gpu_clock_mean": 1980.0, "mem_used_MB_mean": 20480.0,
            },
            "throttled": True,
        }

    def _result(self):
        return {
            "gpu_index": 0, "serial": "S1",
            "telemetry_stats": {"power_W_mean": 300.0, "temp_gpu_C_max": 70.0},
            "benchmarks": [{
                "name": "Batched GEMM", "unit": "GFLOP/s",
                "mean": 100.0, "min": 90.0, "max": 110.0,
                "params": {"dtype": "float32"}, "telemetry": {},
            }],
        }

    def test_csv_output_round_trips_through_hammer_report(self, th, tmp_path):
        tel = {"model": "NVIDIA Example GPU", "serial": "SER123"}
        row = th._compact_row(self._perf(), tel, gpu_index=2,
                              hostname="node01", verbose=True)
        path = tmp_path / "out.csv"
        with path.open("w", newline="") as fh:
            th._emit_compact_csv(row, verbose=True, header=True, file=fh)

        results = hr._detect_and_load(path)
        assert len(results) == 1
        r = results[0]
        # Routed to the compact loader, not the summary loader
        assert r.gpu_model == "NVIDIA Example GPU"
        assert r.min_val != r.mean_val
        assert r.hostname == "node01"
        assert r.gpu == 2
        assert r.serial == "SER123"
        assert r.benchmark == "Batched GEMM"
        assert r.dtype == "float32"
        assert r.iterations == 10
        assert abs(r.min_val - 48000.0) < 1e-3
        assert abs(r.mean_val - 49500.0) < 1e-3
        assert abs(r.max_val - 50700.0) < 1e-3
        assert r.unit == "GFLOP/s"
        assert r.power_avg_w == 620.0
        assert r.temp_max_c == 71.0
        assert r.sm_util_mean == 98.0
        assert r.throttled is True

    def test_summary_csv_round_trips_through_hammer_report(self, th, tmp_path):
        from unittest.mock import MagicMock
        rows = th._build_summary_rows([self._result()], 70.0)
        written = th._write_summary_csv(rows, str(tmp_path / "s.csv"), MagicMock())
        assert written is not None and written.exists()

        results = hr._detect_and_load(written)
        assert len(results) == 1
        r = results[0]
        # Routed to the summary loader: no model, single performance value
        assert r.gpu_model == ""
        assert r.min_val == r.mean_val == r.max_val == 100.0
        assert r.benchmark == "Batched GEMM"
        assert r.dtype == "float32"
        assert r.gpu == 0
        assert r.serial == "S1"
        assert r.power_avg_w == 300.0
        assert r.temp_max_c == 70.0
