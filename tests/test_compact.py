# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Tests for --compact CSV output mode.

Covers:
  - CLI flag parsing
  - CSV column definitions (_compact_csv_columns)
  - CSV emission (_emit_compact_csv)
  - iterations / runtime_s in perf_summary via _log_summary
  - Logging suppression (WARNING level in compact mode)
  - Header control (single-GPU prints header, multi-GPU parent prints header)
  - Verbose extras (5 additional telemetry columns)
  - Edge cases (None perf, empty telemetry)
  - --csv-output: compact rows to a file (_compact_row, _unique_output_path,
    _start_csv_output, _append_csv_output_row, YAML parity)
  - --summary-csv helpers (_build_summary_rows, _write_summary_csv)
"""
import csv
import io
import logging
import re
import socket
import statistics
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────
# 1. CLI flag parsing
# ──────────────────────────────────────────────────────────────
class TestCompactCLIFlag:
    """--compact should be a boolean flag, default False."""

    def test_compact_default_false(self, default_args):
        assert default_args.compact is False

    def test_compact_flag_present(self, parser):
        args = parser.parse_args(["--compact"])
        assert args.compact is True

    def test_compact_with_verbose(self, parser):
        args = parser.parse_args(["--compact", "--verbose"])
        assert args.compact is True
        assert args.verbose is True

    def test_compact_with_batched_gemm(self, parser):
        args = parser.parse_args(["--compact", "--batched-gemm"])
        assert args.compact is True
        assert args.batched_gemm is True


# ──────────────────────────────────────────────────────────────
# 2. _compact_csv_columns
# ──────────────────────────────────────────────────────────────
class TestCompactCSVColumns:
    """Column helper returns correct lists for base / verbose modes."""

    def test_base_columns_count(self, th):
        cols = th._compact_csv_columns(verbose=False)
        assert len(cols) == 14

    def test_verbose_columns_count(self, th):
        cols = th._compact_csv_columns(verbose=True)
        assert len(cols) == 19

    def test_base_column_names(self, th):
        cols = th._compact_csv_columns(verbose=False)
        expected = [
            "hostname", "gpu", "gpu_model", "serial",
            "benchmark", "dtype", "iterations", "runtime_s",
            "min", "mean", "max", "unit",
            "power_avg_w", "temp_max_c",
        ]
        assert cols == expected

    def test_verbose_extra_columns(self, th):
        base = set(th._compact_csv_columns(verbose=False))
        full = set(th._compact_csv_columns(verbose=True))
        extras = full - base
        assert extras == {
            "sm_util_mean", "mem_bw_util_mean",
            "gpu_clock_mean", "mem_used_gb_mean", "throttled",
        }

    def test_columns_are_unique(self, th):
        for verbose in (False, True):
            cols = th._compact_csv_columns(verbose=verbose)
            assert len(cols) == len(set(cols)), "Duplicate column names"


# ──────────────────────────────────────────────────────────────
# 3. _emit_compact_csv
# ──────────────────────────────────────────────────────────────
class TestEmitCompactCSV:
    """_emit_compact_csv writes proper CSV to the given file handle."""

    def _make_row(self):
        return {
            "hostname": "node01",
            "gpu": 0,
            "gpu_model": "MI300X",
            "serial": "ABC123",
            "benchmark": "Batched GEMM",
            "dtype": "float32",
            "iterations": 100,
            "runtime_s": 12.345,
            "min": "1000.0000",
            "mean": "1050.0000",
            "max": "1100.0000",
            "unit": "GFLOP/s",
            "power_avg_w": "350.0",
            "temp_max_c": "72",
        }

    def test_row_without_header(self, th):
        buf = io.StringIO()
        th._emit_compact_csv(self._make_row(), verbose=False,
                             header=False, file=buf)
        output = buf.getvalue()
        assert "hostname" not in output  # no header
        lines = output.strip().split("\n")
        assert len(lines) == 1
        reader = csv.reader(io.StringIO(lines[0]))
        fields = next(reader)
        assert len(fields) == 14

    def test_row_with_header(self, th):
        buf = io.StringIO()
        th._emit_compact_csv(self._make_row(), verbose=False,
                             header=True, file=buf)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("hostname,")

    def test_csv_values_match(self, th):
        row = self._make_row()
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=False, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["hostname"] == "node01"
        assert parsed["benchmark"] == "Batched GEMM"
        assert parsed["unit"] == "GFLOP/s"
        assert parsed["mean"] == "1050.0000"

    def test_verbose_row_has_extra_columns(self, th):
        row = self._make_row()
        row.update({
            "sm_util_mean": "95",
            "mem_bw_util_mean": "80",
            "gpu_clock_mean": "1800",
            "mem_used_gb_mean": "60.50",
            "throttled": "false",
        })
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=True, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["sm_util_mean"] == "95"
        assert parsed["throttled"] == "false"
        assert len(parsed) == 19

    def test_missing_fields_become_empty(self, th):
        row = {"hostname": "node01", "benchmark": "GEMM"}
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=False, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["gpu"] == ""
        assert parsed["unit"] == ""


# ──────────────────────────────────────────────────────────────
# 4. _log_summary: iterations and runtime_s in perf_summary
# ──────────────────────────────────────────────────────────────
class TestLogSummaryIterationsRuntime:
    """_log_summary should include iterations and runtime_s in result dict."""

    def _make_tel(self):
        tel = MagicMock()
        tel.read.return_value = {"device_id": 0, "model": "TestGPU"}
        tel.get_stats.return_value = {}
        tel.throttle_detected = False
        return tel

    def _make_tel_thread(self):
        tt = MagicMock()
        tt.get_latest.return_value = {"device_id": 0}
        tt.lock = MagicMock()
        tt.iteration_samples = {}
        tt.all_samples = []
        return tt

    def test_iterations_count(self, th):
        import torch
        vals = [100.0, 110.0, 105.0, 108.0, 102.0]
        tel = self._make_tel()
        tel_thread = self._make_tel_thread()
        logger = logging.getLogger("test_iterations")
        logger.setLevel(logging.WARNING)  # suppress output

        result = th._log_summary(
            "TestBench", vals, "GFLOP/s", logger, tel,
            torch.device("cpu"), runtime_s=5.678,
            tel_thread=tel_thread,
        )
        assert result["iterations"] == 5
        assert result["runtime_s"] == 5.678

    def test_runtime_none(self, th):
        import torch
        vals = [100.0]
        tel = self._make_tel()
        tel_thread = self._make_tel_thread()
        logger = logging.getLogger("test_runtime_none")
        logger.setLevel(logging.WARNING)

        result = th._log_summary(
            "TestBench", vals, "GFLOP/s", logger, tel,
            torch.device("cpu"), tel_thread=tel_thread,
        )
        assert result["iterations"] == 1
        assert result["runtime_s"] is None

    def test_runtime_rounded(self, th):
        import torch
        vals = [100.0, 200.0]
        tel = self._make_tel()
        tel_thread = self._make_tel_thread()
        logger = logging.getLogger("test_runtime_round")
        logger.setLevel(logging.WARNING)

        result = th._log_summary(
            "TestBench", vals, "GFLOP/s", logger, tel,
            torch.device("cpu"), runtime_s=1.23456789,
            tel_thread=tel_thread,
        )
        assert result["runtime_s"] == 1.235  # rounded to 3 decimal places


# ──────────────────────────────────────────────────────────────
# 5. _log_summary: perf_summary basic structure
# ──────────────────────────────────────────────────────────────
class TestLogSummaryStructure:
    """_log_summary result dict has the expected keys and values."""

    def _make_tel(self):
        tel = MagicMock()
        tel.read.return_value = {"device_id": 0, "model": "TestGPU"}
        tel.get_stats.return_value = {}
        tel.throttle_detected = False
        return tel

    def _make_tel_thread(self):
        tt = MagicMock()
        tt.get_latest.return_value = {"device_id": 0}
        tt.lock = MagicMock()
        tt.iteration_samples = {}
        tt.all_samples = []
        return tt

    def test_min_mean_max(self, th):
        import torch
        vals = [10.0, 20.0, 30.0]
        tel = self._make_tel()
        tel_thread = self._make_tel_thread()
        logger = logging.getLogger("test_mmm")
        logger.setLevel(logging.WARNING)

        result = th._log_summary(
            "TestBench", vals, "GFLOP/s", logger, tel,
            torch.device("cpu"), tel_thread=tel_thread,
        )
        assert result["min"] == 10.0
        assert result["mean"] == 20.0
        assert result["max"] == 30.0
        assert result["unit"] == "GFLOP/s"
        assert result["name"] == "TestBench"


# ──────────────────────────────────────────────────────────────
# 6. Compact-mode logging suppression
# ──────────────────────────────────────────────────────────────
class TestCompactLoggingSuppression:
    """In compact mode, init_logging should use WARNING level on stderr."""

    def test_compact_sets_warning_level(self, th):
        args = types.SimpleNamespace(
            no_log=False, verbose=False, verbose_file_only=False,
            compact=True, log_file=None, log_dir=None,
        )
        logger = th.init_logging(args, gpu_index=0, tel_data=None)
        assert logger.level == logging.WARNING

    def test_compact_handler_is_stderr(self, th):
        args = types.SimpleNamespace(
            no_log=False, verbose=False, verbose_file_only=False,
            compact=True, log_file=None, log_dir=None,
        )
        logger = th.init_logging(args, gpu_index=0, tel_data=None)
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].stream is sys.stderr

    def test_non_compact_uses_info_level(self, th):
        args = types.SimpleNamespace(
            no_log=False, verbose=False, verbose_file_only=False,
            compact=False, log_file=None, log_dir=None,
        )
        logger = th.init_logging(args, gpu_index=0, tel_data=None)
        assert logger.level == logging.INFO

    def test_verbose_non_compact_uses_debug_level(self, th):
        args = types.SimpleNamespace(
            no_log=False, verbose=True, verbose_file_only=False,
            compact=False, log_file=None, log_dir=None,
        )
        logger = th.init_logging(args, gpu_index=0, tel_data=None)
        assert logger.level == logging.DEBUG


# ──────────────────────────────────────────────────────────────
# 7. Header control
# ──────────────────────────────────────────────────────────────
class TestCompactHeaderControl:
    """Header should be emitted exactly once per run."""

    def test_header_true_emits_header_line(self, th):
        buf = io.StringIO()
        row = {"hostname": "n01", "benchmark": "GEMM"}
        th._emit_compact_csv(row, verbose=False, header=True, file=buf)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == ",".join(th._compact_csv_columns(verbose=False))

    def test_header_false_no_header_line(self, th):
        buf = io.StringIO()
        row = {"hostname": "n01", "benchmark": "GEMM"}
        th._emit_compact_csv(row, verbose=False, header=False, file=buf)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 1
        assert not lines[0].startswith("hostname")

    def test_multiple_rows_single_header(self, th):
        buf = io.StringIO()
        row = {"hostname": "n01", "benchmark": "GEMM"}
        th._emit_compact_csv(row, verbose=False, header=True, file=buf)
        th._emit_compact_csv(row, verbose=False, header=False, file=buf)
        th._emit_compact_csv(row, verbose=False, header=False, file=buf)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 4  # 1 header + 3 data rows
        # Only first line should be the header
        header_count = sum(1 for l in lines if l.startswith("hostname,"))
        assert header_count == 1


# ──────────────────────────────────────────────────────────────
# 8. Throttle detection in compact CSV
# ──────────────────────────────────────────────────────────────
class TestCompactThrottle:
    """Throttle status should appear in verbose compact output."""

    def test_throttled_true_in_verbose_row(self, th):
        row = {
            "hostname": "n01", "gpu": 0, "benchmark": "GEMM",
            "throttled": "true",
        }
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=True, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["throttled"] == "true"

    def test_throttled_false_in_verbose_row(self, th):
        row = {
            "hostname": "n01", "gpu": 0, "benchmark": "GEMM",
            "throttled": "false",
        }
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=True, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["throttled"] == "false"


# ──────────────────────────────────────────────────────────────
# 9. Edge cases
# ──────────────────────────────────────────────────────────────
class TestCompactEdgeCases:
    """Edge cases: empty row, commas in values, special characters."""

    def test_empty_row(self, th):
        buf = io.StringIO()
        th._emit_compact_csv({}, verbose=False, header=False, file=buf)
        output = buf.getvalue().strip()
        # Should produce a row of empty fields
        reader = csv.reader(io.StringIO(output))
        fields = next(reader)
        assert len(fields) == 14
        assert all(f == "" for f in fields)

    def test_benchmark_name_with_special_chars(self, th):
        row = {"benchmark": "Schrödinger Equation", "hostname": "n01"}
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=False, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["benchmark"] == "Schrödinger Equation"

    def test_gpu_model_with_comma(self, th):
        """CSV quoting should handle commas in values."""
        row = {"gpu_model": "NVIDIA H100, 80GB", "hostname": "n01"}
        buf = io.StringIO()
        th._emit_compact_csv(row, verbose=False, header=True, file=buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        parsed = next(reader)
        assert parsed["gpu_model"] == "NVIDIA H100, 80GB"


# ──────────────────────────────────────────────────────────────
# 10. Compact does NOT emit per-iteration lines
# ──────────────────────────────────────────────────────────────
class TestCompactNoPerIterationOutput:
    """--compact alone must NOT trigger per-iteration verbose lines.

    Only --verbose gates per-iteration prn.emit() calls in benchmarks.
    --compact is summary-only.
    """

    def test_compact_without_verbose_no_per_iter(self, parser):
        args = parser.parse_args(["--compact", "--batched-gemm"])
        # verbose should still be False — compact doesn't imply verbose
        assert args.verbose is False
        assert args.compact is True


# ──────────────────────────────────────────────────────────────
# Shared fixtures for --csv-output / --summary-csv helpers
# ──────────────────────────────────────────────────────────────
def _fake_perf(**overrides):
    """A perf_summary dict as returned by _log_summary."""
    perf = {
        "name": "Batched GEMM",
        "params": {"dtype": "float32"},
        "iterations": 10,
        "runtime_s": 1.5,
        "min": 48000.0, "mean": 49500.0, "max": 50700.0,
        "unit": "GFLOP/s",
        "telemetry": {
            "power_W_mean": 620.0, "temp_gpu_C_max": 71.0,
            "sm_util_mean": 98.0, "mem_bw_util_mean": 40.0,
            "gpu_clock_mean": 1980.0, "mem_used_MB_mean": 20480.0,
        },
        "throttled": False,
    }
    perf.update(overrides)
    return perf


def _fake_tel():
    return {"model": "NVIDIA Example GPU", "serial": "SER123"}


def _fake_result(gpu_index=0, serial="S1", mean=100.0, throttled=False,
                 efficiency_pct=None, telemetry_stats=None):
    """A run_single_gpu() result dict with a single Batched GEMM benchmark."""
    bench = {
        "name": "Batched GEMM", "unit": "GFLOP/s",
        "mean": mean, "min": mean * 0.9, "max": mean * 1.1,
        "params": {"dtype": "float32"},
        "telemetry": {"throttled": True} if throttled else {},
    }
    if efficiency_pct is not None:
        bench["efficiency_pct"] = efficiency_pct
    if telemetry_stats is None:
        telemetry_stats = {"power_W_mean": 300.0, "temp_gpu_C_max": 70.0}
    return {"gpu_index": gpu_index, "serial": serial,
            "telemetry_stats": telemetry_stats, "benchmarks": [bench]}


SUMMARY_FIELDNAMES = [
    "test", "dtype", "gpu", "serial", "performance", "unit",
    "power_avg_W", "temp_max_C", "status", "notes",
]


# ──────────────────────────────────────────────────────────────
# 11. --csv-output: compact rows to a file, stdout untouched
# ──────────────────────────────────────────────────────────────
class TestCsvOutput:
    """--csv-output writes the compact schema to a per-node file."""

    def test_csv_output_flag_parses(self, parser):
        assert parser.parse_args([]).csv_output is None
        args = parser.parse_args(["--csv-output", "o.csv"])
        assert args.csv_output == "o.csv"

    def test_csv_output_from_yaml_global(self, th, parser):
        """csv_output under global: applies when the CLI did not set it."""
        orig = sys.argv
        sys.argv = ["torch-hammer.py"]  # no CLI flags
        try:
            args = parser.parse_args([])
            args = th.apply_config_to_args(args, {"global": {"csv_output": "o.csv"}})
        finally:
            sys.argv = orig
        assert args.csv_output == "o.csv"

    def test_unique_output_path(self, th, monkeypatch, tmp_path):
        """<hostname>_<timestamp> goes before the suffix; the directory is kept."""
        monkeypatch.setattr(socket, "gethostname", lambda: "node01.example")
        result = th._unique_output_path(str(tmp_path / "out.csv"))
        assert result.parent == tmp_path
        assert re.match(r"^out_node01_\d{8}_\d{6}\.csv$", result.name)

    @pytest.mark.parametrize("verbose", [False, True])
    def test_compact_row_keys_match_columns(self, th, verbose):
        row = th._compact_row(_fake_perf(), _fake_tel(), 0, "h", verbose)
        assert list(row.keys()) == th._compact_csv_columns(verbose)

    def test_compact_row_formats_values(self, th):
        row = th._compact_row(_fake_perf(), _fake_tel(), 3, "node01", verbose=True)
        assert row["hostname"] == "node01"
        assert row["gpu"] == 3
        assert row["gpu_model"] == "NVIDIA Example GPU"
        assert row["mean"] == "49500.0000"
        assert row["power_avg_w"] == "620.0"
        assert row["temp_max_c"] == "71"
        assert row["mem_used_gb_mean"] == "20.00"
        assert row["throttled"] == "false"

    def test_emit_compact_row_identical_to_stdout_and_file(self, th, monkeypatch):
        """The same row must serialise identically to stdout and to a file."""
        row = th._compact_row(_fake_perf(), _fake_tel(), 0, "node01", verbose=False)
        stdout_buf, file_buf = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout_buf)
        th._emit_compact_csv(row, verbose=False, header=False)
        th._emit_compact_csv(row, verbose=False, header=False, file=file_buf)
        assert stdout_buf.getvalue() == file_buf.getvalue()
        assert stdout_buf.getvalue().count("\n") == 1

    def test_start_csv_output_writes_header_only(self, th, tmp_path):
        path = th._start_csv_output(str(tmp_path / "o.csv"), verbose=True)
        written = list(tmp_path.glob("o_*.csv"))
        assert written == [Path(path)]
        lines = written[0].read_text().splitlines()
        assert lines == [",".join(th._compact_csv_columns(verbose=True))]

    def test_append_csv_output_row_appends_without_header(self, th, tmp_path):
        path = th._start_csv_output(str(tmp_path / "o.csv"))
        row = th._compact_row(_fake_perf(), _fake_tel(), 0, "node01", verbose=False)
        th._append_csv_output_row(row, path)
        th._append_csv_output_row(row, path)
        with open(path, newline="") as fh:
            parsed = list(csv.DictReader(fh))
        assert len(parsed) == 2
        assert parsed[0]["benchmark"] == "Batched GEMM"
        assert parsed[1]["mean"] == "49500.0000"


# ──────────────────────────────────────────────────────────────
# 12. --summary-csv helpers (single- AND multi-GPU)
# ──────────────────────────────────────────────────────────────
class TestSummaryCsv:
    """_build_summary_rows / _write_summary_csv behave like the summary table."""

    def test_build_summary_rows_single_result(self, th):
        """Regression for issue #28: one GPU must still yield summary rows."""
        rows = th._build_summary_rows([_fake_result()], 70.0)
        assert len(rows) == 1
        row = rows[0]
        assert list(row.keys()) == SUMMARY_FIELDNAMES
        assert row["test"] == "Batched GEMM"
        assert row["dtype"] == "float32"
        assert row["gpu"] == 0
        assert row["serial"] == "S1"
        assert row["performance"] == 100.0
        assert row["unit"] == "GFLOP/s"
        assert row["power_avg_W"] == 300.0  # falls back to result telemetry_stats
        assert row["temp_max_C"] == 70.0
        assert row["status"] == "PASS"
        assert row["notes"] == ""

    def test_build_summary_rows_throttled_fastest_gets_no_fastest_note(self, th):
        """A throttled fastest GPU is FAIL/Throttled; nobody else is 'Fastest'."""
        results = [_fake_result(0, "S0", mean=100.0),
                   _fake_result(1, "S1", mean=120.0, throttled=True)]
        rows = th._build_summary_rows(results, 70.0)
        assert [(r["gpu"], r["status"], r["notes"]) for r in rows] == [
            (0, "PASS", ""),
            (1, "FAIL", "Throttled"),
        ]

    def test_build_summary_rows_marks_fastest_clean_gpu(self, th):
        results = [_fake_result(0, "S0", mean=100.0, throttled=True),
                   _fake_result(1, "S1", mean=120.0)]
        rows = th._build_summary_rows(results, 70.0)
        assert [(r["gpu"], r["status"], r["notes"]) for r in rows] == [
            (0, "FAIL", "Throttled"),
            (1, "PASS", "Fastest"),
        ]

    def test_build_summary_rows_orders_by_benchmark_then_gpu(self, th):
        """Rows follow the table order (benchmark position, then GPU) and
        low efficiency is flagged; per-benchmark telemetry wins over the
        result-level fallback."""
        fft = {"name": "FFT", "unit": "GFLOP/s", "mean": 10.0, "min": 9.0,
               "max": 11.0, "params": {"dtype": "float32"}, "telemetry": {}}
        gpu1 = _fake_result(1, "S1", mean=120.0, efficiency_pct=50.0,
                            telemetry_stats={"power_W_mean": 250.0, "temp_gpu_C_max": 65.0})
        gpu1["benchmarks"][0]["telemetry"] = {"power_W_mean": 310.0}
        gpu1["benchmarks"].append(fft)
        gpu0 = _fake_result(0, "S0", mean=100.0, telemetry_stats={})
        gpu0["benchmarks"].append(dict(fft, mean=12.0))
        rows = th._build_summary_rows([gpu1, gpu0], 70.0)
        assert [(r["test"], r["gpu"]) for r in rows] == [
            ("Batched GEMM", 0), ("Batched GEMM", 1), ("FFT", 0), ("FFT", 1),
        ]
        assert rows[1]["status"] == "FAIL"
        assert rows[1]["notes"] == "Low efficiency (50%)"
        assert rows[1]["power_avg_W"] == 310.0  # per-benchmark telemetry
        assert rows[1]["temp_max_C"] == 65.0    # result-level fallback
        assert rows[2]["notes"] == "Fastest"
        assert rows[2]["power_avg_W"] == 0      # no telemetry anywhere
        assert rows[3]["power_avg_W"] == 250.0

    def test_build_summary_rows_empty_results(self, th):
        assert th._build_summary_rows([], 70.0) == []

    def test_write_summary_csv_writes_file(self, th, tmp_path):
        rows = th._build_summary_rows([_fake_result()], 70.0)
        log = MagicMock()
        written = th._write_summary_csv(rows, str(tmp_path / "s.csv"), log)
        assert list(tmp_path.glob("s_*_*.csv")) == [written]
        with open(written, newline="") as fh:
            reader = csv.DictReader(fh)
            parsed = list(reader)
            assert reader.fieldnames == SUMMARY_FIELDNAMES
        assert len(parsed) == 1
        assert parsed[0]["performance"] == "100.0"
        assert parsed[0]["status"] == "PASS"
        log.info.assert_called_once()
        assert str(written) in log.info.call_args[0][0]

    def test_write_summary_csv_warns_on_unwritable_path(self, th, tmp_path):
        rows = th._build_summary_rows([_fake_result()], 70.0)
        log = MagicMock()
        written = th._write_summary_csv(rows, str(tmp_path / "missing_dir" / "s.csv"), log)
        assert written is None
        log.warning.assert_called_once()
        assert "Failed to export CSV summary" in log.warning.call_args[0][0]
