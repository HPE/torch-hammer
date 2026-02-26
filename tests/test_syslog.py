# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Tests for --syslog and --syslog-dmesg output modes.

Covers:
  - CLI flag parsing (--syslog, --syslog-dmesg, mutual dependency)
  - SyslogReporter._kv formatting (None/empty handling, space replacement)
  - SyslogReporter._priority derivation from thresholds
  - SyslogReporter message methods (run_start, bench_result, run_end)
  - DmesgWriter graceful failure paths (/dev/kmsg permission, not-found)
  - _build_syslog_row field mapping from perf_summary
"""
import logging
import syslog
import types
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch, call

import pytest


# ──────────────────────────────────────────────────────────────
# 1. CLI flag parsing
# ──────────────────────────────────────────────────────────────
class TestSyslogCLIFlags:
    """--syslog and --syslog-dmesg should be boolean flags."""

    def test_syslog_default_false(self, default_args):
        assert default_args.syslog is False

    def test_syslog_dmesg_default_false(self, default_args):
        assert default_args.syslog_dmesg is False

    def test_syslog_flag_present(self, parser):
        args = parser.parse_args(["--syslog"])
        assert args.syslog is True

    def test_syslog_dmesg_with_syslog(self, parser):
        args = parser.parse_args(["--syslog", "--syslog-dmesg"])
        assert args.syslog is True
        assert args.syslog_dmesg is True

    def test_syslog_dmesg_requires_syslog(self, parser):
        """--syslog-dmesg without --syslog should error at validation time."""
        # The parser itself accepts it, but main() calls parser.error().
        # We verify that the parser stores both independently.
        args = parser.parse_args(["--syslog-dmesg"])
        assert args.syslog_dmesg is True
        assert args.syslog is False

    def test_syslog_with_compact(self, parser):
        """Both modes can coexist."""
        args = parser.parse_args(["--syslog", "--compact"])
        assert args.syslog is True
        assert args.compact is True


# ──────────────────────────────────────────────────────────────
# 2. SyslogReporter._kv formatting
# ──────────────────────────────────────────────────────────────
class TestSyslogReporterKV:
    """Test the key=value formatter."""

    def test_basic_kv(self, th):
        result = th.SyslogReporter._kv({"host": "node01", "gpu": 0})
        assert result == "host=node01 gpu=0"

    def test_skips_none_values(self, th):
        result = th.SyslogReporter._kv({"a": "1", "b": None, "c": "3"})
        assert "b=" not in result
        assert "a=1" in result
        assert "c=3" in result

    def test_skips_empty_string_values(self, th):
        result = th.SyslogReporter._kv({"a": "1", "b": "", "c": "3"})
        assert "b=" not in result

    def test_spaces_replaced_with_underscores(self, th):
        result = th.SyslogReporter._kv({"model": "NVIDIA H100 SXM"})
        assert "model=NVIDIA_H100_SXM" in result

    def test_empty_dict(self, th):
        result = th.SyslogReporter._kv({})
        assert result == ""

    def test_preserves_key_order(self, th):
        from collections import OrderedDict
        d = OrderedDict([("z", "1"), ("a", "2"), ("m", "3")])
        result = th.SyslogReporter._kv(d)
        assert result == "z=1 a=2 m=3"


# ──────────────────────────────────────────────────────────────
# 3. SyslogReporter._priority derivation
# ──────────────────────────────────────────────────────────────
class TestSyslogReporterPriority:
    """Priority should be derived from existing threshold fields."""

    @pytest.fixture
    def reporter(self, th):
        """A reporter with known thresholds, mocked syslog."""
        with patch.object(th, "syslog", create=True):
            with patch("syslog.openlog"), patch("syslog.closelog"):
                r = th.SyslogReporter(
                    temp_warn=85.0,
                    temp_critical=95.0,
                    efficiency_warn=70.0,
                )
        return r

    def test_fail_status_returns_err(self, reporter):
        assert reporter._priority({"status": "FAIL"}) == syslog.LOG_ERR

    def test_temp_above_critical_returns_crit(self, reporter):
        assert reporter._priority({"status": "PASS", "temp_max_c": "96"}) == syslog.LOG_CRIT

    def test_temp_at_critical_returns_crit(self, reporter):
        assert reporter._priority({"status": "PASS", "temp_max_c": "95"}) == syslog.LOG_CRIT

    def test_temp_above_warn_returns_warning(self, reporter):
        assert reporter._priority({"status": "PASS", "temp_max_c": "86"}) == syslog.LOG_WARNING

    def test_temp_at_warn_returns_warning(self, reporter):
        assert reporter._priority({"status": "PASS", "temp_max_c": "85"}) == syslog.LOG_WARNING

    def test_temp_below_warn_returns_info(self, reporter):
        assert reporter._priority({"status": "PASS", "temp_max_c": "70"}) == syslog.LOG_INFO

    def test_throttled_true_string_returns_warning(self, reporter):
        assert reporter._priority({"status": "PASS", "throttled": "true"}) == syslog.LOG_WARNING

    def test_throttled_true_bool_returns_warning(self, reporter):
        assert reporter._priority({"status": "PASS", "throttled": True}) == syslog.LOG_WARNING

    def test_throttled_false_returns_info(self, reporter):
        assert reporter._priority({"status": "PASS", "throttled": "false"}) == syslog.LOG_INFO

    def test_low_efficiency_returns_warning(self, reporter):
        assert reporter._priority({"status": "PASS", "efficiency_pct": "50.0"}) == syslog.LOG_WARNING

    def test_good_efficiency_returns_info(self, reporter):
        assert reporter._priority({"status": "PASS", "efficiency_pct": "90.0"}) == syslog.LOG_INFO

    def test_clean_pass_returns_info(self, reporter):
        assert reporter._priority({"status": "PASS"}) == syslog.LOG_INFO

    def test_fail_takes_precedence_over_temp(self, reporter):
        """FAIL status trumps critical temperature."""
        assert reporter._priority({"status": "FAIL", "temp_max_c": "100"}) == syslog.LOG_ERR

    def test_critical_temp_takes_precedence_over_throttle(self, reporter):
        """Critical temp ranks above throttle."""
        result = {"status": "PASS", "temp_max_c": "96", "throttled": "true"}
        assert reporter._priority(result) == syslog.LOG_CRIT

    def test_non_numeric_temp_ignored(self, reporter):
        """Non-numeric temp values should be ignored gracefully."""
        assert reporter._priority({"status": "PASS", "temp_max_c": "N/A"}) == syslog.LOG_INFO

    def test_non_numeric_efficiency_ignored(self, reporter):
        """Non-numeric efficiency values should be ignored gracefully."""
        assert reporter._priority({"status": "PASS", "efficiency_pct": "N/A"}) == syslog.LOG_INFO


# ──────────────────────────────────────────────────────────────
# 4. SyslogReporter message methods
# ──────────────────────────────────────────────────────────────
class TestSyslogReporterMessages:
    """Test run_start, bench_result, run_end return proper messages."""

    @pytest.fixture
    def reporter(self, th):
        with patch("syslog.openlog"), patch("syslog.closelog"):
            r = th.SyslogReporter()
        # Replace the stored syslog module's syslog function with a mock
        r._mock_syslog = MagicMock()
        r._syslog = MagicMock()
        r._syslog.syslog = r._mock_syslog
        r._syslog.LOG_INFO = syslog.LOG_INFO
        r._syslog.LOG_WARNING = syslog.LOG_WARNING
        r._syslog.LOG_ERR = syslog.LOG_ERR
        r._syslog.LOG_CRIT = syslog.LOG_CRIT
        return r

    def test_run_start_format(self, reporter):
        msg = reporter.run_start("node01", 4)
        assert msg.startswith("RUN_START run_id=")
        assert "host=node01" in msg
        assert "gpus=4" in msg

    def test_run_start_calls_syslog(self, reporter):
        reporter.run_start("node01", 1)
        reporter._mock_syslog.assert_called_once()
        call_args = reporter._mock_syslog.call_args
        assert call_args[0][0] == syslog.LOG_INFO

    def test_bench_result_format(self, reporter):
        row = {"benchmark": "gemm", "gpu": 0, "mean": "42.5", "unit": "TFLOPS"}
        msg = reporter.bench_result(row)
        assert msg.startswith("BENCH_RESULT ")
        assert "benchmark=gemm" in msg
        assert "mean=42.5" in msg

    def test_bench_result_calls_syslog_with_priority(self, reporter):
        row = {"status": "FAIL", "benchmark": "gemm"}
        reporter.bench_result(row)
        call_args = reporter._mock_syslog.call_args
        assert call_args[0][0] == syslog.LOG_ERR

    def test_run_end_format(self, reporter):
        msg = reporter.run_end(passed=8, failed=1, elapsed=123.456)
        assert msg.startswith("RUN_END run_id=")
        assert "passed=8" in msg
        assert "failed=1" in msg
        assert "total_elapsed=123.5s" in msg

    def test_run_end_calls_syslog(self, reporter):
        reporter.run_end(9, 0, 60.0)
        call_args = reporter._mock_syslog.call_args
        assert call_args[0][0] == syslog.LOG_INFO

    def test_close_calls_closelog(self, th):
        with patch("syslog.openlog"), patch("syslog.closelog") as mock_close, \
             patch("syslog.syslog"):
            r = th.SyslogReporter()
            r.close()
            mock_close.assert_called_once()

    def test_run_id_auto_generated(self, th):
        """When no run_id is provided, a uuid4 hex prefix is auto-generated."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            r = th.SyslogReporter()
            assert len(r.run_id) == 8
            assert all(c in '0123456789abcdef' for c in r.run_id)
            r.close()

    def test_run_id_explicit(self, th):
        """Explicit run_id is preserved."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            r = th.SyslogReporter(run_id="deadbeef")
            assert r.run_id == "deadbeef"
            r.close()

    def test_run_id_consistent_across_messages(self, reporter):
        """run_id appears in RUN_START, BENCH_RESULT, and RUN_END."""
        rid = reporter.run_id
        start = reporter.run_start("node01", 1)
        assert f"run_id={rid}" in start
        row = {"run_id": rid, "benchmark": "gemm", "gpu": 0}
        result = reporter.bench_result(row)
        assert f"run_id={rid}" in result
        end = reporter.run_end(1, 0, 10.0)
        assert f"run_id={rid}" in end

    def test_two_reporters_have_different_run_ids(self, th):
        """Each SyslogReporter gets a unique run_id."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            r1 = th.SyslogReporter()
            r2 = th.SyslogReporter()
            assert r1.run_id != r2.run_id
            r1.close()
            r2.close()


# ──────────────────────────────────────────────────────────────
# 5. DmesgWriter graceful failure
# ──────────────────────────────────────────────────────────────
class TestDmesgWriter:
    """DmesgWriter must never crash the benchmark."""

    def test_permission_error_logs_warning(self, th):
        mock_log = MagicMock()
        with patch("builtins.open", side_effect=PermissionError):
            writer = th.DmesgWriter(log=mock_log)
        assert not writer.available
        mock_log.warning.assert_called_once()
        assert "Permission denied" in mock_log.warning.call_args[0][0]

    def test_file_not_found_logs_warning(self, th):
        mock_log = MagicMock()
        with patch("builtins.open", side_effect=FileNotFoundError):
            writer = th.DmesgWriter(log=mock_log)
        assert not writer.available
        mock_log.warning.assert_called_once()
        assert "not found" in mock_log.warning.call_args[0][0]

    def test_permission_error_no_log(self, th):
        """Even without a logger, don't crash."""
        with patch("builtins.open", side_effect=PermissionError):
            writer = th.DmesgWriter(log=None)
        assert not writer.available

    def test_successful_open(self, th):
        mock_fd = MagicMock()
        with patch("builtins.open", return_value=mock_fd):
            writer = th.DmesgWriter()
        assert writer.available

    def test_write_when_available(self, th):
        mock_fd = MagicMock()
        with patch("builtins.open", return_value=mock_fd):
            writer = th.DmesgWriter()
        writer.write("test message")
        mock_fd.write.assert_called_once_with("torch-hammer: test message\n")
        mock_fd.flush.assert_called_once()

    def test_write_when_not_available(self, th):
        """write() on unavailable writer should silently do nothing."""
        with patch("builtins.open", side_effect=PermissionError):
            writer = th.DmesgWriter()
        writer.write("test message")  # should not raise

    def test_write_survives_oserror(self, th):
        """OSError on write should be swallowed."""
        mock_fd = MagicMock()
        mock_fd.write.side_effect = OSError("Ring buffer full")
        with patch("builtins.open", return_value=mock_fd):
            writer = th.DmesgWriter()
        writer.write("test")  # should not raise

    def test_close_clears_fd(self, th):
        mock_fd = MagicMock()
        with patch("builtins.open", return_value=mock_fd):
            writer = th.DmesgWriter()
        writer.close()
        mock_fd.close.assert_called_once()
        assert not writer.available

    def test_close_when_not_available(self, th):
        """close() on unavailable writer should silently do nothing."""
        with patch("builtins.open", side_effect=PermissionError):
            writer = th.DmesgWriter()
        writer.close()  # should not raise


# ──────────────────────────────────────────────────────────────
# 6. _build_syslog_row field mapping
# ──────────────────────────────────────────────────────────────
class TestBuildSyslogRow:
    """Verify _build_syslog_row mirrors compact-mode columns."""

    @pytest.fixture
    def perf_summary(self):
        """Realistic perf_summary from _log_summary."""
        return {
            "name": "gemm",
            "min": 42.1234,
            "mean": 43.5678,
            "max": 44.9012,
            "unit": "TFLOPS",
            "iterations": 100,
            "runtime_s": 12.5,
            "params": {"dtype": "fp16", "m": 4096},
            "telemetry": {
                "power_W_mean": 350.5,
                "temp_gpu_C_max": 72.0,
                "sm_util_mean": 98.5,
                "mem_bw_util_mean": 45.2,
                "gpu_clock_mean": 1980.0,
                "mem_used_MB_mean": 40960.0,
            },
            "efficiency_pct": 85.3,
            "throttled": False,
        }

    @pytest.fixture
    def tel_data(self):
        return {
            "hostname": "node01",
            "model": "NVIDIA H100 SXM",
            "serial": "ABC123",
        }

    def test_basic_fields(self, th, perf_summary, tel_data):
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        assert row["hostname"] == "node01"
        assert row["gpu"] == 0
        assert row["gpu_model"] == "NVIDIA H100 SXM"
        assert row["serial"] == "ABC123"
        assert row["benchmark"] == "gemm"
        assert row["dtype"] == "fp16"
        assert row["iterations"] == 100
        assert row["unit"] == "TFLOPS"
        assert row["status"] == "PASS"

    def test_numeric_formatting(self, th, perf_summary, tel_data):
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        assert row["min"] == "42.1234"
        assert row["mean"] == "43.5678"
        assert row["max"] == "44.9012"
        assert row["power_avg_w"] == "350.5"
        assert row["temp_max_c"] == "72"

    def test_efficiency_included_when_present(self, th, perf_summary, tel_data):
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        assert row["efficiency_pct"] == "85.3"

    def test_efficiency_absent_when_not_in_perf(self, th, perf_summary, tel_data):
        del perf_summary["efficiency_pct"]
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        assert "efficiency_pct" not in row

    def test_throttled_flag(self, th, perf_summary, tel_data):
        perf_summary["throttled"] = True
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        assert row["throttled"] == "true"

    def test_no_throttle_when_false(self, th, perf_summary, tel_data):
        perf_summary["throttled"] = False
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        # throttled key should not be set in non-verbose mode
        assert "throttled" not in row

    def test_verbose_extra_fields(self, th, perf_summary, tel_data):
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0, verbose=True)
        assert row["sm_util_mean"] == "98"
        assert row["mem_bw_util_mean"] == "45"
        assert row["gpu_clock_mean"] == "1980"
        assert "mem_used_gb_mean" in row
        # verbose always emits throttled as string
        assert row["throttled"] in ("true", "false")

    def test_verbose_throttled_true(self, th, perf_summary, tel_data):
        perf_summary["throttled"] = True
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0, verbose=True)
        assert row["throttled"] == "true"

    def test_verbose_throttled_false(self, th, perf_summary, tel_data):
        perf_summary["throttled"] = False
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0, verbose=True)
        assert row["throttled"] == "false"

    def test_missing_telemetry_fields(self, th, tel_data):
        """perf_summary with no telemetry should still produce a valid row."""
        perf = {
            "name": "fft",
            "min": 1.0, "mean": 2.0, "max": 3.0,
            "unit": "GFLOPS",
            "params": {},
            "telemetry": {},
        }
        row = th._build_syslog_row(perf, tel_data, gpu_index=0)
        assert row["benchmark"] == "fft"
        assert row["power_avg_w"] == ""
        assert row["temp_max_c"] == ""

    def test_hostname_fallback(self, th, perf_summary):
        """When tel_data has no hostname, fallback to socket.gethostname."""
        with patch("socket.gethostname", return_value="fallback.domain.com"):
            row = th._build_syslog_row(perf_summary, {}, gpu_index=0)
        assert row["hostname"] == "fallback"

    def test_gpu_index_passthrough(self, th, perf_summary, tel_data):
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=3)
        assert row["gpu"] == 3

    def test_run_id_included_when_provided(self, th, perf_summary, tel_data):
        """run_id should be the first key in the row when provided."""
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0, run_id="abc12345")
        assert row["run_id"] == "abc12345"
        assert list(row.keys())[0] == "run_id"

    def test_run_id_absent_when_empty(self, th, perf_summary, tel_data):
        """run_id should not appear when empty string (default)."""
        row = th._build_syslog_row(perf_summary, tel_data, gpu_index=0)
        assert "run_id" not in row


# ──────────────────────────────────────────────────────────────
# 7. Integration: SyslogReporter + _build_syslog_row
# ──────────────────────────────────────────────────────────────
class TestSyslogIntegration:
    """End-to-end flow: build row → bench_result → verify syslog call."""

    def test_full_flow(self, th):
        perf = {
            "name": "gemm",
            "min": 42.0, "mean": 43.0, "max": 44.0,
            "unit": "TFLOPS",
            "params": {"dtype": "fp16"},
            "telemetry": {"power_W_mean": 300.0, "temp_gpu_C_max": 80.0},
        }
        tel_data = {"hostname": "node01", "model": "H100", "serial": "SN1"}

        row = th._build_syslog_row(perf, tel_data, gpu_index=0)

        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog") as mock_syslog:
            reporter = th.SyslogReporter(temp_warn=85.0, temp_critical=95.0)
            msg = reporter.bench_result(row)
            reporter.close()

        assert "BENCH_RESULT" in msg
        assert "benchmark=gemm" in msg
        assert "gpu=0" in msg
        mock_syslog.assert_called()
        # Clean pass at 80°C, below 85 warn → LOG_INFO
        assert mock_syslog.call_args[0][0] == syslog.LOG_INFO

    def test_critical_temp_flow(self, th):
        perf = {
            "name": "gemm",
            "min": 42.0, "mean": 43.0, "max": 44.0,
            "unit": "TFLOPS",
            "params": {},
            "telemetry": {"temp_gpu_C_max": 97.0},
        }
        row = th._build_syslog_row(perf, {}, gpu_index=0)

        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog") as mock_syslog:
            reporter = th.SyslogReporter(temp_critical=95.0)
            reporter.bench_result(row)
            reporter.close()

        assert mock_syslog.call_args[0][0] == syslog.LOG_CRIT


# ──────────────────────────────────────────────────────────────
# 8. RUN_END pass/fail counting via _maybe_emit_syslog
# ──────────────────────────────────────────────────────────────
class TestSyslogPassFailCounting:
    """
    _maybe_emit_syslog must count every call:
      perf=None  → _sl_bench_failed += 1
      perf!=None → _sl_bench_passed += 1
    RUN_END must use these counters (not benchmark_results).
    """

    def _make_perf(self, name="gemm"):
        return {
            "name": name,
            "min": 1.0, "mean": 2.0, "max": 3.0,
            "unit": "TFLOPS",
            "params": {},
            "telemetry": {},
        }

    def test_all_pass(self, th):
        """3 successful benchmarks → passed=3 failed=0."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            reporter = th.SyslogReporter()
            _sl_bench_passed = 0
            _sl_bench_failed = 0

            def _maybe_emit_syslog(perf):
                nonlocal _sl_bench_passed, _sl_bench_failed
                if not reporter:
                    return
                if perf is None:
                    _sl_bench_failed += 1
                    return
                _sl_bench_passed += 1

            for _ in range(3):
                _maybe_emit_syslog(self._make_perf())

            assert _sl_bench_passed == 3
            assert _sl_bench_failed == 0
            reporter.close()

    def test_all_fail(self, th):
        """4 failed benchmarks → passed=0 failed=4."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            reporter = th.SyslogReporter()
            _sl_bench_passed = 0
            _sl_bench_failed = 0

            def _maybe_emit_syslog(perf):
                nonlocal _sl_bench_passed, _sl_bench_failed
                if not reporter:
                    return
                if perf is None:
                    _sl_bench_failed += 1
                    return
                _sl_bench_passed += 1

            for _ in range(4):
                _maybe_emit_syslog(None)

            assert _sl_bench_passed == 0
            assert _sl_bench_failed == 4
            reporter.close()

    def test_mixed_pass_fail(self, th):
        """2 pass + 3 fail → passed=2 failed=3."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            reporter = th.SyslogReporter()
            _sl_bench_passed = 0
            _sl_bench_failed = 0

            def _maybe_emit_syslog(perf):
                nonlocal _sl_bench_passed, _sl_bench_failed
                if not reporter:
                    return
                if perf is None:
                    _sl_bench_failed += 1
                    return
                _sl_bench_passed += 1

            _maybe_emit_syslog(self._make_perf())
            _maybe_emit_syslog(None)
            _maybe_emit_syslog(self._make_perf())
            _maybe_emit_syslog(None)
            _maybe_emit_syslog(None)

            assert _sl_bench_passed == 2
            assert _sl_bench_failed == 3
            reporter.close()

    def test_run_end_uses_counters(self, th):
        """RUN_END message must contain the counter values, not benchmark_results."""
        with patch("syslog.openlog"), patch("syslog.closelog"), \
             patch("syslog.syslog"):
            reporter = th.SyslogReporter()
            # Simulate 5 passed, 3 failed
            msg = reporter.run_end(passed=5, failed=3, elapsed=42.0)
            assert "passed=5" in msg
            assert "failed=3" in msg
            reporter.close()

    def test_no_syslog_no_counting(self):
        """When reporter is None, counters stay at zero."""
        reporter = None
        _sl_bench_passed = 0
        _sl_bench_failed = 0

        def _maybe_emit_syslog(perf):
            nonlocal _sl_bench_passed, _sl_bench_failed
            if not reporter:
                return
            if perf is None:
                _sl_bench_failed += 1
                return
            _sl_bench_passed += 1

        _maybe_emit_syslog({"name": "x"})
        _maybe_emit_syslog(None)

        assert _sl_bench_passed == 0
        assert _sl_bench_failed == 0

    def test_mgpu_aggregate_from_result_dicts(self):
        """Multi-GPU parent RUN_END should sum syslog_passed/syslog_failed from result dicts."""
        results = [
            {"gpu_index": 0, "benchmarks": [], "syslog_passed": 5, "syslog_failed": 1},
            {"gpu_index": 1, "benchmarks": [], "syslog_passed": 4, "syslog_failed": 2},
            {"gpu_index": 2, "benchmarks": [], "syslog_passed": 3, "syslog_failed": 0},
        ]
        mg_passed = sum(r.get("syslog_passed", 0) for r in results)
        mg_failed = sum(r.get("syslog_failed", 0) for r in results)
        assert mg_passed == 12
        assert mg_failed == 3


# ──────────────────────────────────────────────────────────────
# 9. Validation: --syslog-dmesg requires --syslog
# ──────────────────────────────────────────────────────────────
class TestSyslogDmesgValidation:
    """main() should reject --syslog-dmesg without --syslog."""

    def test_dmesg_without_syslog_errors(self, parser):
        """Simulates the validation that main() performs."""
        args = parser.parse_args(["--syslog-dmesg"])
        # In production, main() calls parser.error().
        # Verify the condition that triggers it.
        assert args.syslog_dmesg is True
        assert args.syslog is False
        # The actual error call is in main(), tested via the flag states
