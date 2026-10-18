# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Tests for melt mode (TIM settling).

Covers:
  - melt_converged: the pure convergence rule on synthetic trajectories
  - per-cycle telemetry classification (thermal vs power-brake throttling)
  - --stress-test sizing for the "melt" key
  - melt-first ordering of the config benchmark list
  - _summary_status / syslog / compact / JSON handling of a Melt result
  - TelemetryThread.clear_iteration_samples (no bleed into the next benchmark)
"""
import json
import logging
import types
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────
# Helpers: synthetic trajectories
# ──────────────────────────────────────────────────────────────
def _flat(gflops, t_from, t_to, step=1.0, throttled=False):
    """One record per *step* seconds at a constant GFLOPS."""
    t = t_from
    out = []
    while t <= t_to:
        out.append((t, gflops, throttled))
        t += step
    return out


def _step_trajectory(low=100.0, high=200.0, t_step=400.0, t_end=600.0):
    """Low plateau, then an abrupt jump to a high plateau at *t_step*."""
    return _flat(low, 0, t_step - 1) + _flat(high, t_step, t_end)


DEFAULTS = dict(min_duration_s=300.0, window_s=60.0, bin_s=10.0,
                plateau_cv_pct=1.0, settled_pct=98.0)


def _noisy(mean, t_from, t_to, step=1.0, cv_pct=2.0, seed=7):
    """Flat trajectory with gaussian per-cycle noise of the given CV (deterministic)."""
    import random
    rng = random.Random(seed)
    out, t = [], t_from
    while t <= t_to + 1e-9:
        out.append((round(t, 6), rng.gauss(mean, mean * cv_pct / 100.0), False))
        t += step
    return out


def _converged(th, records, now_s, **overrides):
    kwargs = dict(DEFAULTS, **overrides)
    return th.melt_converged(records, now_s, **kwargs)


# ──────────────────────────────────────────────────────────────
# 1. melt_converged
# ──────────────────────────────────────────────────────────────
class TestMeltConverged:

    def test_white_per_cycle_noise_on_a_flat_mean_settles(self, th):
        # The GH200 incident: 1.9 % per-cycle DVFS jitter on a plateau that
        # was flat for 90 min.  Per-bin means average the jitter away.
        records = _noisy(53400.0, 0, 600, step=0.25, cv_pct=2.0)
        settled, info = _converged(th, records, 600)
        assert settled is True
        assert info["cycle_cv_pct"] > 1.0
        assert info["window_cv"] < 0.5

    def test_never_settles_when_the_plateau_itself_wanders(self, th):
        # Level changes at the bin scale (10 s) are real instability, not jitter.
        records = [(t, 100.0 if (t // 10) % 2 else 120.0, False) for t in range(0, 601)]
        for now in (300, 450, 600):
            settled, info = _converged(th, records, now)
            assert settled is False
            assert "CV" in info["reason"]

    def test_step_inside_the_window_is_not_a_plateau(self, th):
        # 3 % step half-way through the trailing window: bins before and after
        # differ, so the bin CV exceeds 1 % even though each half is flat.
        records = _flat(100.0, 0, 570) + _flat(103.0, 571, 600)
        settled, info = _converged(th, records, 600)
        assert settled is False
        assert "CV" in info["reason"]
        assert info["window_cv"] > 1.0

    def test_bins_in_window_tolerates_floating_point_rounding(self, th):
        assert th._bins_in_window(0.6, 0.2) == 3      # 0.6 // 0.2 is 2.0 in binary floating point
        assert th._bins_in_window(60.0, 10.0) == 6
        assert th._bins_in_window(60.0, 25.0) == 2
        assert th._bins_in_window(5.0, 10.0) == 1

    def test_too_few_bins_is_not_settled(self, th):
        # Only the last 15 s of the window have cycles -> 2 non-empty bins.
        records = _flat(200.0, 0, 500) + _flat(200.0, 585, 600)
        settled, info = _converged(th, records, 600)
        assert settled is False
        assert "not enough bins" in info["reason"]
        assert info["bins"] == 2

    def test_rules_failed_lists_every_failing_rule(self, th):
        records = _flat(150.0, 0, 600)
        _, info = _converged(th, records, 600, best_window_mean=200.0, target_gflops=1000.0)
        assert info["rules_failed"] == ["below_best", "below_target"]
        _, info = _converged(th, _flat(200.0, 0, 600), 600)
        assert info["rules_failed"] == []

    def test_settles_once_window_is_entirely_on_the_plateau(self, th):
        records = _step_trajectory()
        # window [399, 459] still contains the last low point -> not settled
        settled, info = _converged(th, records, 459)
        assert settled is False
        # window [400, 460] is flat at 200 -> settled
        settled, info = _converged(th, records, 460)
        assert settled is True
        assert info["window_mean"] == pytest.approx(200.0)
        assert info["window_cv"] == pytest.approx(0.0)
        assert info["cycle_cv_pct"] == pytest.approx(0.0)
        assert info["reason"] == "settled"

    def test_thermal_throttle_inside_window_blocks_settling(self, th):
        records = _flat(200.0, 0, 600)
        records[580] = (580.0, 200.0, True)
        settled, info = _converged(th, records, 600)
        assert settled is False
        assert info["throttle_count"] == 1
        assert "throttl" in info["reason"].lower()

    def test_thermal_throttle_outside_window_is_ignored(self, th):
        records = _flat(200.0, 0, 600)
        records[500] = (500.0, 200.0, True)   # 100 s before now, window is 60 s
        settled, info = _converged(th, records, 600)
        assert settled is True
        assert info["throttle_count"] == 0

    def test_plateau_below_settled_pct_of_best_is_not_settled(self, th):
        records = _flat(200.0, 0, 600)
        settled, info = _converged(th, records, 600, best_window_mean=210.0)
        assert settled is False            # 200 / 210 = 95.2 % < 98 %
        assert "best" in info["reason"]
        assert info["best"] == pytest.approx(210.0)

    def test_plateau_within_settled_pct_of_best_is_settled(self, th):
        records = _flat(200.0, 0, 600)
        settled, _ = _converged(th, records, 600, best_window_mean=203.0)
        assert settled is True             # 200 / 203 = 98.5 %

    def test_best_window_mean_is_tracked_upwards(self, th):
        records = _flat(200.0, 0, 600)
        _, info = _converged(th, records, 600, best_window_mean=150.0)
        assert info["best"] == pytest.approx(200.0)
        _, info = _converged(th, records, 600, best_window_mean=None)
        assert info["best"] == pytest.approx(200.0)

    def _track_best(self, th, records, nows, **overrides):
        """Drive melt_converged the way _run_melt_cycles does, carrying best forward."""
        best = None
        outcomes = []
        for now in nows:
            settled, info = _converged(th, records, now, best_window_mean=best, **overrides)
            best = info["best"]
            outcomes.append((now, settled, info))
        return outcomes

    def test_cold_boost_before_min_duration_does_not_set_the_bar(self, th):
        # Already-settled GPU: first minute of cold boost at 100, then flat at 97
        # (3 % below). The boost window is inside the min-duration gate, so it is
        # not an eligible plateau and must not trap the run in "Not settled".
        records = _flat(100.0, 0, 59) + _flat(97.0, 60, 5400)
        outcomes = self._track_best(th, records, range(0, 5401, 10))
        settled_at = [now for now, settled, _ in outcomes if settled]
        assert settled_at, "a flat 97 after min_duration must settle"
        assert settled_at[0] == 300
        _, _, info = outcomes[30]           # now = 300
        assert info["best"] == pytest.approx(97.0)

    def test_throttled_high_window_does_not_raise_best(self, th):
        # Thermal throttling in a window disqualifies it as a plateau, so a
        # spuriously high throttled window must not raise the bar either.
        records = _flat(200.0, 0, 399) + _flat(300.0, 400, 460, throttled=True) + _flat(200.0, 461, 600)
        settled, info = _converged(th, records, 460, best_window_mean=200.0)
        assert settled is False
        assert info["best"] == pytest.approx(200.0)
        settled, info = _converged(th, records, 600, best_window_mean=info["best"])
        assert settled is True

    def test_eligible_high_plateau_still_blocks_a_later_low_one(self, th):
        # Flat 200 past min_duration (eligible, sets best=200), then drops to 150:
        # "stable but low" must still be rejected.
        records = _flat(200.0, 0, 399) + _flat(150.0, 400, 600)
        outcomes = self._track_best(th, records, range(0, 601, 10))
        now, settled, info = outcomes[-1]
        assert now == 600 and settled is False
        assert info["best"] == pytest.approx(200.0)
        assert "best" in info["reason"]

    def test_target_gflops_floor(self, th):
        records = _flat(200.0, 0, 600)
        settled, info = _converged(th, records, 600, target_gflops=250.0)
        assert settled is False
        assert "target" in info["reason"]
        settled, _ = _converged(th, records, 600, target_gflops=150.0)
        assert settled is True

    def test_min_duration_gate(self, th):
        records = _flat(200.0, 0, 299)
        settled, info = _converged(th, records, 299)
        assert settled is False
        assert "min" in info["reason"].lower()
        # window stats are still reported so progress lines can show them
        assert info["window_mean"] == pytest.approx(200.0)
        settled, _ = _converged(th, _flat(200.0, 0, 300), 300)
        assert settled is True

    def test_records_after_now_are_ignored(self, th):
        # The caller only ever has records up to "now"; a full trajectory with an
        # earlier now_s must behave as if the later records did not exist yet.
        records = _flat(200.0, 0, 399) + _flat(150.0, 400, 600)
        _, info = _converged(th, records, 399)
        assert info["window_mean"] == pytest.approx(200.0)
        assert info["window_cv"] == pytest.approx(0.0)

    def test_fewer_than_two_points_in_window_is_not_settled(self, th):
        records = _flat(200.0, 0, 500) + [(600.0, 200.0, False)]
        settled, info = _converged(th, records, 600)
        assert settled is False
        assert info["window_mean"] is None
        assert "fewer" in info["reason"]

    def test_empty_records_is_not_settled(self, th):
        settled, info = _converged(th, [], 600)
        assert settled is False

    def test_accepts_longer_cycle_records(self, th):
        """Caller may pass MeltCycle (8 fields); only the first four are read."""
        records = [th.MeltCycle(float(t), 200.0, False, power_min_W=400.0, power_max_W=650.0)
                   for t in range(0, 601)]
        settled, _ = _converged(th, records, 600)
        assert settled is True


# ──────────────────────────────────────────────────────────────
# 2. per-cycle telemetry classification
# ──────────────────────────────────────────────────────────────
class TestCycleTelemetry:

    def test_hw_or_sw_slowdown_is_thermal(self, th):
        assert th._sample_thermal_throttled({"hw_slowdown": 1, "sw_slowdown": 0}) is True
        assert th._sample_thermal_throttled({"hw_slowdown": 0, "sw_slowdown": 1}) is True
        assert th._sample_thermal_throttled({"thermal_slowdown": 1}) is True   # AMD

    def test_power_brake_alone_is_not_thermal(self, th):
        sample = {"hw_slowdown": 0, "sw_slowdown": 0, "power_limit": 1, "throttled": 1}
        assert th._sample_thermal_throttled(sample) is False

    def test_cpu_sample_without_throttle_keys_is_not_thermal(self, th):
        assert th._sample_thermal_throttled({"vendor": "CPU", "model": "arm64"}) is False

    def test_cycle_summary_aggregates_power_and_throttle(self, th):
        thread = types.SimpleNamespace()
        import threading
        thread.lock = threading.Lock()
        thread.iteration_samples = {
            7: [{"power_W": 650.0, "hw_slowdown": 0},
                {"power_W": 400.0, "hw_slowdown": 1},
                {"power_W": "N/A", "hw_slowdown": 0}],
        }
        summary = th._melt_cycle_telemetry(thread, 7)
        assert summary.thermal_throttled is True
        assert summary.power_min_W == 400.0
        assert summary.power_max_W == 650.0

    def test_cycle_summary_without_samples(self, th):
        import threading
        thread = types.SimpleNamespace(lock=threading.Lock(), iteration_samples={})
        assert th._melt_cycle_telemetry(thread, 0) == (False, None, None, None, None, None)


# ──────────────────────────────────────────────────────────────
# 2b. Host-reload cool gap: footprint and release
# ──────────────────────────────────────────────────────────────
class TestHostReloadCoolGap:

    @pytest.fixture
    def operands(self):
        import torch
        return torch.zeros(2, 8, 4), torch.zeros(2, 4, 16)

    def test_one_host_copy_per_operand(self, th, operands):
        A, Bm = operands
        gap = th._HostReloadCoolGap(A, Bm, pin_memory=False)
        assert gap.host_bytes == (A.numel() + Bm.numel()) * A.element_size()

    def test_host_footprint_formula(self, th):
        assert th._melt_host_footprint_bytes(3, 8, 16, 4, 8) == 3 * (8 * 4 + 4 * 16) * 8

    def test_reload_alternates_operands(self, th, operands):
        A, Bm = operands
        gap = th._HostReloadCoolGap(A, Bm, pin_memory=False)
        gap.reload(0)
        assert A.abs().sum() > 0 and Bm.abs().sum() == 0
        gap.reload(1)
        assert Bm.abs().sum() > 0

    def test_release_drops_host_buffers(self, th, operands):
        A, Bm = operands
        gap = th._HostReloadCoolGap(A, Bm, pin_memory=False)
        gap.release()
        assert gap.host_bytes == 0

    def test_in_place_gap_release_is_harmless(self, th, operands):
        th._InPlaceCoolGap(*operands).release()


class TestMeltDryRunLine:

    def _mb(self, th, parser, precision):
        args = parser.parse_args(["--melt", "--precision-melt", precision,
                                  "--melt-batch-count", "4", "--melt-m", "256", "--melt-n", "256", "--melt-k", "256"])
        line = th._melt_dry_run_line(args)
        assert "Melt" in line and precision in line
        return float(line.split("~")[1].split("MB")[0])

    def test_dry_run_uses_dtype_element_size(self, th, parser):
        assert self._mb(th, parser, "float64") == pytest.approx(2 * self._mb(th, parser, "float32"))
        assert self._mb(th, parser, "float16") == pytest.approx(0.5 * self._mb(th, parser, "float32"))
        assert self._mb(th, parser, "float32") == pytest.approx(4 * (256 * 256 * 2) * 4 / 1024**2)


# ──────────────────────────────────────────────────────────────
# 3. --stress-test sizing
# ──────────────────────────────────────────────────────────────
class TestMeltStressSizing:

    def test_melt_key_sizes_like_gemm_under_melt_names(self, th):
        log = logging.getLogger("test_melt_stress")
        gemm = th.calculate_stress_params("gemm", "float32", 8000, log)
        melt = th.calculate_stress_params("melt", "float32", 8000, log)
        assert set(melt) == {"melt_m", "melt_n", "melt_k", "melt_batch_count"}
        assert melt["melt_m"] == gemm["m"]
        assert melt["melt_k"] == gemm["k"]
        assert melt["melt_batch_count"] == gemm["batch_count_gemm"]

    def test_melt_sizing_is_capped_independently_of_available_memory(self, th):
        log = logging.getLogger("test_melt_stress")
        available_mb = 192 * 1024            # MI300X-class
        melt = th.calculate_stress_params("melt", "float32", available_mb, log)
        gemm = th.calculate_stress_params("gemm", "float32", available_mb, log)
        b, m, n, k = melt["melt_batch_count"], melt["melt_m"], melt["melt_n"], melt["melt_k"]
        operand_bytes = b * (m * k + k * n) * 4
        assert operand_bytes <= th.MELT_MAX_WORKING_SET_MB * 1024**2
        assert m < gemm["m"]                  # GEMM still uses the memory; melt does not

    def test_melt_sizing_matches_gemm_when_under_the_cap(self, th):
        log = logging.getLogger("test_melt_stress")
        small_mb = th.MELT_MAX_WORKING_SET_MB // 2
        gemm = th.calculate_stress_params("gemm", "float32", small_mb, log)
        melt = th.calculate_stress_params("melt", "float32", small_mb, log)
        assert melt["melt_m"] == gemm["m"]

    def test_apply_stress_params_writes_melt_attrs(self, th, parser):
        args = parser.parse_args(["--melt", "--precision-melt", "float64"])
        th.apply_stress_params(args, "melt", 8000, logging.getLogger("t"))
        assert args.melt_m == args.melt_n == args.melt_k
        assert args.melt_m >= 512
        assert args.melt_batch_count > 0
        assert args.m == 512  # GEMM sizes untouched


# ──────────────────────────────────────────────────────────────
# 4. melt-first ordering of the config list
# ──────────────────────────────────────────────────────────────
class TestMeltFirstOrdering:

    def test_melt_entry_is_hoisted_to_front(self, th):
        entries = [{"name": "batched_gemm"}, {"name": "fft"}, {"name": "melt", "m": 64}]
        ordered, hoisted = th._melt_first(entries, melt_flag=False)
        assert [e["name"] for e in ordered] == ["melt", "batched_gemm", "fft"]
        assert ordered[0]["m"] == 64
        assert hoisted is True

    def test_already_first_is_not_reported_as_hoisted(self, th):
        entries = [{"name": "melt"}, {"name": "fft"}]
        ordered, hoisted = th._melt_first(entries, melt_flag=False)
        assert [e["name"] for e in ordered] == ["melt", "fft"]
        assert hoisted is False

    def test_global_melt_flag_injects_an_entry(self, th):
        entries = [{"name": "batched_gemm"}]
        ordered, hoisted = th._melt_first(entries, melt_flag=True)
        assert [e["name"] for e in ordered] == ["melt", "batched_gemm"]
        assert hoisted is True

    def test_no_melt_leaves_list_untouched(self, th):
        entries = [{"name": "batched_gemm"}, {"name": "fft"}]
        ordered, hoisted = th._melt_first(entries, melt_flag=False)
        assert ordered == entries
        assert hoisted is False


# ──────────────────────────────────────────────────────────────
# 5. result plumbing: status, syslog, compact, JSON
# ──────────────────────────────────────────────────────────────
def _melt_perf(settled=True, settle_s=4704.0, throttled=False):
    perf = {
        "name": "Melt", "unit": "GFLOP/s",
        "min": 23900.0, "mean": 48000.0, "max": 54120.0,
        "iterations": 33612, "runtime_s": settle_s or 5400.0,
        "params": {"dtype": "float32", "m": 8192},
        "telemetry": {"throttled": True} if throttled else {},
        "settled": settled, "settle_s": settle_s, "cycles": 33612,
        "start_gflops": 23900.0, "plateau_gflops": 54120.0,
        "thermal_throttle_cycles": 140, "last_throttle_s": 4100.0,
        "ops_per_burst": 20, "burst_ms_measured": 141.2,
        "reload_path": "h2d_pinned", "reload_ms_measured": 3.1, "duty_cycle_pct": 58.5,
    }
    if throttled:
        perf["throttled"] = True
    return perf


class TestMeltSummaryStatus:

    def test_settled_melt_passes_even_if_throttled(self, th):
        status, notes = th._summary_status(_melt_perf(settled=True, throttled=True),
                                           {"throttled": True}, 70.0)
        assert status == "PASS"
        assert notes == "Settled in 78.4 min (23,900 → 54,120)"

    def test_unsettled_melt_fails_with_not_settled(self, th):
        status, notes = th._summary_status(_melt_perf(settled=False, settle_s=None, throttled=True),
                                           {"throttled": True}, 70.0)
        assert (status, notes) == ("FAIL", "Not settled (23,900 → 54,120)")

    def test_non_melt_behaviour_unchanged(self, th):
        gemm = {"name": "Batched GEMM", "telemetry": {"throttled": True}}
        assert th._summary_status(gemm, {}, 70.0) == ("FAIL", "Throttled")
        clean = {"name": "Batched GEMM", "telemetry": {}}
        assert th._summary_status(clean, {}, 70.0) == ("PASS", "")

    def test_summary_rows_carry_melt_status(self, th):
        result = {"gpu_index": 0, "serial": "S1", "telemetry_stats": {},
                  "benchmarks": [_melt_perf(settled=False, settle_s=None)]}
        rows = th._build_summary_rows([result], 70.0)
        assert rows[0]["test"] == "Melt"
        assert rows[0]["status"] == "FAIL"
        assert rows[0]["notes"] == "Not settled (23,900 → 54,120)"

    def test_summary_value_is_the_plateau_not_the_run_mean(self, th):
        # The whole-run mean is dragged down by the throttled start; the
        # number an operator wants in the table is where the GPU ended up.
        result = {"gpu_index": 0, "serial": "S1", "telemetry_stats": {},
                  "benchmarks": [_melt_perf(settled=True)]}
        rows = th._build_summary_rows([result], 70.0)
        assert rows[0]["performance"] == 54120.0

    def test_summary_value_for_non_melt_is_still_the_mean(self, th):
        gemm = {"name": "Batched GEMM", "unit": "GFLOP/s", "mean": 100.0, "min": 90.0, "max": 110.0,
                "params": {"dtype": "float32"}, "telemetry": {}}
        result = {"gpu_index": 0, "serial": "S1", "telemetry_stats": {}, "benchmarks": [gemm]}
        assert th._build_summary_rows([result], 70.0)[0]["performance"] == 100.0


class TestMeltJsonExport:
    tel_data = {"hostname": "node01", "model": "GH200", "serial": "ABC"}

    def test_json_export_carries_melt_fields(self, th, tmp_path, monkeypatch):
        import torch
        monkeypatch.setattr(th, "_unique_output_path", lambda p: tmp_path / "melt.json")
        args = types.SimpleNamespace(json_output=str(tmp_path / "melt.json"), config=None,
                                     duration=None, min_iterations=10, max_iterations=None,
                                     warmup=1, verbose=False)
        perf = _melt_perf()
        perf.update(temp_slope_c_per_min=-0.12, window_temp_mean_C=46.1, temp_mean_C=46.3, temp_max_C=53.0,
                    temp_sibling_delta_C=None)
        perf["iteration_telemetry"] = [{"iteration": 0, "performance": 23900.0,
                                        "power_min_W": 400.0, "power_max_W": 650.0, "temp_mean_C": 61.2,
                                        "thermal_throttled": True, "telemetry": {}}]
        results = [{"gpu_index": 0, "model": "GH200", "serial": "ABC",
                    "final_telemetry": {}, "telemetry_stats": {},
                    "benchmarks": [perf]}]
        th.export_json_results(results, args, logging.getLogger("t"))
        data = json.loads((tmp_path / "melt.json").read_text())
        bench = data["gpus"][0]["benchmarks"][0]
        assert bench["name"] == "Melt"
        assert bench["settled"] is True
        assert bench["settle_s"] == 4704.0
        assert bench["reload_path"] == "h2d_pinned"
        assert bench["iteration_telemetry"][0]["power_min_W"] == 400.0
        assert bench["iteration_telemetry"][0]["temp_mean_C"] == 61.2
        assert (bench["temp_slope_c_per_min"], bench["window_temp_mean_C"]) == (-0.12, 46.1)
        assert (bench["temp_mean_C"], bench["temp_max_C"], bench["temp_sibling_delta_C"]) == (46.3, 53.0, None)


# ──────────────────────────────────────────────────────────────
# 6. TelemetryThread isolation and _log_summary timestamp index
# ──────────────────────────────────────────────────────────────
class TestTelemetryIsolation:

    def test_clear_iteration_samples(self, th):
        import torch
        tel = th.CpuTelemetry(0)
        thread = th.TelemetryThread(tel, torch.device("cpu"))
        thread.mark_iteration_start(0)
        thread.iteration_samples[0].append({"hw_slowdown": 1})
        thread.mark_iteration_end(0)
        thread.clear_iteration_samples()
        assert thread.iteration_samples == {}

    def test_log_summary_resolves_iteration_timestamps_by_identity(self, th):
        """Timestamps come from an identity index, not an O(N^2) equality scan."""
        import torch
        tel = MagicMock()
        tel.read.return_value = {"device_id": 0, "model": "TestGPU"}
        tel.get_stats.return_value = {}
        tel.throttle_detected = False
        thread = th.TelemetryThread(th.CpuTelemetry(0), torch.device("cpu"))
        first = {"device_id": 0, "power_W": 1.0}
        twin = dict(first)   # equal but not identical: must not be confused
        thread.all_samples = [(10.0, twin), (11.0, first)]
        thread.iteration_samples = {0: [first]}
        logger = logging.getLogger("test_ts")
        logger.setLevel(logging.WARNING)
        result = th._log_summary("TestBench", [1.0], "GFLOP/s", logger, tel,
                                 torch.device("cpu"), tel_thread=thread)
        assert result["iteration_telemetry"][0]["timestamp"] == 11.0


# ──────────────────────────────────────────────────────────────
# 7. Cool-gap floor, duty cycle, calibration warning (review bundle B/C)
# ──────────────────────────────────────────────────────────────
class TestCoolGapFloor:

    def test_sleep_tops_reload_up_to_the_floor(self, th):
        assert th._cool_sleep_s(cool_ms=100, reload_elapsed_s=0.003) == pytest.approx(0.097)

    def test_no_sleep_when_reload_already_exceeds_the_floor(self, th):
        assert th._cool_sleep_s(cool_ms=100, reload_elapsed_s=0.150) == 0.0

    def test_zero_floor_means_no_sleep(self, th):
        assert th._cool_sleep_s(cool_ms=0, reload_elapsed_s=0.003) == 0.0

    def test_duty_cycle_uses_the_floor_when_reload_is_short(self, th):
        assert th._duty_cycle_pct(burst_ms=140, cool_ms=100, reload_ms=3.1) == pytest.approx(140 / 240 * 100)

    def test_duty_cycle_uses_the_reload_when_it_is_longer_than_the_floor(self, th):
        assert th._duty_cycle_pct(burst_ms=140, cool_ms=100, reload_ms=150) == pytest.approx(140 / 290 * 100)

    def test_cool_gap_line(self, th):
        line = th._cool_gap_line("[GPU0 Melt]", "h2d_pinned", reload_ms=3.1, cool_ms=100, burst_ms=140)
        assert line == "[GPU0 Melt] Cool gap: h2d_pinned reload 3.1 ms + sleep to 100 ms floor (duty cycle 58% hot)"

    def test_cool_gap_line_without_sleep(self, th):
        line = th._cool_gap_line("[GPU0 Melt]", "uniform_", reload_ms=150.0, cool_ms=100, burst_ms=140)
        assert "no sleep" in line and "48% hot" in line

    def test_result_fields_include_reload_and_duty_cycle(self, th):
        assert "reload_ms_measured" in th.MELT_RESULT_FIELDS
        assert "duty_cycle_pct" in th.MELT_RESULT_FIELDS

    def test_result_fields_include_both_cvs(self, th):
        assert "plateau_cv_pct" in th.MELT_RESULT_FIELDS
        assert "cycle_cv_pct" in th.MELT_RESULT_FIELDS

    def test_result_fields_include_the_failing_rule_names(self, th):
        # Without this the JSON cannot distinguish "CV only" (noise) from "CV + below best" (module).
        assert "stop_rules_failed" in th.MELT_RESULT_FIELDS


class TestBurstSizeWarning:

    def test_fast_matmul_is_too_small_to_heat(self, th):
        msg = th._burst_too_small_warning(ops_per_burst=50, ms_per_matmul=0.4)
        assert msg is not None
        assert "50 matmuls of 0.40 ms" in msg
        assert "increase --melt-m/n/k or --melt-batch-count" in msg

    def test_too_many_ops_per_burst_is_too_small_to_heat(self, th):
        assert th._burst_too_small_warning(ops_per_burst=201, ms_per_matmul=1.5) is not None

    def test_healthy_burst_has_no_warning(self, th):
        assert th._burst_too_small_warning(ops_per_burst=5, ms_per_matmul=27.7) is None


# ──────────────────────────────────────────────────────────────
# 8. Telemetry window spans burst + cool (review bundle A)
# ──────────────────────────────────────────────────────────────
class _OrderRecordingThread:
    """Minimal TelemetryThread stand-in that records the order of melt calls."""

    def __init__(self, calls):
        import threading
        self.calls = calls
        self.lock = threading.Lock()
        self.iteration_samples = {}

    def mark_iteration_start(self, i):
        self.calls.append("mark_start")

    def mark_iteration_end(self, i):
        self.calls.append("mark_end")

    def get_iteration_telemetry(self, i):
        return {}


class _OrderRecordingGap:
    path = "uniform_"

    def __init__(self, calls):
        self.calls = calls

    def reload(self, cycle):
        self.calls.append("reload")

    def release(self):
        pass


class TestCycleTelemetryWindow:

    def test_iteration_end_is_marked_after_the_cool_gap(self, th, parser, monkeypatch):
        import torch
        calls = []
        args = parser.parse_args(["--melt", "--melt-max-duration", "0", "--melt-cool-ms", "5"])
        workload = th._MeltWorkload(torch.ones(1, 4, 4), torch.ones(1, 4, 4),
                                    _OrderRecordingGap(calls), 1, 0.01, "float32")
        monkeypatch.setattr(th.time, "sleep", lambda s: calls.append("sleep"))
        run = th._run_melt_cycles(args, torch.device("cpu"), logging.getLogger("t"),
                                  _OrderRecordingThread(calls), None, workload, "[CPU Melt]")
        assert len(run.records) == 1
        assert calls == ["mark_start", "reload", "sleep", "mark_end"]


# ──────────────────────────────────────────────────────────────
# 9. Verdict and progress legibility (review bundle E/F)
# ──────────────────────────────────────────────────────────────
def _outcome(settled, **over):
    base = dict(settled=settled, settle_s=4704.0 if settled else None, cycles=25910,
                start_gflops=23900.0, plateau_gflops=54120.0, thermal_throttle_cycles=140,
                last_throttle_s=4100.0, power_min_W=180.0, power_max_W=655.0, stop_reason="settled",
                stop_rules_failed=[], plateau_cv_pct=0.3, cycle_cv_pct=1.8,
                start_temp_C=67.7, end_temp_C=58.9, power_mean_W=423.0, start_throttle_cycles=18, end_throttle_cycles=0)
    base.update(over)
    return base


class TestMeltVerdictLines:

    def test_settled_is_headline_plus_before_after(self, th):
        lines = th._melt_verdict_lines("[GPU0 Melt]", _outcome(True), runtime_s=4704.0)
        assert lines == [
            (logging.INFO, "[GPU0 Melt] SETTLED after 78.4 min (cycles 25,910): 23,900 → 54,120 GFLOP/s, power 180–655 W, temp n/a"),
            (logging.INFO, "  start → end: temp 67.7 → 58.9 °C (−8.8 at 423 W) | 23,900 → 54,120 GFLOP/s (+126.4%) | "
                           "thermal throttle 18 → 0 cycles"),
        ]

    def test_not_settled_explains_the_blocking_rule_and_next_step(self, th):
        outcome = _outcome(False, stop_reason="plateau CV 3.20% > 1%", plateau_gflops=51200.0,
                           stop_rules_failed=["plateau_cv", "below_best"])
        lines = th._melt_verdict_lines("[GPU0 Melt]", outcome, runtime_s=5400.0)
        assert lines[0] == (logging.WARNING,
                            "[GPU0 Melt] NOT SETTLED after 90.0 min (cycles 25,910): 23,900 → 51,200 GFLOP/s, power 180–655 W, temp n/a")
        assert lines[1][0] == logging.INFO and lines[1][1].startswith("  start → end: temp 67.7 → 58.9 °C")
        assert lines[2] == (logging.WARNING,
                            "  blocking rule at stop: plateau CV 3.20% > 1%; last thermal throttle at 68.3 min; 140 throttled cycles total")
        assert lines[3] == (logging.WARNING,
                            "  next: rerun --melt once; a second NOT SETTLED points at the module (TIM / cold plate / coolant), not the tool")

    def test_not_settled_without_any_throttle(self, th):
        outcome = _outcome(False, stop_reason="window mean is 95.0% of best (56,000), need 98%",
                           stop_rules_failed=["below_best"], thermal_throttle_cycles=0, last_throttle_s=None)
        lines = th._melt_verdict_lines("[GPU0 Melt]", outcome, runtime_s=5400.0)
        assert "no thermal throttle seen; 0 throttled cycles total" in lines[2][1]
        assert "points at the module" in lines[3][1]

    def test_cv_only_blocker_is_called_measurement_noise_not_tim(self, th):
        # Flat, throttle-free, at best — only the CV rule failed: do not blame the module.
        outcome = _outcome(False, stop_reason="plateau CV 1.70% > 1%", stop_rules_failed=["plateau_cv"],
                           plateau_cv_pct=1.7, thermal_throttle_cycles=0, last_throttle_s=None)
        lines = th._melt_verdict_lines("[GPU0 Melt]", outcome, runtime_s=5400.0)
        assert len(lines) == 3
        assert lines[2] == (logging.WARNING,
                            "  blocking rule at stop: plateau CV 1.70% > 1% — performance is flat and throttle-free; "
                            "this is most likely measurement noise, not TIM. "
                            "Consider --melt-plateau-cv-pct 2 or a larger --melt-bin-s")

    def test_cv_blocker_with_throttling_keeps_the_module_advice(self, th):
        outcome = _outcome(False, stop_reason="plateau CV 1.70% > 1%", stop_rules_failed=["plateau_cv"],
                           thermal_throttle_cycles=140, last_throttle_s=5390.0)
        lines = th._melt_verdict_lines("[GPU0 Melt]", outcome, runtime_s=5400.0)
        assert len(lines) == 4
        assert "points at the module" in lines[3][1]

    def test_power_unknown_is_shown_as_na(self, th):
        lines = th._melt_verdict_lines("[CPU Melt]", _outcome(True, power_min_W=None, power_max_W=None), runtime_s=4704.0)
        assert lines[0][1].endswith("power n/a, temp n/a")


class TestMeltProgressLine:

    def _info(self, **over):
        info = dict(window_mean=48912.0, window_cv=0.3, cycle_cv_pct=1.8, best=51004.0, throttle_count=14,
                    reason="thermal throttling in window (14 cycles)")
        info.update(over)
        return info

    def test_progress_line_format(self, th):
        line = th._melt_progress_line("[GPU0 Melt]", elapsed_s=738.0, max_s=5400.0, cycle_count=5280,
                                      info=self._info(), power_range=(180.0, 655.0))
        assert line == ("[GPU0 Melt] 12.3/90.0 min | 5,280 cycles | window 48,912 GFLOP/s, CV 0.3% (cycle 1.8%) | "
                        "best 51,004 | power 180–655 W | temp n/a | not settled: thermal throttling in window (14 cycles)")

    def test_progress_line_before_any_eligible_window(self, th):
        line = th._melt_progress_line("[GPU0 Melt]", elapsed_s=60.0, max_s=5400.0, cycle_count=430,
                                      info=self._info(best=None), power_range=None)
        assert "| best n/a |" in line
        assert "| power n/a |" in line

    def test_progress_line_without_window_stats(self, th):
        line = th._melt_progress_line("[GPU0 Melt]", elapsed_s=1.0, max_s=5400.0, cycle_count=1,
                                      info=dict(window_mean=None, window_cv=None, cycle_cv_pct=None, best=None,
                                                throttle_count=0, reason="fewer than 2 cycles in the last 60 s"),
                                      power_range=None)
        assert "| window n/a |" in line


class TestMeltClosingLine:

    def test_ok_when_no_melt_or_melt_settled(self, th):
        assert th._run_closing_line("GPU0", []) == "[OK] Benchmark run finished on GPU0"
        assert th._run_closing_line("GPU0", [_melt_perf(settled=True)]) == "[OK] Benchmark run finished on GPU0"

    def test_fail_when_melt_not_settled(self, th):
        perf = _melt_perf(settled=False, settle_s=None)
        perf["runtime_s"] = 5400.0
        assert th._run_closing_line("GPU0", [perf]) == \
            "[FAIL] Melt: GPU0 NOT SETTLED after 90.0 min — see 'NOT SETTLED' line above"

    def test_multi_gpu_lists_unsettled_gpus(self, th):
        ok = {"gpu_index": 0, "benchmarks": [_melt_perf(settled=True)]}
        bad2 = {"gpu_index": 2, "benchmarks": [_melt_perf(settled=False, settle_s=None)]}
        bad5 = {"gpu_index": 5, "benchmarks": [_melt_perf(settled=False, settle_s=None)]}
        assert th._multi_gpu_melt_closing_line([ok]) is None
        assert th._multi_gpu_melt_closing_line([ok, bad2, bad5]) == \
            "[FAIL] Melt: NOT SETTLED on GPU 2, GPU 5 — see 'NOT SETTLED' lines above"


class TestDryRunMeltOnly:

    def test_non_melt_lines_are_marked_skipped(self, th):
        lines = ["  - Melt ...", "  - Batched GEMM ...", "  - Sparse MM ..."]
        assert th._annotate_dry_run_skips(lines, melt_only=True) == [
            "  - Melt ...", "  - Batched GEMM ... (skipped: --melt-only)", "  - Sparse MM ... (skipped: --melt-only)"]
        assert th._annotate_dry_run_skips(lines, melt_only=False) == lines

    def test_melt_dry_run_line_shows_minutes(self, th, parser):
        line = th._melt_dry_run_line(parser.parse_args(["--melt"]))
        assert "min 3600 s (60 min)" in line and "max 3600 s (60 min)" in line
        assert "cool 100ms" in line and "bin 10s" in line


class TestWorkloadRelease:

    def test_release_drops_tensors_before_empty_cache(self, th, monkeypatch):
        import torch
        A, Bm = torch.ones(1, 4, 4), torch.ones(1, 4, 4)
        workload = th._MeltWorkload(A, Bm, th._InPlaceCoolGap(A, Bm), 1, 0.01, "float32")
        seen = {}
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.update(A=workload.A, Bm=workload.Bm))
        th._release_melt_workload(workload, types.SimpleNamespace(type="cuda"))
        assert seen == {"A": None, "Bm": None}
        assert workload.cool_gap.A is None


# ──────────────────────────────────────────────────────────────
# 10. Replay of the 2026-10-06 GH200 control-node run (regression guard)
# ──────────────────────────────────────────────────────────────
class TestGH200ControlNodeReplay:
    """Four known-settled GH200s, float64 8×4096³, 90 min: flat 53.4k GFLOP/s,
    1.9 % per-cycle DVFS jitter, zero throttle — and they were reported
    NOT SETTLED because the per-cycle CV was 1.7–2.2 %.  Must settle at the
    min-duration gate with default knobs."""

    def test_replays_settle_at_min_duration_with_defaults(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-min-duration", "300"])
        records = _noisy(53400.0, 0.235, 5400.0, step=0.235, cv_pct=1.9, seed=20261006)
        tracker = th._ConvergenceTracker.from_args(args)
        settled_at = None
        for i, (t, _, _) in enumerate(records):
            settled, info = tracker.update(records[:i + 1], t)
            if settled:
                settled_at = t
                break
        assert settled_at is not None, f"never settled: {info['reason']}"
        assert settled_at == pytest.approx(args.melt_min_duration, abs=1.0)
        assert info["window_cv"] < args.melt_plateau_cv_pct
        assert info["cycle_cv_pct"] > 1.0

    def test_tracker_carries_best_between_calls(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-min-duration", "300"])
        tracker = th._ConvergenceTracker.from_args(args)
        tracker.update(_flat(200.0, 0, 400), 400)
        assert tracker.best_window_mean == pytest.approx(200.0)
        settled, info = tracker.update(_flat(150.0, 0, 600), 600)
        assert settled is False and "best" in info["reason"]


# ──────────────────────────────────────────────────────────────
# 11. Temperature plateau rule (field run 2: GPU1 settled on GFLOP/s while still cooling)
# ──────────────────────────────────────────────────────────────
def _with_temp(records, temp_of):
    """Attach a temperature (°C) to 3-tuple records: ``temp_of(t_s) -> float | None``."""
    return [(t, g, thr, temp_of(t)) for t, g, thr, *_ in records]


def _trending(records, start_c, c_per_min):
    return _with_temp(records, lambda t: start_c + c_per_min * t / 60.0)


class TestTempSlopeRule:

    def test_falling_temperature_blocks_an_otherwise_flat_plateau(self, th):
        records = _trending(_flat(200.0, 0, 600, step=0.5), 70.0, -1.1)
        settled, info = _converged(th, records, 600)
        assert settled is False
        assert info["rules_failed"] == ["temp_slope"]
        assert info["reason"] == "temperature still trending -1.1 °C/min (limit 0.5)"
        assert info["temp_slope_c_per_min"] == pytest.approx(-1.1, abs=0.01)

    def test_rising_temperature_blocks_too(self, th):
        records = _trending(_flat(200.0, 0, 600, step=0.5), 40.0, +0.8)
        settled, info = _converged(th, records, 600)
        assert settled is False and info["rules_failed"] == ["temp_slope"]
        assert "+0.8 °C/min" in info["reason"]

    def test_flat_temperature_settles_and_reports_the_window_mean(self, th):
        records = _with_temp(_flat(200.0, 0, 600, step=0.5), lambda t: 46.1)
        settled, info = _converged(th, records, 600)
        assert settled is True
        assert info["temp_slope_c_per_min"] == pytest.approx(0.0, abs=1e-9)
        assert info["window_temp_mean_C"] == pytest.approx(46.1)

    def test_window_temp_mean_is_over_the_performance_window(self, th):
        records = _with_temp(_flat(200.0, 0, 600, step=1.0), lambda t: 80.0 if t < 540 else 50.0)
        _, info = _converged(th, records, 600)
        assert info["window_temp_mean_C"] == pytest.approx(50.0)

    def test_temperature_absent_skips_the_rule(self, th):
        settled, info = _converged(th, _flat(200.0, 0, 600), 600)
        assert settled is True
        assert info["temp_slope_c_per_min"] is None
        assert info["window_temp_mean_C"] is None

    def test_zero_limit_disables_the_rule(self, th):
        records = _trending(_flat(200.0, 0, 600, step=0.5), 70.0, -2.0)
        settled, info = _converged(th, records, 600, temp_slope_c_per_min=0)
        assert settled is True
        assert info["temp_slope_c_per_min"] is None
        assert info["window_temp_mean_C"] is not None

    def test_fewer_than_three_temperature_bins_skips_the_rule(self, th):
        # Temperature only in the newest 15 s of the 60 s window -> 2 bins with data.
        records = _with_temp(_flat(200.0, 0, 600, step=0.5),
                             lambda t: (70.0 - 2.0 * t / 60.0) if t > 585 else None)
        settled, info = _converged(th, records, 600)
        assert settled is True
        assert info["temp_slope_c_per_min"] is None
        assert info["window_temp_mean_C"] is not None

    def test_slope_inside_the_limit_passes(self, th):
        records = _trending(_flat(200.0, 0, 600, step=0.5), 70.0, -0.45)
        settled, info = _converged(th, records, 600)
        assert settled is True
        assert info["temp_slope_c_per_min"] == pytest.approx(-0.45, abs=0.01)

    def test_temp_slope_is_a_rule_in_evaluation_order_after_the_performance_rules(self, th):
        records = _trending(_noisy(200.0, 0, 600, step=0.5, cv_pct=5.0), 70.0, -2.0)
        _, info = _converged(th, records, 600)
        assert info["rules_failed"] == ["plateau_cv", "temp_slope"]

    def test_temperature_horizon_is_longer_than_the_performance_window(self, th):
        # Flat for the last 60 s but 4 °C lower than 5 min ago: invisible to a
        # 60 s slope, caught over the default 300 s horizon.
        records = _with_temp(_flat(200.0, 0, 600, step=0.5),
                             lambda t: 59.3 if t >= 540 else 63.6 - (t - 300) * 4.3 / 240)
        settled_short, info_short = _converged(th, records, 600, temp_window_s=60.0)
        settled_long, info_long = _converged(th, records, 600)
        assert settled_short is True
        assert settled_long is False and info_long["rules_failed"] == ["temp_slope"]
        assert info_long["temp_slope_c_per_min"] < -0.5
        assert info_long["window_temp_mean_C"] == pytest.approx(59.3)   # still the 60 s window

    def test_melt_cycle_records_carry_temperature_in_the_fourth_slot(self, th):
        records = [th.MeltCycle(float(t), 200.0, False, temp_mean_C=70.0 - t / 60.0, power_min_W=400.0)
                   for t in range(0, 601)]
        settled, info = _converged(th, records, 600)
        assert settled is False and info["rules_failed"] == ["temp_slope"]
        assert th.MeltCycle(1.0, 2.0, False).temp_mean_C is None

    def test_least_squares_slope(self, th):
        assert th._least_squares_slope([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == pytest.approx(2.0)
        assert th._least_squares_slope([0.0, 10.0, 20.0, 30.0], [5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)


class TestCycleTemperatureTelemetry:

    def test_cycle_summary_reports_mean_and_max_temperature(self, th):
        import threading
        thread = types.SimpleNamespace(lock=threading.Lock(), iteration_samples={
            3: [{"temp_gpu_C": 60.0, "power_W": 400.0}, {"temp_gpu_C": 64.0, "power_W": 420.0},
                {"temp_gpu_C": "N/A", "power_W": 410.0}]})
        summary = th._melt_cycle_telemetry(thread, 3)
        assert summary.temp_mean_C == pytest.approx(62.0)
        assert summary.temp_max_C == 64.0
        assert (summary.power_min_W, summary.power_max_W) == (400.0, 420.0)

    def test_cycle_summary_without_temperature_key(self, th):
        import threading
        thread = types.SimpleNamespace(lock=threading.Lock(),
                                       iteration_samples={0: [{"vendor": "CPU", "power_W": 50.0}]})
        summary = th._melt_cycle_telemetry(thread, 0)
        assert summary.temp_mean_C is None and summary.temp_max_C is None


class TestTempInOutcomeAndDisplay:

    def test_progress_line_shows_temperature_and_slope(self, th):
        info = dict(window_mean=48912.0, window_cv=0.3, cycle_cv_pct=1.8, best=51004.0, throttle_count=0,
                    reason="temperature still trending -1.1 °C/min (limit 0.5)",
                    temp_slope_c_per_min=-1.12, window_temp_mean_C=61.4)
        line = th._melt_progress_line("[GPU1 Melt]", 300.0, 5400.0, 1277, info, (321.0, 484.0))
        assert "| power 321–484 W | temp 61.4 °C (−1.1 °C/min) | not settled:" in line

    def test_progress_line_temperature_without_slope(self, th):
        info = dict(window_mean=48912.0, window_cv=0.3, cycle_cv_pct=1.8, best=51004.0, throttle_count=0,
                    reason="x", temp_slope_c_per_min=None, window_temp_mean_C=46.1)
        assert "| temp 46.1 °C |" in th._melt_progress_line("[GPU0 Melt]", 300.0, 5400.0, 1, info, None)

    def test_progress_line_temperature_unavailable(self, th):
        info = dict(window_mean=48912.0, window_cv=0.3, cycle_cv_pct=1.8, best=51004.0, throttle_count=0,
                    reason="x", temp_slope_c_per_min=None, window_temp_mean_C=None)
        assert "| power n/a | temp n/a |" in th._melt_progress_line("[CPU Melt]", 300.0, 5400.0, 1, info, None)

    def test_verdict_headline_carries_the_window_temperature(self, th):
        lines = th._melt_verdict_lines("[GPU1 Melt]", _outcome(True, window_temp_mean_C=61.4), runtime_s=4704.0)
        assert lines[0][1].endswith("power 180–655 W, temp 61 °C")
        lines = th._melt_verdict_lines("[GPU1 Melt]", _outcome(False, stop_rules_failed=["temp_slope"],
                                                                stop_reason="temperature still trending -1.1 °C/min (limit 0.5)",
                                                                window_temp_mean_C=61.4), runtime_s=5400.0)
        assert lines[0][1].endswith("power 180–655 W, temp 61 °C")
        assert "blocking rule at stop: temperature still trending -1.1 °C/min (limit 0.5)" in lines[2][1]

    def test_temp_slope_blocker_is_not_called_measurement_noise(self, th):
        assert th._noise_limited(_outcome(False, stop_rules_failed=["plateau_cv", "temp_slope"],
                                          thermal_throttle_cycles=0)) is False
        assert th._noise_limited(_outcome(False, stop_rules_failed=["temp_slope"],
                                          thermal_throttle_cycles=0)) is False
        assert th._noise_limited(_outcome(False, stop_rules_failed=["plateau_cv"],
                                          thermal_throttle_cycles=0)) is True

    def test_result_fields_include_temperature(self, th):
        for field in ("temp_slope_c_per_min", "window_temp_mean_C", "temp_mean_C", "temp_max_C",
                      "temp_sibling_delta_C"):
            assert field in th.MELT_RESULT_FIELDS

    def test_outcome_fields_summarise_temperature(self, th):
        records = [th.MeltCycle(float(t), 200.0, False, temp_mean_C=60.0 + (t % 2), temp_max_C=65.0 + (t % 3),
                                power_min_W=400.0, power_max_W=600.0, reload_ms=3.0) for t in range(0, 601)]
        stop_info = dict(reason="settled", rules_failed=[], window_cv=0.2, cycle_cv_pct=1.8,
                         temp_slope_c_per_min=-0.123456, window_temp_mean_C=60.5)
        run = th._MeltRun(records, True, 600.0, stop_info)
        workload = types.SimpleNamespace(ops_per_burst=6, burst_ms_measured=140.0,
                                         cool_gap=types.SimpleNamespace(path="h2d_pinned"))
        outcome = th._melt_outcome_fields(run, workload, 60.0, 100.0)
        assert outcome["temp_slope_c_per_min"] == -0.123
        assert outcome["window_temp_mean_C"] == 60.5
        assert outcome["temp_mean_C"] == pytest.approx(60.5, abs=0.01)
        assert outcome["temp_max_C"] == 67.0
        assert outcome["temp_sibling_delta_C"] is None

    def test_outcome_fields_without_temperature(self, th):
        records = [th.MeltCycle(float(t), 200.0, False) for t in range(0, 601)]
        run = th._MeltRun(records, True, 600.0, dict(reason="settled", rules_failed=[]))
        workload = types.SimpleNamespace(ops_per_burst=6, burst_ms_measured=140.0,
                                         cool_gap=types.SimpleNamespace(path="in_place"))
        outcome = th._melt_outcome_fields(run, workload, 60.0, 100.0)
        assert outcome["temp_mean_C"] is None and outcome["temp_max_C"] is None
        assert outcome["temp_slope_c_per_min"] is None and outcome["window_temp_mean_C"] is None

    def test_trajectory_carries_per_cycle_temperature(self, th):
        perf = {"iteration_telemetry": [{"iteration": 0}, {"iteration": 1}]}
        records = [th.MeltCycle(0.0, 1.0, False, temp_mean_C=61.0, power_min_W=1.0, power_max_W=2.0),
                   th.MeltCycle(1.0, 1.0, False)]
        th._attach_melt_trajectory(perf, records)
        assert perf["iteration_telemetry"][0]["temp_mean_C"] == 61.0
        assert perf["iteration_telemetry"][1]["temp_mean_C"] is None

    def test_melt_params_carry_the_temperature_knobs(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-temp-slope-c-per-min", "0.3", "--melt-temp-window-s", "300",
                                  "--melt-temp-sibling-warn-c", "8"])
        workload = types.SimpleNamespace(dims=(8, 4096, 4096, 4096), dtype_str="float64")
        params = th._melt_params(args, workload)
        assert params["temp_slope_c_per_min"] == 0.3
        assert params["temp_window_s"] == 300.0
        assert params["temp_sibling_warn_c"] == 8.0

    def test_dry_run_line_shows_the_temperature_rule(self, th, parser):
        assert "temp slope ≤ 0.5 °C/min over 300 s" in th._melt_dry_run_line(parser.parse_args(["--melt"]))
        line = th._melt_dry_run_line(parser.parse_args(["--melt", "--melt-temp-window-s", "180"]))
        assert "temp slope ≤ 0.5 °C/min over 180 s" in line
        assert "temp slope off" in th._melt_dry_run_line(parser.parse_args(["--melt", "--melt-temp-slope-c-per-min", "0"]))


# ──────────────────────────────────────────────────────────────
# 12. Sibling temperature offset (multi-GPU parent): field run 2 GPU1 ran 15 °C hotter
#     than its siblings at identical power and identical GFLOP/s.
# ──────────────────────────────────────────────────────────────
def _gpu_result(gpu_index, end_temp_c, power_w, settled=True, temp_mean_c=None):
    perf = _melt_perf(settled=settled, settle_s=300.1 if settled else None)
    perf["end_temp_C"] = end_temp_c
    perf["temp_mean_C"] = end_temp_c if temp_mean_c is None else temp_mean_c
    perf["temp_sibling_delta_C"] = None
    perf["telemetry"] = {"power_W_mean": power_w}
    perf["params"]["temp_sibling_warn_c"] = 10.0
    return {"gpu_index": gpu_index, "serial": f"S{gpu_index}", "telemetry_stats": {}, "benchmarks": [perf]}


def _field_run2(run):
    temps = {1: (46.1, 61.4, 47.1, 45.0), 2: (45.8, 57.4, 46.9, 44.8)}[run]
    powers = (415.0, 415.0, 420.0, 426.0)
    return [_gpu_result(i, t, p) for i, (t, p) in enumerate(zip(temps, powers))]


class TestSiblingTempWarnings:

    def test_run1_flags_gpu1_only(self, th):
        warnings = th._melt_sibling_temp_warnings(_field_run2(1), warn_c=10.0, power_tolerance_pct=5.0)
        assert len(warnings) == 1
        warning = warnings[0]
        assert warning.gpu_index == 1
        assert warning.delta_c == pytest.approx(15.3, abs=0.05)
        assert (warning.own_temp_c, warning.sibling_median_c, warning.power_w) == (61.4, 46.1, 415.0)
        assert warning.message == ("[GPU1 Melt] ends the dose 15.3 °C hotter than siblings at equal power "
                                   "(61.4 vs 46.1 °C @ 415 W) — check thermal path "
                                   "(TIM coverage, mount pressure, coolant flow position)")

    def test_run2_still_flags_gpu1(self, th):
        warnings = th._melt_sibling_temp_warnings(_field_run2(2), warn_c=10.0, power_tolerance_pct=5.0)
        assert [(w.gpu_index, round(w.delta_c, 1)) for w in warnings] == [(1, 11.6)]

    def test_similar_temperatures_give_no_warning(self, th):
        results = [_gpu_result(i, t, 415.0) for i, t in enumerate((46.1, 47.0, 45.2, 46.8))]
        assert th._melt_sibling_temp_warnings(results, warn_c=10.0, power_tolerance_pct=5.0) == []

    def test_hot_gpu_at_lower_power_has_no_comparable_sibling(self, th):
        results = _field_run2(1)
        results[1]["benchmarks"][0]["telemetry"]["power_W_mean"] = 415.0 * 0.8
        assert th._melt_sibling_temp_warnings(results, warn_c=10.0, power_tolerance_pct=5.0) == []

    def test_single_gpu_is_a_no_op(self, th):
        assert th._melt_sibling_temp_warnings(_field_run2(1)[1:2], warn_c=10.0, power_tolerance_pct=5.0) == []

    def test_zero_threshold_disables(self, th):
        assert th._melt_sibling_temp_warnings(_field_run2(1), warn_c=0.0, power_tolerance_pct=5.0) == []

    def test_gpus_without_temperature_or_power_are_left_out(self, th):
        results = _field_run2(1)
        results[2]["benchmarks"][0]["end_temp_C"] = None             # no temperature telemetry
        del results[3]["benchmarks"][0]["telemetry"]["power_W_mean"]  # no power telemetry
        results.append({"gpu_index": 4, "benchmarks": [{"name": "Batched GEMM", "telemetry": {"power_W_mean": 415.0}}]})
        warnings = th._melt_sibling_temp_warnings(results, warn_c=10.0, power_tolerance_pct=5.0)
        assert [w.gpu_index for w in warnings] == [1]
        assert warnings[0].sibling_median_c == 46.1   # GPU0 is the only comparable sibling left

    def test_threshold_comes_from_the_melt_result_params_so_yaml_applies(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-temp-sibling-warn-c", "3"])
        results = _field_run2(1)
        assert th._melt_sibling_warn_threshold(results, args) == 10.0     # the child ran with 10 (YAML or CLI)
        for r in results:
            del r["benchmarks"][0]["params"]["temp_sibling_warn_c"]
        assert th._melt_sibling_warn_threshold(results, args) == 3.0      # older result: fall back to the flag

    def test_annotation_lands_on_the_hot_gpu_only(self, th):
        results = _field_run2(1)
        warnings = th._melt_sibling_temp_warnings(results, warn_c=10.0, power_tolerance_pct=5.0)
        th._annotate_sibling_temp_deltas(results, warnings)
        deltas = [r["benchmarks"][0]["temp_sibling_delta_C"] for r in results]
        assert deltas == [None, 15.3, None, None]

    def test_compares_the_end_of_the_dose_not_the_run_average(self, th):
        # field run 5, 2026-10-13: GPU1 averaged 9.7 °C above its siblings over the hour (under the
        # old 10 °C limit) but ended 8.1 °C above them; healthy GPUs end within ~2 °C of each other.
        means, ends = (43.9, 53.6, 43.6, 43.9), (43.9, 52.0, 43.7, 44.0)
        powers = (401.0, 400.0, 395.0, 407.0)
        results = [_gpu_result(i, e, p, temp_mean_c=m) for i, (m, e, p) in enumerate(zip(means, ends, powers))]
        warnings = th._melt_sibling_temp_warnings(results, warn_c=th.MELT_TEMP_SIBLING_WARN_C, power_tolerance_pct=5.0)
        assert [(w.gpu_index, round(w.delta_c, 1)) for w in warnings] == [(1, 8.1)]
        assert (warnings[0].own_temp_c, warnings[0].sibling_median_c) == (52.0, 43.9)

    def test_gpu_that_started_hot_but_recovered_is_not_flagged(self, th):
        # field run 4: GPU2 averaged +4.5 °C over the hour but ended +0.5 °C.  Its hot start must
        # not count against it once the dose has brought it level with its siblings.
        results = [_gpu_result(i, e, 400.0, temp_mean_c=m)
                   for i, (m, e) in enumerate(zip((46.0, 46.0, 52.0, 46.0), (46.0, 46.0, 46.5, 46.0)))]
        assert th._melt_sibling_temp_warnings(results, warn_c=5.0, power_tolerance_pct=5.0) == []

    def test_default_limit_is_5_c(self, th):
        assert th.MELT_TEMP_SIBLING_WARN_C == 5.0

    def test_summary_note_carries_the_sibling_offset(self, th):
        perf = _melt_perf(settled=True, settle_s=300.1)
        perf["temp_sibling_delta_C"] = 15.3
        status, notes = th._summary_status(perf, {}, 70.0)
        assert status == "PASS"
        assert notes == "Settled in 5.0 min (23,900 → 54,120) · hot vs siblings (+15.3 °C)"
        perf = _melt_perf(settled=False, settle_s=None)
        perf["temp_sibling_delta_C"] = 11.6
        assert th._summary_status(perf, {}, 70.0) == ("FAIL", "Not settled (23,900 → 54,120) · hot vs siblings (+11.6 °C)")

    def test_summary_note_without_offset_is_unchanged(self, th):
        perf = _melt_perf(settled=True)
        perf["temp_sibling_delta_C"] = None
        assert th._summary_status(perf, {}, 70.0)[1] == "Settled in 78.4 min (23,900 → 54,120)"

    def test_summary_rows_show_the_offset_after_annotation(self, th):
        results = _field_run2(1)
        th._annotate_sibling_temp_deltas(results, th._melt_sibling_temp_warnings(results, warn_c=10.0))
        rows = th._build_summary_rows(results, 70.0)
        assert [r["status"] for r in rows] == ["PASS"] * 4
        assert rows[1]["notes"] == "Settled in 5.0 min (23,900 → 54,120) · hot vs siblings (+15.3 °C)"
        assert rows[0]["notes"] == "Settled in 5.0 min (23,900 → 54,120)"


# ──────────────────────────────────────────────────────────────
# 13. Replay of the 2026-10-07 GH200 field run 2 (temperature rule)
# ──────────────────────────────────────────────────────────────
FIELD_RUN2_CYCLE_S = 0.235
FIELD_RUN2_GFLOPS = 53400.0
FIELD_RUN2_FIRST_GPU1_TEMPS = (63.6, 62.2, 60.7, 59.8, 59.3)   # per-minute means; still falling at 300 s
FIELD_RUN2_FIRST_GPU0_TEMPS = (45.3, 45.8, 45.9, 45.9, 45.9)   # warm-up in minute 0, flat after
FIELD_RUN2_RERUN_GPU1_TEMPS = (56.9, 57.8, 57.8, 57.6, 57.4)   # back-to-back rerun: 6.7 °C cooler and flat


def _piecewise_linear(knots):
    """Interpolate between (t, value) knots; hold before the first, extend the last segment after the last."""
    def value_at(t):
        if t <= knots[0][0]:
            return knots[0][1]
        for (t_a, v_a), (t_b, v_b) in zip(knots, knots[1:]):
            if t <= t_b:
                return v_a + (t - t_a) / (t_b - t_a) * (v_b - v_a)
        (t_a, v_a), (t_b, v_b) = knots[-2], knots[-1]
        return v_b + (t - t_b) * (v_b - v_a) / (t_b - t_a)
    return value_at


def _field_run2_records(minute_means, flat_from_s=None, t_end_s=300.0, seed=1491, cv_pct=1.9, temp_sigma_c=1.0):
    """Per-cycle records at the real 0.235 s cadence.

    GFLOP/s flat at 53,400 with 1.9 % cycle noise, no throttle; temperature is
    the linear interpolation of the per-minute means (placed at minute centres)
    plus gaussian σ = 1 °C sample noise, held flat from ``flat_from_s`` on.
    """
    import random
    rng = random.Random(seed)
    trajectory = _piecewise_linear([(60.0 * k + 30.0, c) for k, c in enumerate(minute_means)])
    records, t = [], FIELD_RUN2_CYCLE_S
    while t <= t_end_s + 1e-9:
        t_temp = t if flat_from_s is None else min(t, flat_from_s)
        records.append((round(t, 6), rng.gauss(FIELD_RUN2_GFLOPS, FIELD_RUN2_GFLOPS * cv_pct / 100.0), False,
                        trajectory(t_temp) + rng.gauss(0.0, temp_sigma_c)))
        t += FIELD_RUN2_CYCLE_S
    return records


def _first_settle(th, args, records):
    """``(settle_s, info)`` of the first cycle the tracker declares settled, or ``(None, last info)``.

    Cycles inside the min-duration gate can neither settle nor raise ``best``,
    so evaluation starts at the gate (keeps the replay fast).
    """
    tracker = th._ConvergenceTracker.from_args(args)
    info = {}
    for i, record in enumerate(records):
        if record[0] < args.melt_min_duration:
            continue
        settled, info = tracker.update(records[:i + 1], record[0])
        if settled:
            return record[0], info
    return None, info


class TestGH200FieldRun2Replay:
    """Second GH200 field run.  Performance settled on all four GPUs at the
    300 s gate, but GPU1 was still cooling: 63.6 → 59.3 °C over the five
    minutes (−1.1 °C/min on average) at the same ~415 W and ~53.4k GFLOP/s as
    its siblings.  A rerun minutes later started 6.7 °C cooler and was flat."""

    def test_run1_gpu1_is_held_by_the_temperature_rule_at_defaults(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-min-duration", "300"])
        rules = th._SettleRules.from_args(args)._asdict()
        records = _field_run2_records(FIELD_RUN2_FIRST_GPU1_TEMPS, flat_from_s=300.0, t_end_s=420.0)
        for now_s in (300.0, 330.0):
            settled, info = th.melt_converged(records, now_s, **rules)
            assert settled is False, f"settled at {now_s} s"
            assert info["rules_failed"] == ["temp_slope"], info["rules_failed"]
            assert info["temp_slope_c_per_min"] < -0.5
        settle_s, info = _first_settle(th, args, records)
        assert settle_s is not None, f"never settled: {info['reason']}"
        assert 300.0 < settle_s <= 420.0
        assert abs(info["temp_slope_c_per_min"]) <= 0.5
        assert info["window_temp_mean_C"] == pytest.approx(59.1, abs=0.5)

    def test_run1_gpu1_a_60_s_horizon_is_blind_to_the_trend(self, th, parser):
        # Why the horizon is 300 s and not the performance window: between the
        # last two minute centres GPU1 only lost 0.5 °C, and a 6-bin slope with
        # 1 °C sample noise carries ~0.2 °C/min of estimator noise — so at 60 s
        # the gate waves it through while the 5 min trend is -1.1 °C/min.
        records = _field_run2_records(FIELD_RUN2_FIRST_GPU1_TEMPS, flat_from_s=300.0, t_end_s=420.0)
        short = parser.parse_args(["--melt", "--melt-min-duration", "300", "--melt-temp-window-s", "60"])
        settle_s, info = _first_settle(th, short, records)
        assert settle_s == pytest.approx(300.0, abs=1.0)
        assert -1.0 < info["temp_slope_c_per_min"] < 0.0
        _, at_defaults = th.melt_converged(records, 300.0, **th._SettleRules.from_args(parser.parse_args(["--melt", "--melt-min-duration", "300"]))._asdict())
        assert at_defaults["temp_slope_c_per_min"] == pytest.approx(-1.1, abs=0.1)

    def test_run1_gpu0_warm_up_ramp_does_not_delay_a_healthy_gpu(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-min-duration", "300"])
        settle_s, info = _first_settle(th, args, _field_run2_records(FIELD_RUN2_FIRST_GPU0_TEMPS, t_end_s=420.0))
        assert settle_s == pytest.approx(300.0, abs=1.0), info["reason"]
        assert info["window_temp_mean_C"] == pytest.approx(45.9, abs=0.3)

    def test_run2_gpu1_flat_and_cooler_settles_at_the_gate(self, th, parser):
        for extra in ([], ["--melt-temp-window-s", "60"]):
            args = parser.parse_args(["--melt", "--melt-min-duration", "300"] + extra)
            settle_s, info = _first_settle(th, args, _field_run2_records(FIELD_RUN2_RERUN_GPU1_TEMPS, t_end_s=420.0))
            assert settle_s == pytest.approx(300.0, abs=1.0), (extra, info["reason"])
            assert info["window_temp_mean_C"] == pytest.approx(57.3, abs=0.5)

    def test_temperature_rule_disabled_ignores_a_steep_trend(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-min-duration", "300", "--melt-temp-slope-c-per-min", "0"])
        records = _trending(_field_run2_records((60.0,) * 5, t_end_s=420.0), 80.0, -2.0)
        settle_s, info = _first_settle(th, args, records)
        assert settle_s == pytest.approx(300.0, abs=1.0)
        assert info["temp_slope_c_per_min"] is None

    def test_cpu_records_without_temperature_settle_at_the_gate(self, th, parser):
        args = parser.parse_args(["--melt", "--melt-min-duration", "300"])
        records = [(t, g, thr) for t, g, thr, _ in _field_run2_records((0.0,) * 5, t_end_s=420.0)]
        settle_s, info = _first_settle(th, args, records)
        assert settle_s == pytest.approx(300.0, abs=1.0)
        assert info["temp_slope_c_per_min"] is None and info["window_temp_mean_C"] is None


# ──────────────────────────────────────────────────────────────
# 14. Fixed dose: melt runs for a set duration, then the rules judge the final window
# ──────────────────────────────────────────────────────────────
class TestBeforeAfterLine:

    def test_full_line(self, th):
        line = th._melt_before_after_line(_outcome(True, start_gflops=52154.0, plateau_gflops=52521.0))
        assert line == ("  start → end: temp 67.7 → 58.9 °C (−8.8 at 423 W) | 52,154 → 52,521 GFLOP/s (+0.7%) | "
                        "thermal throttle 18 → 0 cycles")

    def test_temperature_absent(self, th):
        line = th._melt_before_after_line(_outcome(True, start_temp_C=None, end_temp_C=None, power_mean_W=None,
                                                   start_throttle_cycles=0, end_throttle_cycles=0,
                                                   start_gflops=26.0, plateau_gflops=25.5))
        assert line == "  start → end: temp n/a | 26 → 26 GFLOP/s (−1.9%) | thermal throttle 0 → 0 cycles"

    def test_temperature_without_power(self, th):
        line = th._melt_before_after_line(_outcome(True, power_mean_W=None, start_temp_C=50.0, end_temp_C=52.5))
        assert "temp 50.0 → 52.5 °C (+2.5) |" in line

    def test_result_fields_include_before_after(self, th):
        for field in ("start_temp_C", "end_temp_C", "start_throttle_cycles", "end_throttle_cycles", "power_mean_W"):
            assert field in th.MELT_RESULT_FIELDS

    def test_outcome_fields_summarise_start_and_end(self, th):
        records = [th.MeltCycle(float(t), 200.0, thermal_throttled=t < 30, temp_mean_C=70.0 - t / 60.0,
                                power_min_W=400.0, power_max_W=600.0, power_mean_W=500.0 + (t % 2))
                   for t in range(0, 601)]
        run = th._MeltRun(records, True, 600.0, dict(reason="settled", rules_failed=[]))
        workload = types.SimpleNamespace(ops_per_burst=6, burst_ms_measured=140.0,
                                         cool_gap=types.SimpleNamespace(path="h2d_pinned"))
        outcome = th._melt_outcome_fields(run, workload, 60.0, 100.0)
        assert outcome["start_temp_C"] == pytest.approx(69.5, abs=0.01)
        assert outcome["end_temp_C"] == pytest.approx(60.5, abs=0.01)
        assert outcome["power_mean_W"] == pytest.approx(500.5, abs=0.01)
        assert (outcome["start_throttle_cycles"], outcome["end_throttle_cycles"]) == (30, 0)

    def test_outcome_fields_without_telemetry(self, th):
        records = [th.MeltCycle(float(t), 200.0, False) for t in range(0, 601)]
        run = th._MeltRun(records, True, 600.0, dict(reason="settled", rules_failed=[]))
        workload = types.SimpleNamespace(ops_per_burst=6, burst_ms_measured=140.0,
                                         cool_gap=types.SimpleNamespace(path="in_place"))
        outcome = th._melt_outcome_fields(run, workload, 60.0, 100.0)
        assert outcome["start_temp_C"] is None and outcome["end_temp_C"] is None and outcome["power_mean_W"] is None


class _FakeClock:
    """``time.perf_counter`` stand-in advancing a fixed step per call — makes the melt loop deterministic."""

    def __init__(self, step_s):
        self.now, self.step_s = 0.0, step_s

    def __call__(self):
        self.now += self.step_s
        return self.now


class TestFixedDose:

    def _run(self, th, parser, monkeypatch, extra):
        import torch
        args = parser.parse_args(["--melt", "--melt-min-duration", "2", "--melt-max-duration", "2",
                                  "--melt-window-s", "1", "--melt-bin-s", "0.25", "--melt-cool-ms", "0",
                                  "--melt-plateau-cv-pct", "100", "--melt-settled-pct", "1"] + extra)
        workload = th._MeltWorkload(torch.ones(1, 4, 4), torch.ones(1, 4, 4), _OrderRecordingGap([]), 1, 0.01, "float32")
        monkeypatch.setattr(th.time, "perf_counter", _FakeClock(0.02))
        monkeypatch.setattr(th.time, "sleep", lambda s: None)
        return th._run_melt_cycles(args, torch.device("cpu"), logging.getLogger("t"),
                                   _OrderRecordingThread([]), None, workload, "[CPU Melt]")

    def test_dose_ends_settled_when_the_final_window_passes(self, th, parser, monkeypatch):
        run = self._run(th, parser, monkeypatch, [])
        assert run.settled is True
        assert run.settle_s >= 2.0
        assert run.records[-1].t_s == run.settle_s
        assert run.stop_info["reason"] == "settled"

    def test_dose_ends_not_settled_when_the_final_window_fails(self, th, parser, monkeypatch):
        run = self._run(th, parser, monkeypatch, ["--melt-target-gflops", "1e12"])
        assert run.settled is False
        assert run.records[-1].t_s >= 2.0
        assert run.stop_info["rules_failed"] == ["below_target"]

    def test_min_equal_max_judges_the_final_window(self, th):
        records = _noisy(53400.0, 0.235, 3600.5, step=0.235, cv_pct=1.9, seed=3600)
        settled, info = th.melt_converged(records, 3600.5, **dict(DEFAULTS, min_duration_s=3600.0))
        assert settled is True, info["reason"]
        settled, _ = th.melt_converged(records, 3599.0, **dict(DEFAULTS, min_duration_s=3600.0))
        assert settled is False
