#!/usr/bin/env python3
# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""torch-hammer-reporter -- Fleet report from torch-hammer benchmark output.

Generates a CLI summary (default) and optional HTML report from torch-hammer
output files.  Supports compact CSV (--compact), summary CSV (--summary-csv),
JSON (--json-output), and shell-dump formats.  Auto-detects the input format.

Usage::

    # CLI summary (default -- works over SSH, meaningful exit code)
    python hammer_report.py results/
    python hammer_report.py results.csv
    python hammer_report.py results.json

    # CLI summary + HTML report
    python hammer_report.py results/ -o report.html

    # Filter to one benchmark
    python hammer_report.py results.csv --benchmark "Batched GEMM" --dtype float32

    # Shell dump from HPC
    python hammer_report.py dump.txt --shell-output

    # Quiet mode for CI (exit 0 = pass, exit 1 = outliers)
    python hammer_report.py results/ --quiet --outlier-threshold 10
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import html
import io
import json
import math
import re
import statistics
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import plotly.offline as _plotly_offline
    _PLOTLY_AVAILABLE = True
except ImportError:
    _plotly_offline = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Per-iteration data (populated by load_verbose_log / load_json)
# ---------------------------------------------------------------------------
# Key: "hostname:gpu_idx:benchmark:dtype"
# Value: list of per-iteration dicts
_iteration_data = {}  # type: Dict[str, List[Dict]]


# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe-Ito)
# ---------------------------------------------------------------------------

OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPUResult:
    """One benchmark result for one GPU on one node."""
    hostname: str
    gpu: int
    gpu_model: str
    serial: str
    benchmark: str
    dtype: str
    iterations: int
    runtime_s: float
    min_val: float
    mean_val: float
    max_val: float
    unit: str
    power_avg_w: float
    temp_max_c: float
    sm_util_mean: float = 0.0
    mem_bw_util_mean: float = 0.0
    gpu_clock_mean: float = 0.0
    throttled: bool = False
    throttle_samples: int = 0


# ---------------------------------------------------------------------------
# Parsing -- multi-format input
# ---------------------------------------------------------------------------

COMPACT_COLS = {
    "hostname", "gpu", "gpu_model", "serial", "benchmark", "dtype",
    "iterations", "runtime_s", "min", "mean", "max", "unit",
    "power_avg_w", "temp_max_c",
}

SUMMARY_COLS = {
    "test", "dtype", "gpu", "serial", "performance", "unit",
    "power_avg_W", "temp_max_C", "status", "notes",
}

INTERACTIVE_RESULT_COLS = [
    "hostname", "gpu", "gpu_model", "serial", "benchmark", "dtype",
    "mean_val", "min_val", "max_val", "unit", "power_avg_w", "temp_max_c",
    "sm_util_mean", "mem_bw_util_mean", "gpu_clock_mean",
    "throttled", "throttle_samples",
]

_parse_warnings = 0


def _safe_float(val, default=0.0):
    # type: (str, float) -> float
    try:
        v = float(val)
        if math.isfinite(v):
            return v
        return default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    # type: (str, int) -> int
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _opt_float(val):
    # type: (object) -> Optional[float]
    """Optional telemetry value: None when absent, blank, N/A or unparseable.

    A measured 0 stays 0.0 -- zero is data, missing is null (T19).
    """
    if val is None:
        return None
    try:
        v = float(val)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


_OPTIONAL_ITERATION_TELEMETRY = (
    "power_W", "temp_gpu_C", "sm_util", "mem_bw_util", "gpu_clock",
)


def _optional_iteration_telemetry(source):
    # type: (Dict[str, object]) -> Dict[str, Optional[float]]
    """Per-iteration telemetry columns that a collector may not provide."""
    return {k: _opt_float(source.get(k)) for k in _OPTIONAL_ITERATION_TELEMETRY}


def _safe_json(obj):
    # type: (object) -> str
    """json.dumps with `</` escaped so the text can sit inside <script>.

    None serialises as `null`, which the JS reads as a gap (T19).
    """
    return json.dumps(obj).replace("</", "<\\/")


def _parse_compact_row(row):
    # type: (Dict[str, str]) -> Optional[GPUResult]
    """Parse one row from --compact CSV output."""
    global _parse_warnings
    try:
        return GPUResult(
            hostname=row.get("hostname", "").strip(),
            gpu=int(row["gpu"]),
            gpu_model=row.get("gpu_model", "").strip(),
            serial=row.get("serial", "").strip(),
            benchmark=row["benchmark"].strip(),
            dtype=row.get("dtype", "").strip(),
            iterations=_safe_int(row.get("iterations", "0")),
            runtime_s=_safe_float(row.get("runtime_s", "0")),
            min_val=float(row["min"]),
            mean_val=float(row["mean"]),
            max_val=float(row["max"]),
            unit=row.get("unit", "").strip(),
            power_avg_w=_safe_float(row.get("power_avg_w", "0")),
            temp_max_c=_safe_float(row.get("temp_max_c", "0")),
            sm_util_mean=_safe_float(row.get("sm_util_mean", "0")),
            mem_bw_util_mean=_safe_float(row.get("mem_bw_util_mean", "0")),
            gpu_clock_mean=_safe_float(row.get("gpu_clock_mean", "0")),
            throttled=row.get("throttled", "").strip().lower() in ("true", "1", "yes"),
        )
    except (KeyError, ValueError):
        _parse_warnings += 1
        return None


def _parse_summary_row(row, hostname=""):
    # type: (Dict[str, str], str) -> Optional[GPUResult]
    """Parse one row from --summary-csv output."""
    global _parse_warnings
    try:
        perf = float(row["performance"])
        return GPUResult(
            hostname=hostname,
            gpu=_safe_int(row.get("gpu", "0")),
            gpu_model="",
            serial=row.get("serial", "").strip(),
            benchmark=row["test"].strip(),
            dtype=row.get("dtype", "").strip(),
            iterations=0,
            runtime_s=0.0,
            min_val=perf,
            mean_val=perf,
            max_val=perf,
            unit=row.get("unit", "").strip(),
            power_avg_w=_safe_float(row.get("power_avg_W", "0")),
            temp_max_c=_safe_float(row.get("temp_max_C", "0")),
        )
    except (KeyError, ValueError):
        _parse_warnings += 1
        return None


def load_compact_csv(path, hostname_override=""):
    # type: (Path, str) -> List[GPUResult]
    """Load compact-mode CSV (--compact)."""
    results = []  # type: List[GPUResult]
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not COMPACT_COLS.issubset(cols):
            if "hostname" not in cols and not hostname_override:
                print("  Warning: {} missing expected columns: {}".format(
                    path.name, COMPACT_COLS - cols), file=sys.stderr)
        for row in reader:
            r = _parse_compact_row(row)
            if r:
                if hostname_override and not r.hostname:
                    r.hostname = hostname_override
                results.append(r)
    return results


def load_summary_csv(path, hostname=""):
    # type: (Path, str) -> List[GPUResult]
    """Load summary CSV (--summary-csv)."""
    results = []  # type: List[GPUResult]
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = _parse_summary_row(row, hostname=hostname)
            if r:
                results.append(r)
    return results


def load_json(path):
    # type: (Path) -> List[GPUResult]
    """Load JSON output (--json-output)."""
    results = []  # type: List[GPUResult]
    with path.open() as f:
        data = json.load(f)

    hostname = data.get("metadata", {}).get("hostname", path.stem)

    for gpu_entry in data.get("gpus", []):
        gpu_idx = gpu_entry.get("gpu_index", 0)
        model = gpu_entry.get("model", "")
        serial = gpu_entry.get("serial", "")

        for bench in gpu_entry.get("benchmarks", []):
            if bench is None:
                continue
            results.append(GPUResult(
                hostname=hostname,
                gpu=gpu_idx,
                gpu_model=model,
                serial=serial,
                benchmark=bench.get("name", ""),
                dtype=bench.get("params", {}).get("dtype", ""),
                iterations=bench.get("iterations", 0),
                runtime_s=bench.get("runtime_s", 0.0),
                min_val=bench.get("min", 0.0),
                mean_val=bench.get("mean", 0.0),
                max_val=bench.get("max", 0.0),
                unit=bench.get("unit", ""),
                power_avg_w=_safe_float(
                    bench.get("telemetry", {}).get("power_W_mean", 0)),
                temp_max_c=_safe_float(
                    bench.get("telemetry", {}).get("temp_gpu_C_max", 0)),
                sm_util_mean=_safe_float(
                    bench.get("telemetry", {}).get("sm_util_mean", 0)),
                mem_bw_util_mean=_safe_float(
                    bench.get("telemetry", {}).get("mem_bw_util_mean", 0)),
                gpu_clock_mean=_safe_float(
                    bench.get("telemetry", {}).get("gpu_clock_mean", 0)),
            ))

            # Detect throttling from JSON: check bench-level flag and
            # per-iteration telemetry readings
            new_r = results[-1]
            if bench.get("throttled"):
                new_r.throttled = True
            # Count throttled iterations from per-iteration telemetry
            iter_tel = bench.get("iteration_telemetry", [])
            thr_count = 0
            for entry in iter_tel:
                tel = entry.get("telemetry", {})
                if (_safe_int(tel.get("throttled", 0)) > 0
                        or _safe_int(tel.get("hw_slowdown", 0)) > 0
                        or _safe_int(tel.get("sw_slowdown", 0)) > 0
                        or _safe_int(tel.get("power_limit", 0)) > 0):
                    thr_count += 1
            if thr_count > 0:
                new_r.throttled = True
                new_r.throttle_samples = thr_count

            # Populate per-iteration data from JSON iteration_telemetry
            bench_name = bench.get("name", "")
            bench_dtype = bench.get("params", {}).get("dtype", "")
            iter_tel = bench.get("iteration_telemetry", [])
            if iter_tel:
                iter_key = "{}:{}:{}:{}".format(
                    hostname, gpu_idx, bench_name, bench_dtype)
                iter_records = []
                for entry in iter_tel:
                    tel = entry.get("telemetry", {})
                    record = {
                        "iteration": entry.get("iteration", 0),
                        "performance": _safe_float(entry.get("performance", 0)),
                    }
                    record.update(_optional_iteration_telemetry(tel))
                    record["throttled"] = _safe_int(tel.get("throttled", 0))
                    iter_records.append(record)
                _iteration_data[iter_key] = iter_records

    return results


def load_shell_output(path):
    # type: (Path) -> List[GPUResult]
    """Parse output of: for file in *; do echo "file: $file"; cat $file; done"""
    results = []  # type: List[GPUResult]
    current_host = None  # type: Optional[str]
    current_lines = []  # type: List[str]
    header = None  # type: Optional[str]

    def flush():
        # type: () -> None
        nonlocal header
        if current_host is None or header is None:
            return
        text = header + "\n" + "\n".join(current_lines)
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            r = _parse_compact_row(row)
            if r:
                if not r.hostname:
                    r.hostname = current_host
                results.append(r)

    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("file: "):
                flush()
                current_host = line[6:].strip()
                current_lines = []
                header = None
            elif current_host is not None:
                if header is None and line.startswith("hostname"):
                    header = line
                elif header and line:
                    current_lines.append(line)
    flush()
    return results


# ---------------------------------------------------------------------------
# Verbose log-dir files (per-GPU, timestamp-prefixed CSV)
# ---------------------------------------------------------------------------

_LOG_PREFIX_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s+\w+\s+')

_METRIC_UNITS = {
    "gflops": "GFLOP/s",
    "img_s": "img/s",
    "gb_s": "GB/s",
    "mlups": "MLUP/s",
    "iter_s": "iter/s",
    "iter/s": "iter/s",
    "mops": "Mops/s",
}

_VERBOSE_BENCH_NAMES = {
    "gemm": "Batched GEMM",
    "conv": "Convolution",
    "fft3d": "3D FFT",
    "einsum": "Einsum Attention",
    "mem": "Memory Traffic",
    "heat": "Heat Equation",
    "schrod": "Schr\u00f6dinger Equation",
    "atomic": "Atomic Contention",
    "sparse_mm": "Sparse MM",
}


def _is_verbose_log(path):
    # type: (Path) -> bool
    """Return True if *path* looks like a verbose log-dir file."""
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    return bool(_LOG_PREFIX_RE.match(line))
    except (OSError, UnicodeDecodeError):
        pass
    return False


def _strip_log_prefix(line):
    # type: (str) -> str
    return _LOG_PREFIX_RE.sub('', line)


def _parse_filename_meta(path):
    # type: (Path) -> Tuple[int, str, str]
    """Extract (gpu_index, hostname, serial) from verbose log filename."""
    stem = path.stem
    parts = stem.split("_", 2)
    gpu_idx = 0
    hostname = ""
    serial = ""
    if len(parts) >= 1:
        gpu_str = parts[0].replace("gpu", "").replace("mps", "").replace("cpu", "")
        gpu_idx = _safe_int(gpu_str, 0)
    if len(parts) >= 2:
        hostname = parts[1]
    if len(parts) >= 3:
        serial = parts[2]
    return gpu_idx, hostname, serial


def load_verbose_log(path):
    # type: (Path) -> List[GPUResult]
    """Parse a verbose log-dir file into GPUResult entries."""
    file_gpu, file_hostname, file_serial = _parse_filename_meta(path)

    current_cols = []   # type: List[str]
    metric_col = 4      # default column index for the metric
    metric_name = "gflops"
    groups = OrderedDict()  # type: OrderedDict[Tuple[str, str], List[Dict[str, str]]]

    with path.open() as f:
        for raw_line in f:
            stripped = _strip_log_prefix(raw_line.rstrip("\n")).strip()
            if not stripped:
                continue

            # Detect header line: "repeat, iter, test, dtype, <metric>, ..."
            if stripped.startswith("repeat") and ", iter," in stripped:
                current_cols = [c.strip() for c in stripped.split(",")]
                # The metric column is at index 4
                if len(current_cols) > 4:
                    metric_name = current_cols[4]
                continue

            # Skip lines that don't start with a digit (non-data)
            if not current_cols or not stripped[:1].isdigit():
                continue

            values = [v.strip() for v in stripped.split(",")]
            if len(values) < len(current_cols):
                continue

            row = OrderedDict(zip(current_cols, values))
            test = row.get("test", "")
            dtype = row.get("dtype", "")
            key = (test, dtype)
            row["_metric_name"] = metric_name
            groups.setdefault(key, []).append(row)

    results = []  # type: List[GPUResult]
    for (test, dtype), rows in groups.items():
        perf_vals = [_safe_float(r.get(r["_metric_name"], "0")) for r in rows]
        perf_vals = [v for v in perf_vals if v > 0]
        if not perf_vals:
            continue

        power_vals = [_safe_float(r.get("power_W", "0")) for r in rows]
        power_vals = [v for v in power_vals if v > 0]
        temp_vals = [_safe_float(r.get("temp_gpu_C", "0")) for r in rows]
        temp_vals = [v for v in temp_vals if v > 0]
        sm_vals = [_safe_float(r.get("sm_util", "0")) for r in rows]
        sm_vals = [v for v in sm_vals if v > 0]
        mem_bw_vals = [_safe_float(r.get("mem_bw_util", "0")) for r in rows]
        mem_bw_vals = [v for v in mem_bw_vals if v > 0]
        clock_vals = [_safe_float(r.get("gpu_clock", "0")) for r in rows]
        clock_vals = [v for v in clock_vals if v > 0]

        # Throttle: any iteration with hw_slowdown/sw_slowdown/power_limit/throttled
        throttle_samples = sum(
            1 for r in rows
            if _safe_int(r.get("throttled", "0")) > 0
            or _safe_int(r.get("hw_slowdown", "0")) > 0
            or _safe_int(r.get("sw_slowdown", "0")) > 0
            or _safe_int(r.get("power_limit", "0")) > 0
        )

        first = rows[0]
        hostname = first.get("hostname", "").strip() or file_hostname
        gpu_idx = _safe_int(first.get("device_id", ""), file_gpu)
        gpu_model = first.get("model", "").strip()
        serial = first.get("serial", "").strip() or file_serial
        mn = first.get("_metric_name", "gflops")
        unit = _METRIC_UNITS.get(mn, "GFLOP/s")
        bench_name = _VERBOSE_BENCH_NAMES.get(test, test)

        results.append(GPUResult(
            hostname=hostname,
            gpu=gpu_idx,
            gpu_model=gpu_model,
            serial=serial,
            benchmark=bench_name,
            dtype=dtype,
            iterations=len(perf_vals),
            runtime_s=0.0,
            min_val=min(perf_vals),
            mean_val=statistics.mean(perf_vals),
            max_val=max(perf_vals),
            unit=unit,
            power_avg_w=statistics.mean(power_vals) if power_vals else 0.0,
            temp_max_c=max(temp_vals) if temp_vals else 0.0,
            sm_util_mean=statistics.mean(sm_vals) if sm_vals else 0.0,
            mem_bw_util_mean=statistics.mean(mem_bw_vals) if mem_bw_vals else 0.0,
            gpu_clock_mean=statistics.mean(clock_vals) if clock_vals else 0.0,
            throttled=throttle_samples > 0,
            throttle_samples=throttle_samples,
        ))

        # Populate per-iteration data for interactive dashboards
        iter_key = "{}:{}:{}:{}".format(hostname, gpu_idx, bench_name, dtype)
        iter_records = []
        for i, r in enumerate(rows):
            mn = r.get("_metric_name", "gflops")
            record = {
                "iteration": i,
                "performance": _safe_float(r.get(mn, "0")),
            }
            record.update(_optional_iteration_telemetry(r))
            record["throttled"] = _safe_int(r.get("throttled", "0"))
            iter_records.append(record)
        if iter_records:
            _iteration_data[iter_key] = iter_records

    return results


def _detect_and_load(path):
    # type: (Path) -> List[GPUResult]
    """Auto-detect format and load a single file."""
    if path.suffix == ".json":
        return load_json(path)

    # Peek at first 512 bytes to detect format
    with path.open() as f:
        peek = f.read(512)

    # Shell dump?
    if "file: " in peek and "hostname" in peek:
        return load_shell_output(path)

    # JSON content?
    if peek.lstrip().startswith("{"):
        return load_json(path)

    # Summary CSV? (has "test" column instead of "benchmark")
    if "test," in peek and "performance," in peek:
        return load_summary_csv(path)

    # Verbose log file? (timestamp-prefixed lines from --log-dir)
    first_line = peek.lstrip().split('\n', 1)[0] if peek.strip() else ''
    if _LOG_PREFIX_RE.match(first_line):
        return load_verbose_log(path)

    # Default: compact CSV
    return load_compact_csv(path)


def load_node_map(path):
    # type: (Path) -> Dict[str, str]
    """Load hostname,location CSV into {hostname: location} dict."""
    node_map = {}  # type: Dict[str, str]
    with open(str(path), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            host = row.get("hostname", "").strip()
            loc = row.get("location", "").strip()
            if host and loc:
                node_map[host] = loc
    return node_map


def load_input(source, shell_mode=False):
    # type: (Path, bool) -> List[GPUResult]
    """Entry point: load from file, directory, or shell dump."""
    if shell_mode:
        return load_shell_output(source)

    if source.is_dir():
        results = []  # type: List[GPUResult]
        files = sorted(source.glob("*.csv")) + sorted(source.glob("*.json"))
        if not files:
            files = sorted(p for p in source.iterdir()
                           if p.is_file() and not p.name.startswith("."))
        for f in files:
            results.extend(_detect_and_load(f))
        return results

    return _detect_and_load(source)


# ---------------------------------------------------------------------------
# Grouping and stats
# ---------------------------------------------------------------------------

def _group_by(items, key):
    groups = OrderedDict()
    for item in items:
        k = key(item)
        groups.setdefault(k, []).append(item)
    return groups


def bench_key(r):
    # type: (GPUResult) -> Tuple[str, str]
    """Group key: (benchmark, dtype)."""
    return (r.benchmark, r.dtype)


def bench_label(r):
    # type: (GPUResult) -> str
    return "{} ({})".format(r.benchmark, r.dtype) if r.dtype else r.benchmark


def bench_label_from_key(key):
    # type: (Tuple[str, str]) -> str
    return "{} ({})".format(key[0], key[1]) if key[1] else key[0]


@dataclass
class BenchmarkStats:
    """Aggregated stats for one (benchmark, dtype) across the fleet."""
    name: str
    dtype: str
    unit: str
    gpu_means: List[float] = field(default_factory=list)
    node_means: Dict[str, float] = field(default_factory=dict)
    node_intra_cv: Dict[str, float] = field(default_factory=dict)
    results: List[GPUResult] = field(default_factory=list)
    power_values: List[float] = field(default_factory=list)
    temp_values: List[float] = field(default_factory=list)
    sm_util_values: List[float] = field(default_factory=list)
    mem_bw_util_values: List[float] = field(default_factory=list)
    gpu_clock_values: List[float] = field(default_factory=list)
    throttled_results: List[GPUResult] = field(default_factory=list)

    @property
    def fleet_mean(self):
        # type: () -> float
        return statistics.mean(self.gpu_means) if self.gpu_means else 0.0

    @property
    def fleet_cv(self):
        # type: () -> float
        m = self.fleet_mean
        if len(self.gpu_means) < 2 or m == 0:
            return 0.0
        return statistics.stdev(self.gpu_means) / m * 100

    @property
    def fleet_min(self):
        # type: () -> float
        return min(self.gpu_means) if self.gpu_means else 0.0

    @property
    def fleet_max(self):
        # type: () -> float
        return max(self.gpu_means) if self.gpu_means else 0.0


def compute_benchmark_stats(results):
    # type: (List[GPUResult]) -> OrderedDict
    """Compute per-benchmark stats grouped by (benchmark, dtype).

    Returns OrderedDict[(name, dtype)] -> BenchmarkStats.
    """
    groups = _group_by(results, bench_key)
    out = OrderedDict()

    for key, rows in groups.items():
        unit = rows[0].unit if rows else ""
        bs = BenchmarkStats(name=key[0], dtype=key[1], unit=unit, results=rows)

        bs.gpu_means = [r.mean_val for r in rows if r.mean_val > 0]
        bs.power_values = [r.power_avg_w for r in rows if r.power_avg_w > 0]
        bs.temp_values = [r.temp_max_c for r in rows if r.temp_max_c > 0]
        bs.sm_util_values = [r.sm_util_mean for r in rows if r.sm_util_mean > 0]
        bs.mem_bw_util_values = [r.mem_bw_util_mean for r in rows if r.mem_bw_util_mean > 0]
        bs.gpu_clock_values = [r.gpu_clock_mean for r in rows if r.gpu_clock_mean > 0]
        bs.throttled_results = [r for r in rows if r.throttled]

        by_node = _group_by(rows, lambda r: r.hostname)
        for host, node_rows in by_node.items():
            means = [r.mean_val for r in node_rows if r.mean_val > 0]
            if means:
                node_m = statistics.mean(means)
                bs.node_means[host] = node_m
                if len(means) > 1 and node_m > 0:
                    bs.node_intra_cv[host] = (
                        statistics.stdev(means) / node_m * 100
                    )
                else:
                    bs.node_intra_cv[host] = 0.0

        out[key] = bs

    return out


# Unit auto-scaling: GFLOP/s -> TFLOP/s, GB/s -> TB/s when values are large
_UNIT_UPSCALE = {
    "GFLOP/s": ("TFLOP/s", 1000.0),
    "GB/s":    ("TB/s",    1000.0),
    "MLUP/s":  ("GLUP/s",  1000.0),
    "Mops/s":  ("Gops/s",  1000.0),
}


def _auto_scale_units(results, bench_stats, iteration_data):
    # type: (List[GPUResult], OrderedDict, Dict[str, List[Dict]]) -> None
    """Scale performance values and units in-place when values are large enough.

    If the minimum GPU mean for a benchmark exceeds the scale threshold (1000),
    divide all performance values by the divisor and update the unit string.
    """
    for key, bs in bench_stats.items():
        if bs.unit not in _UNIT_UPSCALE:
            continue
        if not bs.gpu_means or min(bs.gpu_means) < 1000:
            continue
        new_unit, divisor = _UNIT_UPSCALE[bs.unit]
        # Scale BenchmarkStats
        bs.unit = new_unit
        bs.gpu_means = [v / divisor for v in bs.gpu_means]
        for host in bs.node_means:
            bs.node_means[host] /= divisor
        # Scale individual GPUResult rows
        for r in bs.results:
            r.mean_val /= divisor
            r.min_val /= divisor
            r.max_val /= divisor
            r.unit = new_unit
        # Scale iteration data matching this benchmark+dtype
        bench_name = key[0]
        dtype = key[1]
        for iter_key, rows in iteration_data.items():
            parts = iter_key.split(":")
            if len(parts) >= 4 and parts[2] == bench_name and parts[3] == dtype:
                for row in rows:
                    row["performance"] /= divisor


def detect_outliers(bench_stats, threshold_pct=15.0):
    # type: (OrderedDict, float) -> List[Dict]
    """Flag GPUs whose mean deviates > threshold% from fleet mean.

    Note: the interactive Plotly dashboard uses a different, sigma-based
    outlier algorithm (mean +/- N*sigma, adjustable via slider).  The two
    methods intentionally complement each other: this percentage-based
    approach provides a deterministic, config-driven threshold for CI
    exit codes, while the sigma-based approach in the dashboard allows
    interactive exploration with adjustable sensitivity.
    """
    outliers = []  # type: List[Dict]
    for key, bs in bench_stats.items():
        fleet_m = bs.fleet_mean
        if fleet_m == 0 or len(bs.gpu_means) < 2:
            continue
        for r in bs.results:
            if r.mean_val <= 0:
                continue
            dev = abs(r.mean_val - fleet_m) / fleet_m * 100
            if dev > threshold_pct:
                direction = "above" if r.mean_val > fleet_m else "below"
                severity = "good" if r.mean_val > fleet_m else "bad"
                outliers.append({
                    "host": r.hostname,
                    "gpu": r.gpu,
                    "serial": r.serial[:13] if r.serial else "",
                    "benchmark": bench_label(r),
                    "value": r.mean_val,
                    "unit": r.unit,
                    "fleet_mean": fleet_m,
                    "deviation_pct": dev,
                    "direction": direction,
                    "severity": severity,
                })
    outliers.sort(key=lambda o: -o["deviation_pct"])
    return outliers


# ---------------------------------------------------------------------------
# CLI Summary
# ---------------------------------------------------------------------------

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _use_color():
    # type: () -> bool
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _c(text, code, color):
    # type: (str, str, bool) -> str
    return "{}{}{}".format(code, text, RESET) if color else text


def print_summary(results, bench_stats, outliers, threshold,
                  system_name="", job_name="", node_map=None):
    # type: (List[GPUResult], OrderedDict, List[Dict], float, str, str, Optional[Dict[str, str]]) -> None
    """Print fleet summary to stderr."""
    color = _use_color()
    hosts = sorted(set(r.hostname for r in results))
    gpus = sorted(set((r.hostname, r.gpu) for r in results))
    models = sorted(set(r.gpu_model for r in results if r.gpu_model))
    benchmarks = list(bench_stats.keys())

    out = sys.stderr

    print(file=out)
    print("=" * 100, file=out)
    header = "TORCH HAMMER -- FLEET REPORT"
    if system_name or job_name:
        parts = [p for p in [system_name, job_name] if p]
        header += " -- " + " -- ".join(parts)
    print(_c(header, BOLD, color), file=out)
    print("=" * 100, file=out)
    host_list = ", ".join(hosts[:10])
    if len(hosts) > 10:
        host_list += "..."
    print("  Nodes:      {} ({})".format(len(hosts), host_list), file=out)
    print("  GPU Model:  {}".format(", ".join(models[:3]) if models else "--"), file=out)
    print("  GPUs:       {} total".format(len(gpus)), file=out)
    print("  Benchmarks: {}".format(len(benchmarks)), file=out)
    print("  Rows:       {}".format(len(results)), file=out)
    print("  Threshold:  {}%".format(threshold), file=out)
    print(file=out)

    # Per-benchmark table
    hdr = "{:<40} {:<10} {:>12} {:>7} {:>8} {:>7} {:>5} {:>5}".format(
        "Benchmark", "Unit", "Fleet Mean", "CV%", "Power", "Temp", "N", "Thrt")
    print(hdr, file=out)
    print("-" * len(hdr), file=out)

    for key in benchmarks:
        bs = bench_stats[key]
        label = bench_label_from_key(key)
        n = len(bs.gpu_means)
        cv_s = "{:.1f}%".format(bs.fleet_cv) if n > 1 else "--"
        if color and n > 1 and bs.fleet_cv > 5:
            cv_s = _c("{:.1f}%".format(bs.fleet_cv), RED, color)
        avg_p = "{:.0f}W".format(statistics.mean(bs.power_values)) if bs.power_values else "--"
        max_t = "{:.0f}C".format(max(bs.temp_values)) if bs.temp_values else "--"
        n_thrt = len(bs.throttled_results)
        thrt_s = str(n_thrt) if n_thrt > 0 else "--"
        if color and n_thrt > 0:
            thrt_s = _c(str(n_thrt), RED, color)

        print("{:<40} {:<10} {:>12.1f} {:>7} {:>8} {:>7} {:>5} {:>5}".format(
            label, bs.unit, bs.fleet_mean, cv_s, avg_p, max_t, n, thrt_s), file=out)

    print(file=out)

    # Per-node health (only with multiple nodes)
    if len(hosts) > 1:
        node_groups = _group_by(results, lambda r: r.hostname)
        has_loc = bool(node_map)
        if has_loc:
            hdr2 = "{:<20} {:<16} {:>5} {:>6} {:>10} {:>9} {:>8}".format(
                "Node", "Location", "GPUs", "Tests", "Mean Power", "Max Temp", "Status")
        else:
            hdr2 = "{:<20} {:>5} {:>6} {:>10} {:>9} {:>8}".format(
                "Node", "GPUs", "Tests", "Mean Power", "Max Temp", "Status")
        print(hdr2, file=out)
        print("-" * len(hdr2), file=out)

        bad_outliers_per_host = {}  # type: Dict[str, int]
        for o in outliers:
            if o["severity"] == "bad":
                bad_outliers_per_host[o["host"]] = (
                    bad_outliers_per_host.get(o["host"], 0) + 1
                )

        # Collect throttled hosts
        throttled_per_host = {}  # type: Dict[str, int]
        for r in results:
            if r.throttled:
                throttled_per_host[r.hostname] = (
                    throttled_per_host.get(r.hostname, 0) + 1
                )

        # Classify hosts and build display rows
        _CLI_NODE_TRUNCATE = 30
        warn_hosts = []  # type: list
        pass_hosts = []  # type: list
        host_rows = {}  # type: dict

        for host in sorted(hosts):
            hrows = node_groups.get(host, [])
            n_gpu = len(set(r.gpu for r in hrows))
            n_tests = len(hrows)
            powers = [r.power_avg_w for r in hrows if r.power_avg_w > 0]
            temps = [r.temp_max_c for r in hrows if r.temp_max_c > 0]
            avg_p = "{:.0f}W".format(statistics.mean(powers)) if powers else "--"
            max_t = "{:.0f}C".format(max(temps)) if temps else "--"
            n_bad = bad_outliers_per_host.get(host, 0)
            n_thrt = throttled_per_host.get(host, 0)
            if n_bad > 0 and n_thrt > 0:
                status = _c("WARN+THRT", YELLOW, color)
                warn_hosts.append(host)
            elif n_bad > 0:
                status = _c("WARN", YELLOW, color)
                warn_hosts.append(host)
            elif n_thrt > 0:
                status = _c("THRT", YELLOW, color)
                warn_hosts.append(host)
            else:
                status = _c("PASS", GREEN, color)
                pass_hosts.append(host)
            if has_loc:
                host_rows[host] = (host, node_map.get(host, "--"), n_gpu, n_tests, avg_p, max_t, status)
            else:
                host_rows[host] = (host, n_gpu, n_tests, avg_p, max_t, status)

        def _print_host_row(h):
            # type: (str) -> None
            r = host_rows[h]
            if has_loc:
                print("{:<20} {:<16} {:>5} {:>6} {:>10} {:>9} {:>8}".format(*r), file=out)
            else:
                print("{:<20} {:>5} {:>6} {:>10} {:>9} {:>8}".format(*r), file=out)

        if len(hosts) <= _CLI_NODE_TRUNCATE:
            for host in sorted(hosts):
                _print_host_row(host)
        else:
            # Truncated: first 5 PASS, separator, last 5 PASS, then all WARN
            head = pass_hosts[:5]
            tail = pass_hosts[-5:] if len(pass_hosts) > 10 else pass_hosts[5:]
            for h in head:
                _print_host_row(h)
            omitted = len(pass_hosts) - len(head) - len(tail)
            if omitted > 0:
                print("  ... {} PASS nodes omitted ...".format(omitted), file=out)
            for h in tail:
                _print_host_row(h)
            for h in warn_hosts:
                _print_host_row(h)

        print(file=out)

    # Outliers
    bad = [o for o in outliers if o["severity"] == "bad"]
    if bad:
        print(_c("OUTLIERS ({} below threshold)".format(len(bad)), BOLD, color), file=out)
        print("-" * 90, file=out)
        for o in bad[:20]:
            loc_str = ""
            if node_map and o["host"] in node_map:
                loc_str = " [{}]".format(node_map[o["host"]])
            print("  {} {}{}:GPU{} -- {}: {:.1f} {} ({:.1f}% {} fleet)".format(
                _c("x", RED, color), o["host"], loc_str, o["gpu"],
                o["benchmark"], o["value"], o["unit"],
                o["deviation_pct"], o["direction"]), file=out)
        if len(bad) > 20:
            print("  ... and {} more".format(len(bad) - 20), file=out)
        print(file=out)
    else:
        print(_c("  No outliers detected", GREEN, color), file=out)
        print(file=out)

    # Location cluster summary for outliers
    if node_map and bad:
        loc_counts = {}  # type: Dict[str, int]
        for o in bad:
            loc = node_map.get(o["host"], "")
            if loc:
                loc_counts[loc] = loc_counts.get(loc, 0) + 1
        multi = {l: c for l, c in loc_counts.items() if c > 1}
        if multi:
            print(_c("LOCATION CLUSTERS", BOLD, color), file=out)
            for loc, cnt in sorted(multi.items(), key=lambda x: -x[1]):
                print("  {} -- {} outlier GPU(s)".format(loc, cnt), file=out)
            print(file=out)

    # Thermal throttling summary
    all_throttled = [r for r in results if r.throttled]
    if all_throttled:
        throttled_hosts = sorted(set(r.hostname for r in all_throttled))
        throttled_gpus = len(all_throttled)
        print(_c("THERMAL THROTTLING ({} GPU(s) on {} node(s))".format(
            throttled_gpus, len(throttled_hosts)), BOLD, color), file=out)
        print("-" * 90, file=out)
        for r in all_throttled[:20]:
            loc_str = ""
            if node_map and r.hostname in node_map:
                loc_str = " [{}]".format(node_map[r.hostname])
            samples_str = ""
            if r.throttle_samples > 0:
                samples_str = " ({} samples)".format(r.throttle_samples)
            print("  {} {}{}:GPU{} -- {}{}".format(
                _c("!", YELLOW, color), r.hostname, loc_str, r.gpu,
                bench_label(r), samples_str), file=out)
        if len(all_throttled) > 20:
            print("  ... and {} more".format(len(all_throttled) - 20), file=out)
        print(file=out)

    # Verdict
    n_outlier_hosts = len(set(o["host"] for o in bad))
    n_throttled_hosts = len(set(r.hostname for r in all_throttled)) if all_throttled else 0
    n_warn_hosts = len(set(
        list(o["host"] for o in bad) +
        [r.hostname for r in all_throttled]
    ))
    n_pass = len(hosts) - n_warn_hosts
    parts = ["{}/{} PASS".format(n_pass, len(hosts))]
    if n_outlier_hosts > 0:
        parts.append("{} OUTLIER".format(n_outlier_hosts))
    if n_throttled_hosts > 0:
        parts.append("{} THROTTLED".format(n_throttled_hosts))
    if n_outlier_hosts == 0 and n_throttled_hosts == 0:
        parts.append("0 WARN")
    verdict = "FLEET VERDICT: " + " | ".join(parts)
    print(verdict, file=out)
    print("=" * 100, file=out)
    print(file=out)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _esc(s):
    # type: (object) -> str
    """HTML-escape any value."""
    return html.escape(str(s))


def _fmt_html(v, decimals=0):
    # type: (float, int) -> str
    if decimals == 0:
        return "{:,.0f}".format(v)
    return "{:,.{}f}".format(v, decimals)


SCALE_THRESHOLD = 50  # Switch to histogram + truncated table above this many nodes


def _compute_node_summaries(bs, by_node, nodes, gpu_indices, node_map=None):
    # type: (BenchmarkStats, dict, list, list, Optional[Dict[str, str]]) -> list
    """Compute per-node summary dicts for table rendering."""
    fleet_m = bs.fleet_mean
    summaries = []
    for node in nodes:
        node_rows = by_node[node]
        gpu_vals = {}  # type: dict
        means = []
        for gpu_idx in gpu_indices:
            gpu_row = next((r for r in node_rows if r.gpu == gpu_idx), None)
            if gpu_row:
                means.append(gpu_row.mean_val)
                gpu_vals[gpu_idx] = gpu_row.mean_val
            else:
                gpu_vals[gpu_idx] = None
        node_avg = statistics.mean(means) if means else 0
        if fleet_m > 0 and node_avg > 0:
            vs_fleet = (node_avg - fleet_m) / fleet_m * 100
        else:
            vs_fleet = 0.0
        s = {
            "node": node,
            "gpu_vals": gpu_vals,
            "node_avg": node_avg,
            "vs_fleet": vs_fleet,
            "throttled": any(r.throttled for r in node_rows),
        }  # type: dict
        if node_map:
            s["location"] = node_map.get(node, "")
        summaries.append(s)
    return summaries


def _render_table_row(s, gpu_indices, outlier_hosts, threshold, has_location=False):
    # type: (dict, list, set, float, bool) -> str
    """Render one <tr> from a node summary dict with data-sort attributes."""
    is_outlier = s["node"] in outlier_hosts
    host_cls = ' class="mono host warn"' if is_outlier else ' class="mono host"'

    loc_cell = ""
    if has_location:
        loc = s.get("location", "")
        loc_cell = '<td class="mono" data-sort="{sv}">{v}</td>'.format(
            sv=_esc(loc), v=_esc(loc) if loc else "--")

    gpu_cells = ""
    for gpu_idx in gpu_indices:
        v = s["gpu_vals"].get(gpu_idx)
        if v is not None:
            gpu_cells += '<td class="mono" data-sort="{:.2f}">{}</td>'.format(
                v, _esc(_fmt_html(v)))
        else:
            gpu_cells += '<td class="mono" data-sort="0">--</td>'

    vs = s["vs_fleet"]
    sign = "+" if vs >= 0 else ""
    vs_cls = ' warn' if vs < -threshold else ''
    vs_text = "{}{:.1f}%".format(sign, vs)

    thrt = s.get("throttled", False)
    thrt_cell = (
        '<td class="mono warn" data-sort="1" '
        'title="Thermal throttling detected">\u26a0</td>'
        if thrt else
        '<td class="mono" data-sort="0"></td>'
    )

    return (
        '<tr>'
        '<td{h}>{node}</td>'
        '{loc}'
        '{gpu}'
        '<td class="mono" data-sort="{avg:.2f}">{avg_fmt}</td>'
        '<td class="mono{vcls}" data-sort="{vs:.4f}">{vtxt}</td>'
        '{thrt}'
        '</tr>'
    ).format(
        h=host_cls, node=_esc(s["node"]),
        loc=loc_cell,
        gpu=gpu_cells,
        avg=s["node_avg"], avg_fmt=_esc(_fmt_html(s["node_avg"])),
        vcls=vs_cls, vs=vs, vtxt=vs_text,
        thrt=thrt_cell,
    )


def _build_table_html(summaries, gpu_indices, outlier_hosts, threshold, truncate,
                      bench_id=0, has_location=False):
    # type: (list, list, set, float, bool, int, bool) -> str
    """Build <table> body rows, optionally truncated with a separator."""
    if not truncate or len(summaries) <= SCALE_THRESHOLD:
        # Full table
        return "\n".join(
            _render_table_row(s, gpu_indices, outlier_hosts, threshold, has_location)
            for s in sorted(summaries, key=lambda s: s["node"])
        )

    # Truncated: bottom 5, outliers, top 5 (by vs_fleet)
    by_vs = sorted(summaries, key=lambda s: s["vs_fleet"])
    bottom_5 = by_vs[:5]
    top_5 = by_vs[-5:]

    # Outlier nodes not already in top/bottom
    shown_nodes = {s["node"] for s in bottom_5 + top_5}
    outlier_rows = [s for s in summaries
                    if s["node"] in outlier_hosts and s["node"] not in shown_nodes]
    outlier_rows.sort(key=lambda s: s["vs_fleet"])

    # Hidden rows for "Show all" toggle
    hidden_nodes = {s["node"] for s in summaries} - shown_nodes - {s["node"] for s in outlier_rows}
    hidden_rows = sorted(
        [s for s in summaries if s["node"] in hidden_nodes],
        key=lambda s: s["vs_fleet"])
    n_hidden = len(hidden_rows)
    n_cols = 4 + len(gpu_indices) + (1 if has_location else 0)  # node [+ loc] + GPUs + avg + vs fleet + thrt
    toggle_id = "hidden_{}".format(bench_id)

    rows = []
    # Section: lowest performing
    rows.append(
        '<tr class="section-label"><td colspan="{}" style="text-align:left;'
        'color:var(--muted);font-size:11px;font-weight:600;padding:6px 12px;'
        'text-transform:uppercase;letter-spacing:0.04em;">'
        'Lowest performing</td></tr>'.format(n_cols)
    )
    for s in bottom_5:
        rows.append(_render_table_row(s, gpu_indices, outlier_hosts, threshold, has_location))

    # Separator with "Show all" toggle
    if n_hidden > 0:
        rows.append(
            '<tr class="separator"><td colspan="{cols}" style="text-align:center;'
            'color:var(--muted);font-style:italic;padding:10px;">'
            '... {n} nodes within threshold ...'
            ' <button class="show-all-btn" onclick="'
            "var el=document.getElementById('{tid}');var b=this;"
            "if(el.style.display==='none'){{el.style.display='';b.textContent='Hide';}}"
            "else{{el.style.display='none';b.textContent='Show all';}}"
            '">Show all</button>'
            '</td></tr>'.format(cols=n_cols, n=n_hidden, tid=toggle_id)
        )
        # Hidden rows (collapsed by default)
        rows.append(
            '<tr id="{tid}" style="display:none"><td colspan="{cols}" '
            'style="padding:0">'
            '<table class="inner-expand" style="width:100%;border-collapse:collapse">'.format(
                tid=toggle_id, cols=n_cols)
        )
        for s in hidden_rows:
            rows.append(_render_table_row(s, gpu_indices, outlier_hosts, threshold, has_location))
        rows.append('</table></td></tr>')

    # Outlier nodes
    if outlier_rows:
        rows.append(
            '<tr class="section-label"><td colspan="{}" style="text-align:left;'
            'color:var(--warn);font-size:11px;font-weight:600;padding:6px 12px;'
            'text-transform:uppercase;letter-spacing:0.04em;">'
            'Outliers</td></tr>'.format(n_cols)
        )
        for s in outlier_rows:
            rows.append(_render_table_row(s, gpu_indices, outlier_hosts, threshold, has_location))

    # Section: highest performing
    rows.append(
        '<tr class="section-label"><td colspan="{}" style="text-align:left;'
        'color:var(--muted);font-size:11px;font-weight:600;padding:6px 12px;'
        'text-transform:uppercase;letter-spacing:0.04em;">'
        'Highest performing</td></tr>'.format(n_cols)
    )
    for s in top_5:
        rows.append(_render_table_row(s, gpu_indices, outlier_hosts, threshold, has_location))

    return "\n".join(rows)


def _fmt_tick(v):
    # type: (float) -> str
    """Format a numeric tick label concisely."""
    av = abs(v)
    if av >= 1_000_000:
        return "{:.1f}M".format(v / 1_000_000)
    if av >= 10_000:
        return "{:.0f}k".format(v / 1_000)
    if av >= 100:
        return "{:,.0f}".format(v)
    if av >= 1:
        return "{:.1f}".format(v)
    return "{:.2f}".format(v)


def _nice_ticks(lo, hi, n=5):
    # type: (float, float, int) -> list
    """Return a list of 'nice' tick values spanning lo..hi."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    nice_steps = [1, 2, 2.5, 5, 10]
    step = mag
    for s in nice_steps:
        if s * mag >= raw:
            step = s * mag
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.01:
        if v >= lo - step * 0.01:
            ticks.append(round(v, 10))
        v += step
    return ticks if ticks else [lo, hi]


# ---------------------------------------------------------------------------
# SVG chart generators
# ---------------------------------------------------------------------------

_SVG_W = 960
_SVG_H = 320
# T24: small multiples get their own viewBox. The three-up .chart-row cell is
# ~350 px wide on a 1200 px page, so a 360-unit box renders ~1 unit per px
# and 11-unit text stays at the T23 legibility floor (a 960-unit box scaled
# into the same cell renders 11-unit text at ~4 px).
_SVG_W_SM = 360
_SVG_H_SM = 220  # h ~ 0.62 w
_SVG_H_STRIP = 260  # shorter full-width strip used by _svg_single_bar
_ML = 72   # margin left
_MR = 24   # margin right
_MT = 24   # margin top
_MB = 60   # margin bottom
# T25: what a compact chart may carry
_COMPACT_MAX_X_LABELS = 8
_COMPACT_Y_TICKS = 4
_COMPACT_NOTE_FONT = 11  # threshold / mean annotations; 10 in full charts


@dataclass
class _ChartGeometry:
    """ViewBox and margins for one SVG chart.

    ``compact`` charts (T24/T25) shrink margins with the box so the plot area
    keeps ~80% of the width, and drop legends and per-point labels.
    """
    w: int
    h: int
    ml: int
    mr: int
    mt: int
    mb: int
    compact: bool = False

    @property
    def plot_w(self):
        # type: () -> int
        return self.w - self.ml - self.mr

    @property
    def plot_h(self):
        # type: () -> int
        return self.h - self.mt - self.mb

    @property
    def note_font(self):
        # type: () -> int
        return _COMPACT_NOTE_FONT if self.compact else 10

    @property
    def y_title_x(self):
        # type: () -> int
        """Centre x of the rotated y-axis title, clear of the tick labels."""
        return 10 if self.compact else 16


def _chart_geometry(compact=False, h=_SVG_H):
    # type: (bool, int) -> _ChartGeometry
    if compact:
        return _ChartGeometry(w=_SVG_W_SM, h=_SVG_H_SM, ml=52, mr=10, mt=18, mb=36,
                              compact=True)
    return _ChartGeometry(w=_SVG_W, h=h, ml=_ML, mr=_MR, mt=_MT, mb=_MB)


def _label_step(n_items, max_labels):
    # type: (int, int) -> int
    """Every k-th item gets a label so that at most ``max_labels`` are drawn."""
    return max(1, -(-n_items // max_labels))


def _svg_open(w, h):
    # type: (int, int) -> str
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'role="img" style="width:100%;height:auto;display:block;'
        'font-family:system-ui,-apple-system,sans-serif">'
    ).format(w=w, h=h)


def _svg_bar_chart(nodes, gpu_indices, by_node, unit, y_min, y_max, fleet_mean=None,
                   compact=False):
    # type: (list, list, dict, str, float, float, ..., bool) -> str
    """Grouped bar chart as inline SVG for <=50 nodes."""
    g = _chart_geometry(compact)
    w, h = g.w, g.h
    ml, mr, mt, mb = g.ml, g.mr, g.mt, g.mb
    plot_w, plot_h = g.plot_w, g.plot_h

    parts = [_svg_open(w, h)]

    # Y-axis ticks + gridlines
    ticks = _nice_ticks(y_min, y_max, _COMPACT_Y_TICKS if compact else 5)
    for tv in ticks:
        if y_max > y_min:
            yp = mt + plot_h - (tv - y_min) / (y_max - y_min) * plot_h
        else:
            yp = mt + plot_h / 2
        parts.append(
            '<line x1="{ml}" y1="{y}" x2="{xr}" y2="{y}" '
            'stroke="rgba(128,128,128,0.12)" stroke-dasharray="2,4"/>'.format(
                ml=ml, xr=w - mr, y=round(yp, 1)))
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="end" '
            'font-size="12" fill="var(--muted,#6b6b65)">{lbl}</text>'.format(
                x=ml - 6, y=round(yp + 3, 1), lbl=_esc(_fmt_tick(tv))))

    # Y-axis title
    parts.append(
        '<text x="{tx}" y="{cy}" text-anchor="middle" font-size="13" '
        'fill="var(--muted,#6b6b65)" transform="rotate(-90,{tx},{cy})">{u}</text>'.format(
            tx=g.y_title_x, cy=round(mt + plot_h / 2), u=_esc(unit)))

    # Bars
    n_nodes = len(nodes)
    n_gpus = len(gpu_indices)
    if n_nodes == 0 or n_gpus == 0:
        parts.append('</svg>')
        return "".join(parts)

    group_w = plot_w / n_nodes
    bar_w = group_w / (n_gpus + 0.5)
    gap = bar_w * 0.25

    for ni, node in enumerate(nodes):
        gx = ml + ni * group_w
        for gi, gpu_idx in enumerate(gpu_indices):
            gpu_row = next((r for r in by_node[node] if r.gpu == gpu_idx), None)
            if not gpu_row:
                continue
            val = gpu_row.mean_val
            if val <= 0 or y_max <= y_min:
                continue
            bar_h = (val - y_min) / (y_max - y_min) * plot_h
            bx = gx + gi * bar_w + gap
            by = mt + plot_h - bar_h
            color = OKABE_ITO[gi % len(OKABE_ITO)]
            parts.append(
                '<rect x="{x}" y="{y}" width="{bw}" height="{bh}" '
                'fill="{c}" rx="2">'
                '<title>GPU {gpu}: {val} {unit}</title>'
                '</rect>'.format(
                    x=round(bx, 1), y=round(by, 1),
                    bw=round(bar_w - gap * 2, 1), bh=round(bar_h, 1),
                    c=color, gpu=gpu_idx,
                    val=_esc(_fmt_html(val)), unit=_esc(unit)))

    # Fleet mean reference line
    if fleet_mean is not None and y_max > y_min and y_min <= fleet_mean <= y_max:
        my = mt + plot_h - (fleet_mean - y_min) / (y_max - y_min) * plot_h
        parts.append(
            '<line x1="{ml}" y1="{y}" x2="{xr}" y2="{y}" '
            'stroke="var(--danger,#f09595)" stroke-width="1.5" '
            'stroke-dasharray="6,4" opacity="0.8"/>'.format(
                ml=ml, xr=w - mr, y=round(my, 1)))
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="end" '
            'font-size="11" fill="var(--danger,#f09595)" opacity="0.8">'
            'fleet mean</text>'.format(
                x=w - mr, y=round(my - 4, 1)))

    # X-axis labels (T25: a compact chart labels at most every k-th node)
    rotate = n_nodes > 10
    step = _label_step(n_nodes, _COMPACT_MAX_X_LABELS) if compact else 1
    for ni, node in enumerate(nodes):
        if ni % step:
            continue
        cx = ml + ni * group_w + group_w / 2
        if rotate:
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="end" font-size="11" '
                'fill="var(--muted,#6b6b65)" '
                'transform="rotate(-45,{x},{y})">{n}</text>'.format(
                    x=round(cx, 1), y=h - mb + 14, n=_esc(node)))
        else:
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" font-size="12" '
                'fill="var(--muted,#6b6b65)">{n}</text>'.format(
                    x=round(cx, 1), y=h - mb + 16, n=_esc(node)))

    if not compact:  # T25: a compact chart has no legend
        _svg_gpu_legend(parts, gpu_indices, w - mr, mt + 4)

    parts.append('</svg>')
    return "\n".join(parts)


def _svg_gpu_legend(parts, gpu_indices, right_x, top_y):
    # type: (list, list, int, int) -> None
    """Append the per-GPU colour legend, right-aligned at ``right_x``."""
    n_gpus = len(gpu_indices)
    for gi, gpu_idx in enumerate(gpu_indices):
        lx = right_x - (n_gpus - gi) * 80
        color = OKABE_ITO[gi % len(OKABE_ITO)]
        parts.append(
            '<rect x="{x}" y="{y}" width="12" height="12" fill="{c}" rx="2"/>'.format(
                x=lx, y=top_y, c=color))
        parts.append(
            '<text x="{x}" y="{y}" font-size="12" '
            'fill="var(--muted,#6b6b65)">GPU {gpu}</text>'.format(
                x=lx + 16, y=top_y + 10, gpu=gpu_idx))


def _svg_histogram(vals, unit, mean_val, n_bins=40, threshold_pct=None, compact=False):
    # type: (list, str, float, int, ..., bool) -> str
    """Histogram distribution as inline SVG."""
    g = _chart_geometry(compact)
    w, h = g.w, g.h
    ml, mr, mt, mb = g.ml, g.mr, g.mt, g.mb
    plot_w, plot_h = g.plot_w, g.plot_h

    if not vals:
        return ""

    # Auto-adjust bins for small data
    if len(vals) < 10:
        n_bins = len(vals)
    elif len(vals) < 50:
        n_bins = max(10, len(vals) // 3)

    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        hi = lo + 1
    bin_width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for v in vals:
        idx = int((v - lo) / bin_width)
        if idx >= n_bins:
            idx = n_bins - 1
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    if max_count == 0:
        max_count = 1

    # Outlier threshold boundaries
    lo_bound = None
    hi_bound = None
    if threshold_pct is not None:
        lo_bound = mean_val * (1 - threshold_pct / 100)
        hi_bound = mean_val * (1 + threshold_pct / 100)

    parts = [_svg_open(w, h)]

    bar_w = plot_w / n_bins

    # Y-axis ticks (T25: fewer in a compact chart)
    y_ticks = _nice_ticks(0, max_count, _COMPACT_Y_TICKS if compact else 5)
    for tv in y_ticks:
        yp = mt + plot_h - tv / max_count * plot_h
        parts.append(
            '<line x1="{ml}" y1="{y}" x2="{xr}" y2="{y}" '
            'stroke="rgba(128,128,128,0.12)" stroke-dasharray="2,4"/>'.format(
                ml=ml, xr=w - mr, y=round(yp, 1)))
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="end" '
            'font-size="12" fill="var(--muted,#6b6b65)">{lbl}</text>'.format(
                x=ml - 6, y=round(yp + 3, 1), lbl=int(tv)))

    # Y-axis title
    parts.append(
        '<text x="{tx}" y="{cy}" text-anchor="middle" font-size="13" '
        'fill="var(--muted,#6b6b65)" transform="rotate(-90,{tx},{cy})">GPU count</text>'.format(
            tx=g.y_title_x, cy=round(mt + plot_h / 2)))

    # Bars
    for i in range(n_bins):
        count = bins[i]
        if count == 0:
            continue
        bx = ml + i * bar_w
        bar_h = count / max_count * plot_h
        by = mt + plot_h - bar_h
        bin_lo = lo + i * bin_width
        bin_hi = bin_lo + bin_width
        bin_center = bin_lo + bin_width / 2
        # Color: vermillion for bins below lo_bound, blue otherwise
        if lo_bound is not None and bin_center < lo_bound:
            color = "#D55E00"
        else:
            color = "#0072B2"
        parts.append(
            '<rect x="{x}" y="{y}" width="{bw}" height="{bh}" '
            'fill="{c}" rx="1">'
            '<title>{blo}\u2013{bhi} {unit}: {cnt} GPUs</title>'
            '</rect>'.format(
                x=round(bx, 1), y=round(by, 1),
                bw=round(bar_w - 1, 1), bh=round(bar_h, 1),
                c=color,
                blo=_esc(_fmt_tick(bin_lo)), bhi=_esc(_fmt_tick(bin_hi)),
                unit=_esc(unit), cnt=count))

    # X-axis labels (spread across range; T25 caps a compact chart at 8)
    if compact:
        step = _label_step(n_bins, _COMPACT_MAX_X_LABELS)
    else:
        step = max(1, n_bins // min(10, n_bins))
    for i in range(0, n_bins, step):
        cx = ml + i * bar_w + bar_w / 2
        val = lo + i * bin_width + bin_width / 2
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="middle" font-size="11" '
            'fill="var(--muted,#6b6b65)">{lbl}</text>'.format(
                x=round(cx, 1), y=h - mb + 16, lbl=_esc(_fmt_tick(val))))

    # X-axis title
    parts.append(
        '<text x="{cx}" y="{y}" text-anchor="middle" font-size="13" '
        'fill="var(--muted,#6b6b65)">{u}</text>'.format(
            cx=round(ml + plot_w / 2), y=h - 4, u=_esc(unit)))

    # Threshold lines (drawn on top of bars)
    if threshold_pct is not None:
        if lo < lo_bound < hi:
            lx = ml + (lo_bound - lo) / (hi - lo) * plot_w
            parts.append(
                '<line x1="{x}" y1="{mt}" x2="{x}" y2="{mb}" '
                'stroke="var(--danger,#f09595)" stroke-width="1" '
                'stroke-dasharray="4,3" opacity="0.7"/>'.format(
                    x=round(lx, 1), mt=mt, mb=mt + plot_h))
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" '
                'font-size="{fs}" fill="var(--danger,#f09595)" opacity="0.7">'
                '-{t}%</text>'.format(
                    x=round(lx, 1), y=mt - 4, fs=g.note_font, t=int(threshold_pct)))
        if lo < hi_bound < hi:
            hx = ml + (hi_bound - lo) / (hi - lo) * plot_w
            parts.append(
                '<line x1="{x}" y1="{mt}" x2="{x}" y2="{mb}" '
                'stroke="var(--danger,#f09595)" stroke-width="1" '
                'stroke-dasharray="4,3" opacity="0.7"/>'.format(
                    x=round(hx, 1), mt=mt, mb=mt + plot_h))
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" '
                'font-size="{fs}" fill="var(--danger,#f09595)" opacity="0.7">'
                '+{t}%</text>'.format(
                    x=round(hx, 1), y=mt - 4, fs=g.note_font, t=int(threshold_pct)))

    # Fleet mean vertical line
    if lo <= mean_val <= hi:
        mx = ml + (mean_val - lo) / (hi - lo) * plot_w
        parts.append(
            '<line x1="{x}" y1="{mt}" x2="{x}" y2="{mb}" '
            'stroke="var(--muted,#9a9a92)" stroke-width="1.5" '
            'opacity="0.8"/>'.format(
                x=round(mx, 1), mt=mt, mb=mt + plot_h))
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="middle" '
            'font-size="{fs}" fill="var(--muted,#9a9a92)" opacity="0.8">'
            'mean</text>'.format(
                x=round(mx, 1), y=mt - 4, fs=g.note_font))

    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dot plot (sigma-band coloring)
# ---------------------------------------------------------------------------

SIGMA_COLORS = [
    ("#0072B2", "within 1\u03c3"),    # blue - normal
    ("#E69F00", "1\u03c3 to 2\u03c3"),  # orange - attention
    ("#D55E00", "2\u03c3 to 3\u03c3"),  # vermillion - concern
    ("#000000", "beyond 3\u03c3"),     # black - broken
]


def _sigma_band(val, mean, stdev):
    # type: (float, float, float) -> Tuple[str, str]
    """Return (color, label) for a value's sigma distance from mean."""
    if stdev <= 0:
        return SIGMA_COLORS[0]
    z = abs(val - mean) / stdev
    if z < 1:
        return SIGMA_COLORS[0]
    elif z < 2:
        return SIGMA_COLORS[1]
    elif z < 3:
        return SIGMA_COLORS[2]
    else:
        return SIGMA_COLORS[3]


def _svg_dot_plot(vals, unit, mean_val, n_bins=40, threshold_pct=None,
                  results=None, compact=False):
    # type: (list, str, float, int, ..., ..., bool) -> str
    """Rug-density plot: KDE density curve above, sigma-colored rug ticks below.

    Top zone (~70% of plot height): filled density curve showing fleet
    distribution shape.  Bottom zone (~30%): rug plot where every GPU is a
    vertical tick mark colored by sigma band.  Outliers beyond 2 sigma get
    taller ticks; beyond 3 sigma get text labels (full size only, T25).
    """
    g = _chart_geometry(compact)
    w, h = g.w, g.h
    ml, mr, mt, mb = g.ml, g.mr, g.mt, g.mb
    plot_w, plot_h = g.plot_w, g.plot_h

    if not vals:
        return ""

    n = len(vals)
    mean = statistics.mean(vals) if vals else 0
    stdev = statistics.stdev(vals) if n > 1 else 0.0

    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        hi = lo + 1

    # ----- Layout zones -----
    rug_frac = 0.28
    density_h = plot_h * (1 - rug_frac)      # top zone for density
    rug_h = plot_h * rug_frac                 # bottom zone for rug
    rug_top = mt + density_h                 # y where rug zone starts
    rug_base = mt + plot_h                   # bottom of rug zone (baseline)

    def x_pos(v):
        """Map a data value to pixel x-coordinate."""
        return ml + (v - lo) / (hi - lo) * plot_w

    # ----- KDE (Gaussian kernel density estimate) -----
    # Silverman bandwidth; evaluated over n_bins sample points for the curve.
    kde_points = max(n_bins, 80)
    if stdev > 0 and n > 1:
        bw = 1.06 * stdev * n ** (-1.0 / 5)
        bw = max(bw, (hi - lo) / 200)  # floor to avoid infinitely narrow KDE
    else:
        bw = (hi - lo) / 10 if hi > lo else 1.0

    xs_kde = [lo + i * (hi - lo) / (kde_points - 1) for i in range(kde_points)]
    density = []
    inv_bw = 1.0 / bw
    for xk in xs_kde:
        s = 0.0
        for v in vals:
            z = (xk - v) * inv_bw
            s += math.exp(-0.5 * z * z)
        density.append(s * inv_bw / (n * math.sqrt(2 * math.pi)))

    max_density = max(density) if density else 1.0
    if max_density <= 0:
        max_density = 1.0

    parts = [_svg_open(w, h)]

    # ----- Density curve (filled area) -----
    # Build SVG polygon: bottom-left -> curve points -> bottom-right
    poly_points = []
    poly_points.append("{},{}" .format(round(x_pos(xs_kde[0]), 1),
                                       round(rug_top, 1)))
    for i, xk in enumerate(xs_kde):
        yp = rug_top - (density[i] / max_density) * density_h
        poly_points.append("{},{}".format(round(x_pos(xk), 1), round(yp, 1)))
    poly_points.append("{},{}".format(round(x_pos(xs_kde[-1]), 1),
                                      round(rug_top, 1)))
    parts.append(
        '<polygon points="{pts}" fill="#0072B2" opacity="0.30" '
        'stroke="#0072B2" stroke-width="1.8" stroke-opacity="0.85"/>'.format(
            pts=" ".join(poly_points)))

    # ----- Rug ticks -----
    # Adaptive tick width: thinner for large fleets to avoid solid ink blocks
    tick_w = max(0.3, min(2.0, plot_w / (n * 1.5)))
    normal_h = rug_h * 0.55    # body tick height
    outlier_h = rug_h * 0.85   # 2-3 sigma tick height
    extreme_h = rug_h * 1.0    # >3 sigma tick height

    # Collect labels for extreme outliers (>2 sigma) to render after all ticks
    outlier_labels = []

    for idx, v in enumerate(vals):
        px = x_pos(v)
        color, _ = _sigma_band(v, mean, stdev)

        # Tick height and opacity depend on sigma distance
        if stdev > 0:
            z = abs(v - mean) / stdev
        else:
            z = 0.0

        if z >= 3:
            th = extreme_h
            opacity = 0.95
        elif z >= 2:
            th = outlier_h
            opacity = 0.90
        else:
            th = normal_h
            opacity = 0.7

        ty = rug_base - th

        # Tooltip
        res = results[idx] if results and idx < len(results) else None
        if res:
            title = '{} GPU{}: {} {}'.format(
                _esc(res.hostname), res.gpu,
                _esc(_fmt_html(v)), _esc(unit))
        else:
            title = '{} {}'.format(_esc(_fmt_html(v)), _esc(unit))

        parts.append(
            '<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
            'stroke="{c}" stroke-width="{sw}" opacity="{op}">'
            '<title>{t}</title>'
            '</line>'.format(
                x=round(px, 1), y1=round(ty, 1), y2=round(rug_base, 1),
                c=color, sw=round(tick_w, 2), op=opacity, t=title))

        # Collect >3 sigma outliers for labeling (if results available)
        if z >= 3 and res:
            outlier_labels.append((px, res.hostname, res.gpu, v))

    # ----- Outlier labels (>3 sigma, deduplicated by position) -----
    # Limit labels to avoid clutter; pick the most extreme values.
    # T25: a compact chart carries no per-point labels at all.
    outlier_labels.sort(key=lambda t: abs(t[3] - mean), reverse=True)
    max_labels = 0 if compact else min(12, len(outlier_labels))
    placed_labels = []
    min_label_gap = 50  # minimum px between label anchors
    for px, hostname, gpu, v in outlier_labels[:max_labels]:
        too_close = any(abs(px - prev_x) < min_label_gap
                        for prev_x in placed_labels)
        if too_close:
            continue
        placed_labels.append(px)
        lbl = "{}:{}".format(_esc(hostname), gpu)
        # Place label just above the rug zone
        ly = rug_top - 4
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="middle" font-size="8" '
            'fill="var(--muted,#6b6b65)" opacity="0.9">{lbl}</text>'.format(
                x=round(px, 1), y=round(ly, 1), lbl=lbl))

    # ----- Separator line between density and rug zones -----
    parts.append(
        '<line x1="{ml}" y1="{y}" x2="{xr}" y2="{y}" '
        'stroke="rgba(128,128,128,0.15)" stroke-width="0.5"/>'.format(
            ml=ml, xr=w - mr, y=round(rug_top, 1)))

    # ----- X-axis labels (T25: fewer in a compact chart) -----
    x_ticks = _nice_ticks(lo, hi, 5 if compact else 8)
    for tv in x_ticks:
        if lo <= tv <= hi:
            tx = x_pos(tv)
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" font-size="11" '
                'fill="var(--muted,#6b6b65)">{lbl}</text>'.format(
                    x=round(tx, 1), y=h - mb + 16,
                    lbl=_esc(_fmt_tick(tv))))

    # X-axis title
    parts.append(
        '<text x="{cx}" y="{y}" text-anchor="middle" font-size="13" '
        'fill="var(--muted,#6b6b65)">{u}</text>'.format(
            cx=round(ml + plot_w / 2), y=h - 4, u=_esc(unit)))

    # Y-axis label for density zone
    parts.append(
        '<text x="{tx}" y="{cy}" text-anchor="middle" font-size="13" '
        'fill="var(--muted,#6b6b65)" '
        'transform="rotate(-90,{tx},{cy})">density</text>'.format(
            tx=g.y_title_x, cy=round(mt + density_h / 2)))

    # ----- Threshold lines (full height) -----
    if threshold_pct is not None:
        lo_bound = mean_val * (1 - threshold_pct / 100)
        hi_bound = mean_val * (1 + threshold_pct / 100)
        if lo < lo_bound < hi:
            lx = x_pos(lo_bound)
            parts.append(
                '<line x1="{x}" y1="{mt}" x2="{x}" y2="{mb}" '
                'stroke="var(--danger,#f09595)" stroke-width="1" '
                'stroke-dasharray="4,3" opacity="0.7"/>'.format(
                    x=round(lx, 1), mt=mt, mb=rug_base))
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" '
                'font-size="{fs}" fill="var(--danger,#f09595)" opacity="0.7">'
                '-{t}%</text>'.format(
                    x=round(lx, 1), y=mt - 4, fs=g.note_font, t=int(threshold_pct)))
        if lo < hi_bound < hi:
            hx = x_pos(hi_bound)
            parts.append(
                '<line x1="{x}" y1="{mt}" x2="{x}" y2="{mb}" '
                'stroke="var(--danger,#f09595)" stroke-width="1" '
                'stroke-dasharray="4,3" opacity="0.7"/>'.format(
                    x=round(hx, 1), mt=mt, mb=rug_base))
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" '
                'font-size="{fs}" fill="var(--danger,#f09595)" opacity="0.7">'
                '+{t}%</text>'.format(
                    x=round(hx, 1), y=mt - 4, fs=g.note_font, t=int(threshold_pct)))

    # ----- Fleet mean vertical line -----
    if lo <= mean_val <= hi:
        mx = x_pos(mean_val)
        parts.append(
            '<line x1="{x}" y1="{mt}" x2="{x}" y2="{mb}" '
            'stroke="var(--muted,#9a9a92)" stroke-width="1.5" '
            'opacity="0.8"/>'.format(
                x=round(mx, 1), mt=mt, mb=rug_base))
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="middle" '
            'font-size="{fs}" fill="var(--muted,#9a9a92)" opacity="0.8">'
            'mean</text>'.format(
                x=round(mx, 1), y=mt - 4, fs=g.note_font))

    # ----- Sigma band legend (T25: none in a compact chart) -----
    if stdev > 0 and not compact:
        lx = w - mr - 280
        ly = h - 14
        for si, (sc, slbl) in enumerate(SIGMA_COLORS):
            sx = lx + si * 70
            parts.append(
                '<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                'stroke="{c}" stroke-width="2"/>'.format(
                    x=sx, y1=ly - 4, y2=ly + 4, c=sc))
            parts.append(
                '<text x="{x}" y="{y}" font-size="9" '
                'fill="var(--muted,#6b6b65)">{lbl}</text>'.format(
                    x=sx + 5, y=ly + 3, lbl=slbl))

    parts.append('</svg>')
    return "\n".join(parts)


def _svg_single_bar(nodes, node_values, unit, color, compact=False):
    # type: (list, dict, str, str, bool) -> str
    """Simple per-node bar chart as inline SVG for secondary metrics."""
    g = _chart_geometry(compact, h=_SVG_H_STRIP)
    w, h = g.w, g.h
    ml, mr, mt, mb = g.ml, g.mr, g.mt, g.mb
    plot_w, plot_h = g.plot_w, g.plot_h

    data = [node_values.get(n, 0) for n in nodes]
    pos_vals = [v for v in data if v > 0]
    if not pos_vals:
        return ""

    y_min = min(pos_vals) * 0.9
    y_max = max(pos_vals) * 1.05
    if y_max <= y_min:
        y_max = y_min + 1

    parts = [_svg_open(w, h)]

    # Y-axis
    ticks = _nice_ticks(y_min, y_max, 4)
    for tv in ticks:
        yp = mt + plot_h - (tv - y_min) / (y_max - y_min) * plot_h
        parts.append(
            '<line x1="{ml}" y1="{y}" x2="{xr}" y2="{y}" '
            'stroke="rgba(128,128,128,0.12)" stroke-dasharray="2,4"/>'.format(
                ml=ml, xr=w - mr, y=round(yp, 1)))
        parts.append(
            '<text x="{x}" y="{y}" text-anchor="end" '
            'font-size="12" fill="var(--muted,#6b6b65)">{lbl}</text>'.format(
                x=ml - 6, y=round(yp + 3, 1), lbl=_esc(_fmt_tick(tv))))

    # Y-axis title
    parts.append(
        '<text x="{tx}" y="{cy}" text-anchor="middle" font-size="13" '
        'fill="var(--muted,#6b6b65)" transform="rotate(-90,{tx},{cy})">{u}</text>'.format(
            tx=g.y_title_x, cy=round(mt + plot_h / 2), u=_esc(unit)))

    # Bars
    n = len(nodes)
    bar_w = plot_w / n if n > 0 else plot_w
    gap = max(1, bar_w * 0.15)
    for i, node in enumerate(nodes):
        val = node_values.get(node, 0)
        if val <= 0:
            continue
        bar_h = (val - y_min) / (y_max - y_min) * plot_h
        bx = ml + i * bar_w + gap
        by = mt + plot_h - bar_h
        parts.append(
            '<rect x="{x}" y="{y}" width="{bw}" height="{bh}" '
            'fill="{c}" rx="2">'
            '<title>{node}: {val} {unit}</title>'
            '</rect>'.format(
                x=round(bx, 1), y=round(by, 1),
                bw=round(bar_w - gap * 2, 1), bh=round(bar_h, 1),
                c=color, node=_esc(node),
                val=_esc(_fmt_tick(val)), unit=_esc(unit)))

    # X-axis labels (T25: a compact chart labels at most every k-th node)
    rotate = n > 10
    step = _label_step(n, _COMPACT_MAX_X_LABELS) if compact else 1
    for i, node in enumerate(nodes):
        if i % step:
            continue
        cx = ml + i * bar_w + bar_w / 2
        if rotate:
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="end" font-size="11" '
                'fill="var(--muted,#6b6b65)" '
                'transform="rotate(-45,{x},{y})">{n}</text>'.format(
                    x=round(cx, 1), y=h - mb + 14, n=_esc(node)))
        else:
            parts.append(
                '<text x="{x}" y="{y}" text-anchor="middle" font-size="12" '
                'fill="var(--muted,#6b6b65)">{n}</text>'.format(
                    x=round(cx, 1), y=h - mb + 16, n=_esc(node)))

    parts.append('</svg>')
    return "\n".join(parts)



def _build_bench_section(bench_id, bs, outlier_hosts, threshold, node_map=None,
                         dot_plot=False):
    # type: (int, BenchmarkStats, set, float, Optional[Dict[str, str]], bool) -> str
    """Build one benchmark section: SVG chart(s) + table.

    Scale-adaptive: >50 nodes switches to histogram + truncated table.
    Multi-metric: shows power and temperature charts when data is available.
    """
    label = _esc(bench_label_from_key((bs.name, bs.dtype)))
    unit = _esc(bs.unit)

    by_node = _group_by(bs.results, lambda r: r.hostname)
    nodes = sorted(by_node.keys())
    gpu_indices = sorted({r.gpu for r in bs.results})
    large_fleet = len(nodes) > SCALE_THRESHOLD

    all_perf = [r.mean_val for r in bs.results if r.mean_val > 0]
    all_power = [r.power_avg_w for r in bs.results if r.power_avg_w > 0]
    all_temp = [r.temp_max_c for r in bs.results if r.temp_max_c > 0]
    all_sm_util = [r.sm_util_mean for r in bs.results if r.sm_util_mean > 0]
    all_mem_bw = [r.mem_bw_util_mean for r in bs.results if r.mem_bw_util_mean > 0]
    all_gpu_clock = [r.gpu_clock_mean for r in bs.results if r.gpu_clock_mean > 0]

    chart_fn = _svg_dot_plot if dot_plot else _svg_histogram

    letter = chr(97 + bench_id) if bench_id < 26 else str(bench_id)

    # ----- Build chart area -----
    # T4: every subtitle names population (GPU), statistic and metric.
    # Temperature is the per-GPU run max (GPUResult.temp_max_c), not a mean.
    panels = []  # type: list  # (subtitle, svg_html)
    if all_perf:
        n_bins = 40
        if len(all_perf) < 50:
            n_bins = max(10, len(all_perf) // 3)
        if len(all_perf) < 10:
            n_bins = len(all_perf)
        if dot_plot:
            panels.append(("Fleet distribution of GPU mean performance ({})".format(unit),
                           chart_fn(all_perf, unit, bs.fleet_mean,
                                    n_bins=n_bins, threshold_pct=threshold,
                                    results=bs.results)))
        else:
            panels.append(("Fleet distribution of GPU mean performance ({})".format(unit),
                           chart_fn(all_perf, unit, bs.fleet_mean,
                                    n_bins=n_bins, threshold_pct=threshold)))

    # T24/T26: the performance chart stays full size; the diagnostic panels
    # beside it are compact small multiples.
    diagnostics = [
        (all_power, "W", "Fleet distribution of GPU mean power (W)"),
        (all_temp, "\u00b0C", "Fleet distribution of GPU max temperature (\u00b0C)"),
        (all_sm_util, "%", "Fleet distribution of GPU mean SM utilization (%)"),
        (all_mem_bw, "%", "Fleet distribution of GPU mean memory BW utilization (%)"),
        (all_gpu_clock, "MHz", "Fleet distribution of GPU mean GPU clock (MHz)"),
    ]
    for values, metric_unit, subtitle in diagnostics:
        if not values:
            continue
        panels.append((subtitle,
                       chart_fn(values, metric_unit, statistics.mean(values),
                                n_bins=min(20, max(5, len(values) // 3)),
                                compact=True)))

    if len(panels) == 1:
        chart_area = (
            '<div class="chart-subtitle">{sub}</div>\n{svg}'
        ).format(sub=_esc(panels[0][0]), svg=panels[0][1])
    elif len(panels) > 1:
        # Performance chart full-width on top, rest in grid below
        perf_panel = panels[0]
        diag_panels = panels[1:]
        chart_area = (
            '<div class="chart-subtitle">{sub}</div>\n{svg}'
        ).format(sub=_esc(perf_panel[0]), svg=perf_panel[1])
        if diag_panels:
            diag_html = "".join(
                '<div class="chart-panel">'
                '<div class="chart-subtitle">{sub}</div>'
                '{svg}'
                '</div>'.format(sub=_esc(p[0]), svg=p[1])
                for p in diag_panels)
            chart_area += '\n<div class="chart-row chart-secondary">{}</div>'.format(diag_html)
    else:
        chart_area = ""

    # Stats line (T1): mean is the headline; median precedes the p5/p95
    # interval so the robust centre is read before its spread.
    stats_parts = ["fleet mean: {} {}".format(_fmt_html(bs.fleet_mean, 1), unit)]
    if bs.fleet_cv > 0:
        stats_parts.append("CV: {:.1f}%".format(bs.fleet_cv))
    if len(all_perf) >= 10:
        sorted_vals = sorted(all_perf)
        p5 = sorted_vals[int(len(sorted_vals) * 0.05)]
        p50 = sorted_vals[int(len(sorted_vals) * 0.50)]
        p95 = sorted_vals[int(len(sorted_vals) * 0.95)]
        stats_parts.append("median: {}".format(_fmt_html(p50)))
        stats_parts.append("p5/p95: {}/{}".format(_fmt_html(p5), _fmt_html(p95)))
    if bs.power_values:
        stats_parts.append("mean power: {:.0f}W".format(statistics.mean(bs.power_values)))
    if bs.temp_values:
        stats_parts.append("max temp: {:.0f} C".format(max(bs.temp_values)))
    if bs.throttled_results:
        stats_parts.append("\u26a0 {} GPU(s) throttled".format(len(bs.throttled_results)))
    stats_line = _esc(" | ".join(stats_parts))

    # Table
    has_loc = bool(node_map)
    summaries = _compute_node_summaries(bs, by_node, nodes, gpu_indices, node_map)
    loc_col_offset = 1 if has_loc else 0
    gpu_headers = "".join(
        '<th class="sortable" data-col="{}">GPU {}</th>'.format(i + 1 + loc_col_offset, idx)
        for i, idx in enumerate(gpu_indices))
    table_body = _build_table_html(
        summaries, gpu_indices, outlier_hosts, threshold, truncate=large_fleet,
        bench_id=bench_id, has_location=has_loc)

    loc_header = ""
    if has_loc:
        loc_header = '        <th class="sortable" data-col="1">location</th>\n'

    return (
        '<div class="section">\n'
        '  <h2 class="section-title">({letter}) {label}'
        '    <span class="unit">[{unit}]</span></h2>\n'
        '  <div class="stats-line">{stats_line}</div>\n'
        '  <div class="chart-wrap">\n'
        '    <button class="zoom-btn" onclick="this.parentElement.classList.toggle(\'chart-zoomed\')"'
        ' title="Toggle full-width view">\u2922</button>\n'
        '    {chart_area}\n'
        '  </div>\n'
        '  <div class="table-wrap">\n'
        '    <table class="sortable-table">\n'
        '      <thead><tr>\n'
        '        <th class="sortable" data-col="0">node</th>\n'
        '{loc_header}'
        '        {gpu_headers}\n'
        '        <th class="sortable" data-col="{avg_col}">node mean</th>\n'
        '        <th class="sortable" data-col="{vs_col}" title="% deviation from fleet mean performance">vs fleet</th>\n'
        '        <th class="sortable" data-col="{thrt_col}" title="Thermal throttling detected">thrt</th>\n'
        '      </tr></thead>\n'
        '      <tbody>{table_body}</tbody>\n'
        '    </table>\n'
        '  </div>\n'
        '</div>\n'
    ).format(
        letter=letter, label=label, unit=unit, stats_line=stats_line,
        chart_area=chart_area,
        loc_header=loc_header,
        gpu_headers=gpu_headers, table_body=table_body,
        avg_col=1 + loc_col_offset + len(gpu_indices),
        vs_col=2 + loc_col_offset + len(gpu_indices),
        thrt_col=3 + loc_col_offset + len(gpu_indices),
    )


def render_html(results, bench_stats, outliers, source_name, threshold,
                system_name="", job_name="", node_map=None, dot_plot=False,
                theme_mode="auto"):
    # type: (List[GPUResult], OrderedDict, List[Dict], str, float, str, str, Optional[Dict[str, str]], bool, str) -> str
    """Render full HTML report with one section per (benchmark, dtype)."""
    hosts = sorted(set(r.hostname for r in results))
    gpus = sorted(set((r.hostname, r.gpu) for r in results))
    models = sorted(set(r.gpu_model for r in results if r.gpu_model))

    bad_outlier_hosts = set()
    for o in outliers:
        if o["severity"] == "bad":
            bad_outlier_hosts.add(o["host"])

    # Report title
    title_parts = [p for p in [system_name, job_name, source_name] if p]
    page_title = " \u2014 ".join(title_parts)

    # Build per-benchmark sections
    sections = []
    for idx, (key, bs) in enumerate(bench_stats.items()):
        bench_outlier_hosts = set()
        for o in outliers:
            if o["severity"] == "bad" and o["benchmark"] == bench_label_from_key(key):
                bench_outlier_hosts.add(o["host"])
        sections.append(_build_bench_section(idx, bs, bench_outlier_hosts, threshold,
                                             node_map, dot_plot=dot_plot))

    bench_sections_html = "\n".join(sections)

    # Outlier section
    bad = [o for o in outliers if o["severity"] == "bad"]
    if bad:
        rows = []
        for o in bad[:30]:
            loc_span = ""
            if node_map and o["host"] in node_map:
                loc_span = ' <span style="color:var(--muted)">[{}]</span>'.format(
                    _esc(node_map[o["host"]]))
            rows.append(
                '<div class="outlier-row">'
                '<span class="marker">x</span> '
                '{}{}:GPU{} -- '
                '{}: '
                '{} {} '
                '({:.1f}% {} fleet)'
                '</div>'.format(
                    _esc(o["host"]), loc_span, o["gpu"],
                    _esc(o["benchmark"]),
                    _fmt_html(o["value"], 1), _esc(o["unit"]),
                    o["deviation_pct"], o["direction"],
                )
            )
        if len(bad) > 30:
            rows.append('<div class="outlier-row">... and {} more</div>'.format(len(bad) - 30))

        # Location cluster summary
        cluster_html = ""
        if node_map:
            loc_counts = {}  # type: Dict[str, int]
            for o in bad:
                loc = node_map.get(o["host"], "")
                if loc:
                    loc_counts[loc] = loc_counts.get(loc, 0) + 1
            multi = {l: c for l, c in loc_counts.items() if c > 1}
            if multi:
                cluster_parts = []
                for loc, cnt in sorted(multi.items(), key=lambda x: -x[1]):
                    cluster_parts.append(
                        '<div class="outlier-row" style="color:var(--warn)">'
                        '  {} \u2014 {} outlier GPU(s)</div>'.format(
                            _esc(loc), cnt))
                cluster_html = (
                    '<h3 style="font-size:13px;margin-top:1rem;color:var(--muted)">'
                    'Location clusters</h3>{}'.format("".join(cluster_parts)))

        outlier_html = (
            '<div class="section outlier-section">'
            '<h2 class="section-title">Outliers (>{:.0f}% from fleet mean)</h2>'
            '{}{}'
            '</div>'.format(threshold, "".join(rows), cluster_html)
        )
    else:
        outlier_html = (
            '<div class="section outlier-section">'
            '<h2 class="section-title">Outliers</h2>'
            '<div style="color:var(--success)">None detected</div>'
            '</div>'
        )

    # Build page
    css = _get_css(theme_mode=theme_mode)
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Torch Hammer report -- {title}</title>\n'
        '<style>\n{css}\n</style>\n'
        '</head>\n'
        '<body>\n'
        '<div class="page">\n'
        '  <div class="header">\n'
        '    <h1>Torch Hammer benchmark report</h1>\n'
        '    {subtitle}\n'
        '    <div class="meta">\n'
        '      <span>source: {source}</span>\n'
        '      <span>GPU: {gpu_model}</span>\n'
        '      <span>nodes: {n_nodes}</span>\n'
        '      <span>GPUs: {n_gpus}</span>\n'
        '      <span>benchmarks: {n_benchmarks}</span>\n'
        '    </div>\n'
        '  </div>\n'
        '\n'
        '  <div class="metrics">\n'
        '    <div class="metric-card ok">\n'
        '      <div class="label">nodes</div>\n'
        '      <div class="value">{n_nodes}</div>\n'
        '    </div>\n'
        '    <div class="metric-card">\n'
        '      <div class="label">GPUs</div>\n'
        '      <div class="value">{n_gpus}</div>\n'
        '    </div>\n'
        '    <div class="metric-card">\n'
        '      <div class="label">benchmarks</div>\n'
        '      <div class="value">{n_benchmarks}</div>\n'
        '    </div>\n'
        '    <div class="metric-card {outlier_card_cls}">\n'
        '      <div class="label">outliers</div>\n'
        '      <div class="value">{n_outliers}</div>\n'
        '    </div>\n'
        '  </div>\n'
        '\n'
        '  {bench_sections}\n'
        '\n'
        '  {outlier_section}\n'
        '\n'
        '  <footer>generated by torch-hammer-reporter | palette: Okabe-Ito (colorblind-safe)</footer>\n'
        '<script>\n'
        '(function() {{\n'
        '  document.querySelectorAll("th.sortable").forEach(th => {{\n'
        '    th.addEventListener("click", function() {{\n'
        '      const table = th.closest("table");\n'
        '      const tbody = table.querySelector("tbody");\n'
        '      const col = parseInt(th.dataset.col);\n'
        '      const rows = Array.from(tbody.querySelectorAll("tr:not(.separator):not(.section-label)"));\n'
        '      const seps = Array.from(tbody.querySelectorAll("tr.separator, tr.section-label"));\n'
        '      const isAsc = th.classList.contains("asc");\n'
        '      table.querySelectorAll("th.sortable").forEach(h => h.classList.remove("asc","desc"));\n'
        '      th.classList.add(isAsc ? "desc" : "asc");\n'
        '      const dir = isAsc ? -1 : 1;\n'
        '      rows.sort((a,b) => {{\n'
        '        const ca = a.children[col]; const cb = b.children[col];\n'
        '        const va = ca ? (ca.dataset.sort !== undefined ? parseFloat(ca.dataset.sort) : ca.textContent) : "";\n'
        '        const vb = cb ? (cb.dataset.sort !== undefined ? parseFloat(cb.dataset.sort) : cb.textContent) : "";\n'
        '        if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;\n'
        '        return String(va).localeCompare(String(vb)) * dir;\n'
        '      }});\n'
        '      while (tbody.firstChild) tbody.removeChild(tbody.firstChild);\n'
        '      rows.forEach(r => tbody.appendChild(r));\n'
        '      seps.forEach(r => tbody.appendChild(r));\n'
        '    }});\n'
        '  }});\n'
        '}})();\n'
        '</script>\n'
        '</div>\n'
        '</body>\n'
        '</html>\n'
    ).format(
        title=_esc(page_title),
        css=css,
        subtitle=('<div class="subtitle">{}</div>'.format(_esc(page_title))
                  if system_name or job_name else ''),
        source=_esc(source_name),
        gpu_model=_esc(models[0] if models else "--"),
        n_nodes=len(hosts),
        n_gpus=len(gpus),
        n_benchmarks=len(bench_stats),
        n_outliers=len(bad),
        outlier_card_cls="bad" if bad else "ok",
        bench_sections=bench_sections_html,
        outlier_section=outlier_html,
    )


def _get_css(theme_mode="auto"):
        # type: (str) -> str
        if theme_mode == "dark":
                theme_block = """
    :root {
        --bg: #1a1a18; --surface: #242422; --surface2: #2c2c2a;
        --border: rgba(255,255,255,0.08); --border2: rgba(255,255,255,0.15);
        --text: #e8e6df; --muted: #9a9a92; --hint: #6b6b65;
        --success: #5dcaa5; --danger: #f09595; --danger-bg: #451a1a;
        --warn: #fbbf24; --warn-bg: #422006;
    }
"""
        elif theme_mode == "auto":
                theme_block = """
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #1a1a18; --surface: #242422; --surface2: #2c2c2a;
            --border: rgba(255,255,255,0.08); --border2: rgba(255,255,255,0.15);
            --text: #e8e6df; --muted: #9a9a92; --hint: #6b6b65;
            --success: #5dcaa5; --danger: #f09595; --danger-bg: #451a1a;
            --warn: #fbbf24; --warn-bg: #422006;
        }
    }
"""
        else:
                theme_block = ""

        return """
  :root {
    --bg: #fafafa; --surface: #ffffff; --surface2: #f5f5f4;
    --border: rgba(0,0,0,0.08); --border2: rgba(0,0,0,0.15);
    --text: #1a1a18; --muted: #6b6b65; --hint: #9a9a92;
    --success: #0f6e56; --danger: #b91c1c; --danger-bg: #fef2f2;
    --warn: #92400e; --warn-bg: #fffbeb;
    --radius: 8px; --radius-lg: 12px;
  }
{theme_block}
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6;
  }
  .page { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
  .header { margin-bottom: 2rem; border-bottom: 1px solid var(--border2); padding-bottom: 1.25rem; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header .subtitle { font-size: 15px; color: var(--muted); margin-top: 2px; font-weight: 500; }
  .header .meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .header .meta span { margin-right: 1.5rem; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 10px; margin-bottom: 2rem; }
  .metric-card { background: var(--surface2); border-radius: var(--radius); padding: 0.875rem 1rem; }
  .metric-card .label { font-size: 11px; color: var(--muted); margin-bottom: 2px;
                        text-transform: uppercase; letter-spacing: 0.04em; }
  .metric-card .value { font-size: 22px; font-weight: 600;
                        font-family: ui-monospace, "SF Mono", monospace; }
  .metric-card.ok .value { color: var(--success); }
  .metric-card.bad .value { color: var(--danger); }
  .metric-card.warn .value { color: var(--warn); }
  .section { margin-bottom: 2.5rem; }
  .section-title { font-size: 15px; font-weight: 600; margin-bottom: 0.5rem; }
  .section-title .unit { font-size: 12px; color: var(--muted); font-weight: 400; }
  .stats-line { font-size: 12px; color: var(--muted); margin-bottom: 0.75rem;
                font-family: ui-monospace, "SF Mono", monospace; }
  .chart-wrap { background: var(--surface); border: 1px solid var(--border);
                border-radius: var(--radius-lg); padding: 1.25rem; margin-bottom: 0.75rem;
                position: relative; transition: max-width 0.2s ease; }
  .chart-wrap.chart-zoomed { max-width: 100vw; margin-left: calc(-50vw + 50%);
                             margin-right: calc(-50vw + 50%); padding: 1.5rem 2rem; }
  .zoom-btn { position: absolute; top: 8px; right: 10px; background: var(--surface2);
              border: 1px solid var(--border2); border-radius: 4px; padding: 2px 7px;
              font-size: 16px; line-height: 1; cursor: pointer; color: var(--muted);
              z-index: 1; transition: color 0.15s; }
  .zoom-btn:hover { color: var(--text); border-color: var(--text); }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 0.75rem; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }
  .legend-swatch { width: 10px; height: 10px; border-radius: 2px; }
  .chart-svg-wrap svg { width: 100%; height: auto; display: block; }
  .chart-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.75rem; }
  .chart-row .chart-panel { min-width: 0; }
  .chart-secondary { margin-top: 1rem; }
  .chart-secondary .chart-panel { padding: 0.25rem 0; }
  .chart-secondary .chart-subtitle { margin-bottom: 0.25rem; }
  .table-wrap { background: var(--surface); border: 1px solid var(--border);
                border-radius: var(--radius-lg); overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px;
          font-family: ui-monospace, "SF Mono", monospace; }
  thead tr { background: var(--surface2); }
  th { text-align: right; padding: 8px 12px; font-size: 11px; font-weight: 500;
       letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
       border-bottom: 1px solid var(--border2);
       font-family: system-ui, -apple-system, sans-serif; }
  th:first-child { text-align: left; }
  td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--border); }
  td:first-child { text-align: left; }
  td.host { font-weight: 500; }
  td.warn { color: var(--warn); font-weight: 600; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface2); }
  .mono { font-family: ui-monospace, "SF Mono", monospace; }
  .outlier-section { margin-top: 2rem; }
  .outlier-row { font-size: 13px; padding: 4px 0;
                 font-family: ui-monospace, "SF Mono", monospace; }
  .outlier-row .marker { color: var(--danger); font-weight: 600; }
  .chart-subtitle { font-size: 11px; color: var(--muted); margin-bottom: 0.5rem;
                    text-transform: uppercase; letter-spacing: 0.04em; }
  .separator td { border-bottom: none; }
  th.sortable { cursor: pointer; user-select: none; position: relative; }
  th.sortable:hover { color: var(--text); }
  th.sortable::after { content: ' \2195'; font-size: 10px; opacity: 0.5; }
  th.sortable.asc::after { content: ' \2191'; opacity: 0.8; }
  th.sortable.desc::after { content: ' \2193'; opacity: 0.8; }
  .section-label td { background: var(--surface2); }
  .show-all-btn { background: var(--surface2); border: 1px solid var(--border2);
                  border-radius: 4px; padding: 2px 10px; font-size: 11px;
                  color: var(--muted); cursor: pointer; margin-left: 8px;
                  font-family: system-ui, -apple-system, sans-serif; }
  .show-all-btn:hover { color: var(--text); border-color: var(--text); }
  .inner-expand td { padding: 7px 12px; text-align: right;
                     border-bottom: 1px solid var(--border); }
  .inner-expand td:first-child { text-align: left; }
  footer { margin-top: 3rem; font-size: 11px; color: var(--hint);
           border-top: 1px solid var(--border); padding-top: 1rem; }
""".replace("{theme_block}", theme_block)


# ---------------------------------------------------------------------------
# Interactive Plotly dashboard
# ---------------------------------------------------------------------------

def _render_interactive_html(results, bench_stats, outliers, source_name, threshold,
                             system_name="", job_name="", node_map=None,
                             theme_mode="auto"):
    # type: (List[GPUResult], OrderedDict, List[Dict], str, float, str, str, Optional[Dict[str,str]], str) -> str
    """Render a self-contained interactive HTML dashboard using Plotly.js.

    Embeds the full Plotly.js library from the installed pip package via
    ``plotly.offline.get_plotlyjs()``.  No CDN or external resources.
    """
    plotly_js = _plotly_offline.get_plotlyjs()

    results_data = []
    _results_cols = INTERACTIVE_RESULT_COLS
    for r in results:
        results_data.append([
            r.hostname, r.gpu, r.gpu_model, r.serial,
            r.benchmark, r.dtype,
            r.mean_val, r.min_val, r.max_val, r.unit,
            r.power_avg_w, r.temp_max_c,
            r.sm_util_mean, r.mem_bw_util_mean, r.gpu_clock_mean,
            r.throttled, r.throttle_samples,
        ])

    stats_data = {}
    for key, bs in bench_stats.items():
        label = bench_label_from_key(key)
        stats_data[label] = {
            "fleet_mean": bs.fleet_mean,
            "fleet_cv": bs.fleet_cv,
            "fleet_min": bs.fleet_min,
            "fleet_max": bs.fleet_max,
            "unit": bs.unit,
        }

    hosts = sorted(set(r.hostname for r in results))
    gpus = sorted(set((r.hostname, r.gpu) for r in results))
    models = sorted(set(r.gpu_model for r in results if r.gpu_model))
    bad_outliers = [o for o in outliers if o["severity"] == "bad"]
    benchmarks = sorted(set(bench_label(r) for r in results))

    title_parts = [p for p in [system_name, job_name, source_name] if p]
    page_title = " - ".join(title_parts) if title_parts else source_name

    has_iteration_data = bool(_iteration_data)

    results_json = _safe_json(results_data)
    results_cols_json = _safe_json(_results_cols)
    stats_json = _safe_json(stats_data)
    outliers_json = _safe_json(outliers)
    iteration_json = _safe_json(_iteration_data) if has_iteration_data else "{}"
    node_map_json = _safe_json(node_map) if node_map else "{}"

    benchmark_options = "".join(
        '<option value="{v}">{v}</option>'.format(v=_esc(b)) for b in benchmarks)
    hostname_options = "".join(
        '<option value="{v}">{v}</option>'.format(v=_esc(h)) for h in hosts)

    time_series_display = "block" if has_iteration_data else "none"

    parts = []
    parts.append('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
    parts.append('<meta charset="UTF-8">\n')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    parts.append('<title>Torch Hammer interactive report - {}</title>\n'.format(_esc(page_title)))
    parts.append('<script>{}</script>\n'.format(plotly_js))
    parts.append('<style>\n')
    parts.append(':root {\n')
    parts.append('  --bg: #0f1419; --surface: #1a1f26; --surface2: #242b34;\n')
    parts.append('  --border: rgba(255,255,255,0.08); --border2: rgba(255,255,255,0.15);\n')
    parts.append('  --text: #e8eaed; --muted: #9aa0a6; --hint: #5f6368;\n')
    parts.append('  --success: #5dcaa5; --danger: #f09595; --warn: #fbbf24;\n')
    parts.append('  --radius: 8px; --radius-lg: 12px;\n')
    parts.append('  --plot-bg: #1a1f26; --plot-paper: #0f1419;\n')
    parts.append('  --grid-color: rgba(255,255,255,0.06); --zero-color: rgba(255,255,255,0.1);\n')
    parts.append('  --axis-font: #9aa0a6; --plot-font: #e8eaed;\n')
    parts.append('}\n')
    parts.append(':root.light {\n')
    parts.append('  --bg: #f5f6f8; --surface: #ffffff; --surface2: #eef0f3;\n')
    parts.append('  --border: rgba(0,0,0,0.08); --border2: rgba(0,0,0,0.15);\n')
    parts.append('  --text: #1a1a1a; --muted: #5f6368; --hint: #9aa0a6;\n')
    parts.append('  --success: #1a8f6e; --danger: #c0392b; --warn: #d4880f;\n')
    parts.append('  --plot-bg: #ffffff; --plot-paper: #f5f6f8;\n')
    parts.append('  --grid-color: rgba(0,0,0,0.06); --zero-color: rgba(0,0,0,0.1);\n')
    parts.append('  --axis-font: #5f6368; --plot-font: #1a1a1a;\n')
    parts.append('}\n')
    parts.append('* { box-sizing: border-box; margin: 0; padding: 0; }\n')
    parts.append('body {\n')
    parts.append('  font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;\n')
    parts.append('  background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6;\n')
    parts.append('}\n')
    parts.append('.page { max-width: 1400px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }\n')
    parts.append('.header { margin-bottom: 2rem; border-bottom: 1px solid var(--border2); padding-bottom: 1.25rem; }\n')
    parts.append('.header h1 { font-size: 20px; font-weight: 600; }\n')
    parts.append('.header .subtitle { font-size: 15px; color: var(--muted); margin-top: 2px; }\n')
    parts.append('.header .meta { font-size: 12px; color: var(--muted); margin-top: 4px; }\n')
    parts.append('.header .meta span { margin-right: 1.5rem; }\n')
    parts.append('.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));\n')
    parts.append('           gap: 10px; margin-bottom: 2rem; }\n')
    parts.append('.metric-card { background: var(--surface2); border-radius: var(--radius); padding: 0.875rem 1rem; }\n')
    parts.append('.metric-card .label { font-size: 11px; color: var(--muted); margin-bottom: 2px;\n')
    parts.append('                      text-transform: uppercase; letter-spacing: 0.04em; }\n')
    parts.append('.metric-card .value { font-size: 22px; font-weight: 600;\n')
    parts.append('                      font-family: ui-monospace, "SF Mono", monospace; }\n')
    parts.append('.metric-card.ok .value { color: var(--success); }\n')
    parts.append('.metric-card.bad .value { color: var(--danger); }\n')
    parts.append('.controls { background: var(--surface); border: 1px solid var(--border);\n')
    parts.append('            border-radius: var(--radius-lg); padding: 1rem 1.25rem; margin-bottom: 1.5rem;\n')
    parts.append('            display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end; }\n')
    parts.append('.control-group { display: flex; flex-direction: column; gap: 4px; }\n')
    parts.append('.control-group label { font-size: 11px; color: var(--muted); text-transform: uppercase;\n')
    parts.append('                       letter-spacing: 0.04em; }\n')
    parts.append('.control-group select, .control-group input { background: var(--surface2); color: var(--text);\n')
    parts.append('  border: 1px solid var(--border2); border-radius: 4px; padding: 6px 10px; font-size: 13px; }\n')
    parts.append('.chart-section { background: var(--surface); border: 1px solid var(--border);\n')
    parts.append('                 border-radius: var(--radius-lg); padding: 1.25rem; margin-bottom: 1.5rem; }\n')
    parts.append('.chart-section h2 { font-size: 15px; font-weight: 600; margin-bottom: 0.75rem; }\n')
    parts.append('.chart-div { width: 100%; min-height: 400px; }\n')
    parts.append('table { width: 100%; border-collapse: collapse; font-size: 13px;\n')
    parts.append('        font-family: ui-monospace, "SF Mono", monospace; }\n')
    parts.append('thead tr { background: var(--surface2); }\n')
    parts.append('th { text-align: right; padding: 8px 12px; font-size: 11px; font-weight: 500;\n')
    parts.append('     letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);\n')
    parts.append('     border-bottom: 1px solid var(--border2); }\n')
    parts.append('th:first-child { text-align: left; }\n')
    parts.append('th.sortable { cursor: pointer; user-select: none; position: relative; }\n')
    parts.append('th.sortable:hover { color: var(--text); }\n')
    parts.append('th.sortable::after { content: " \\2195"; font-size: 10px; opacity: 0.5; }\n')
    parts.append('th.sortable.asc::after { content: " \\2191"; opacity: 0.8; }\n')
    parts.append('th.sortable.desc::after { content: " \\2193"; opacity: 0.8; }\n')
    parts.append('td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--border); }\n')
    parts.append('td:first-child { text-align: left; }\n')
    parts.append('tr:hover td { background: var(--surface2); }\n')
    parts.append('.outlier { color: var(--danger); font-weight: 600; }\n')
    parts.append('.nc { border-radius:3px; cursor:pointer; transition:transform 0.1s; }\n')
    parts.append('.nc:hover { transform:scale(1.3); z-index:10; }\n')
    parts.append('.chart-subtitle { font-size: 12px; color: var(--muted); margin-top: -0.25rem; margin-bottom: 0.5rem; }\n')
    parts.append('#theme-toggle { background: var(--surface2); border: 1px solid var(--border2);\n')
    parts.append('  color: var(--muted); border-radius: var(--radius); padding: 4px 10px;\n')
    parts.append('  font-size: 12px; cursor: pointer; float: right; margin-top: -2px; }\n')
    parts.append('#theme-toggle:hover { color: var(--text); }\n')
    parts.append('footer { margin-top: 3rem; font-size: 11px; color: var(--hint);\n')
    parts.append('         border-top: 1px solid var(--border); padding-top: 1rem; }\n')
    parts.append('</style>\n</head>\n<body>\n<div class="page">\n')

    # Header
    parts.append('  <div class="header">\n')
    parts.append('    <button id="theme-toggle" onclick="toggleTheme()">dark mode</button>\n')
    parts.append('    <h1>Torch Hammer interactive report</h1>\n')
    if system_name or job_name:
        parts.append('    <div class="subtitle">{}</div>\n'.format(_esc(page_title)))
    parts.append('    <div class="meta">\n')
    parts.append('      <span>source: {}</span>\n'.format(_esc(source_name)))
    parts.append('      <span>GPU: {}</span>\n'.format(_esc(models[0] if models else "--")))
    parts.append('      <span>nodes: {}</span>\n'.format(len(hosts)))
    parts.append('      <span>GPUs: {}</span>\n'.format(len(gpus)))
    parts.append('      <span>outliers: {}</span>\n'.format(len(bad_outliers)))
    throttled_gpus = set()
    for r in results:
        if r.throttled:
            throttled_gpus.add((r.hostname, r.gpu))
    if throttled_gpus:
        parts.append('      <span style="color:var(--warn)">\u26a0 {} GPU(s) throttled</span>\n'.format(len(throttled_gpus)))
    parts.append('    </div>\n  </div>\n')

    # Metric cards
    outlier_cls = "bad" if bad_outliers else "ok"
    parts.append('  <div class="metrics">\n')
    parts.append('    <div class="metric-card"><div class="label">nodes</div>'
                 '<div class="value">{}</div></div>\n'.format(len(hosts)))
    parts.append('    <div class="metric-card"><div class="label">GPUs</div>'
                 '<div class="value">{}</div></div>\n'.format(len(gpus)))
    parts.append('    <div class="metric-card"><div class="label">benchmarks</div>'
                 '<div class="value">{}</div></div>\n'.format(len(bench_stats)))
    parts.append('    <div class="metric-card" id="card-outliers"><div class="label">outliers</div>'
                 '<div class="value">{}</div></div>\n'.format(len(bad_outliers)))
    throttle_cls = "bad" if throttled_gpus else "ok"
    parts.append('    <div class="metric-card {}"><div class="label">throttled</div>'
                 '<div class="value">{}</div></div>\n'.format(throttle_cls, len(throttled_gpus)))
    parts.append('  </div>\n')

    # Filter controls
    parts.append('  <div class="controls">\n')
    parts.append('    <div class="control-group">\n')
    parts.append('      <label for="filter-host">Hostname</label>\n')
    parts.append('      <select id="filter-host"><option value="all">All</option>'
                 '{}</select>\n'.format(hostname_options))
    parts.append('    </div>\n')
    parts.append('    <div class="control-group">\n')
    parts.append('      <label for="filter-bench">Benchmark</label>\n')
    parts.append('      <select id="filter-bench"><option value="all">All</option>'
                 '{}</select>\n'.format(benchmark_options))
    parts.append('    </div>\n')
    parts.append('    <div class="control-group">\n')
    parts.append('      <label for="metric-select">Metric</label>\n')
    parts.append('      <select id="metric-select">\n')
    parts.append('        <option value="mean_val">Performance</option>\n')
    parts.append('        <option value="power_avg_w">Power (W)</option>\n')
    parts.append('        <option value="temp_max_c">Temperature (C)</option>\n')
    parts.append('        <option value="sm_util_mean">SM Utilization (%)</option>\n')
    parts.append('        <option value="mem_bw_util_mean">Mem BW Utilization (%)</option>\n')
    parts.append('        <option value="gpu_clock_mean">GPU Clock (MHz)</option>\n')
    parts.append('      </select>\n')
    parts.append('    </div>\n')
    parts.append('    <div class="control-group">\n')
    parts.append('      <label for="outlier-sigma">Outlier threshold</label>\n')
    parts.append('      <input type="range" id="outlier-sigma" min="1" max="3" step="0.5" value="2">\n')
    parts.append('      <span id="sigma-label" style="font-size:12px;color:var(--muted)">2.0 sigma</span>\n')
    parts.append('    </div>\n')
    parts.append('  </div>\n')

    # Chart sections
    parts.append('  <div class="chart-section">\n')
    parts.append('    <h2>Fleet Map</h2>\n')
    parts.append('    <div class="chart-subtitle">Each square uses the node mean of the selected metric. Spots topology-correlated problems. Uniform color = healthy fleet. Clusters of red/blue within a cabinet = localized issue (cooling, power, hardware fault).</div>\n')
    parts.append('    <div id="fleetmap-legend" style="font-size:11px;color:var(--muted);margin-bottom:8px"></div>\n')
    parts.append('    <div id="fleetmap-chart" style="padding:0.5rem 0"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <div class="chart-section">\n')
    parts.append('    <h2>Fleet Distribution</h2>\n')
    parts.append('    <div class="chart-subtitle">Distribution of GPU mean of the selected metric (GPU max for temperature). Tight bell curve = consistent fleet. Long tails or secondary peaks = subpopulations worth investigating. GPUs beyond \u00b12\u03c3 (red) are statistical outliers.</div>\n')
    parts.append('    <div id="distribution-chart" class="chart-div"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <div class="chart-section" id="efficiency-section">\n')
    parts.append('    <h2 id="efficiency-title">Power vs Performance</h2>\n')
    parts.append('    <div class="chart-subtitle">Reveals whether power limits performance. Color = temperature (blue cool, red hot). Larger dots = outlier or throttled GPUs. Tight cluster = healthy. Horizontal spread at fixed performance = power variation without impact. Diagonal trend = power-limited GPUs.</div>\n')
    parts.append('    <div id="efficiency-chart" class="chart-div"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <div class="chart-section">\n')
    parts.append('    <h2>Node Variability</h2>\n')
    parts.append('    <div class="chart-subtitle">Separates node-level problems from GPU-level problems. Tight dots per node = GPUs agree (node is fine or uniformly degraded). Scattered dots = intra-node variation.</div>\n')
    parts.append('    <div id="strip-chart" class="chart-div"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <div class="chart-section" id="timeseries-section" '
                 'style="display:{}">\n'.format(time_series_display))
    parts.append('    <h2>Iteration Trace</h2>\n')
    parts.append('    <div class="chart-subtitle">Shows stability over time. Flat envelope = steady state. Widening envelope = fleet diverging. Downward drift = thermal throttling or degradation.</div>\n')
    parts.append('    <div id="timeseries-chart" class="chart-div"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <div class="chart-section" id="waterfall-section" '
                 'style="display:{}">\n'.format(time_series_display))
    parts.append('    <h2 id="waterfall-title">Fleet Waterfall</h2>\n')
    parts.append('    <div class="chart-subtitle" id="waterfall-subtitle">Vertical stripes = consistent per-GPU difference. '
                 'Horizontal bands = all GPUs affected simultaneously. Uniform color = healthy fleet.</div>\n')
    parts.append('    <div style="margin-bottom:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">\n')
    parts.append('      <label style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:0.04em">Metric</label>\n')
    parts.append('      <select id="waterfall-metric" style="background:var(--surface2);color:var(--text);'
                 'border:1px solid var(--border2);border-radius:4px;padding:4px 8px;font-size:12px">\n')
    parts.append('        <option value="performance">Performance</option>\n')
    parts.append('        <option value="temp_gpu_C">Temperature</option>\n')
    parts.append('        <option value="gpu_clock">GPU Clock</option>\n')
    parts.append('        <option value="sm_util">SM Utilization</option>\n')
    parts.append('        <option value="mem_bw_util">Memory BW Utilization</option>\n')
    parts.append('        <option value="power_W">Power</option>\n')
    parts.append('      </select>\n')
    parts.append('      <label style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:0.04em">Sort</label>\n')
    parts.append('      <select id="waterfall-sort" style="background:var(--surface2);color:var(--text);'
                 'border:1px solid var(--border2);border-radius:4px;padding:4px 8px;font-size:12px">\n')
    parts.append('        <option value="topology">Topology</option>\n')
    parts.append('        <option value="performance">By value (ascending)</option>\n')
    parts.append('        <option value="hostname">By Hostname</option>\n')
    parts.append('      </select>\n')
    parts.append('    </div>\n')
    parts.append('    <div id="waterfall-chart" class="chart-div"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <div class="chart-section">\n')
    parts.append('    <h2>Fleet Inventory</h2>\n')
    parts.append('    <div style="margin-bottom:8px;display:flex;align-items:center;gap:12px">\n')
    parts.append('      <label style="font-size:12px;color:var(--muted);cursor:pointer;display:flex;align-items:center;gap:6px">\n')
    parts.append('        <input type="checkbox" id="inv-show-all" style="accent-color:var(--success)">\n')
    parts.append('        <span>Show all GPUs</span>\n')
    parts.append('      </label>\n')
    parts.append('      <span id="inv-count" style="font-size:11px;color:var(--hint)"></span>\n')
    parts.append('    </div>\n')
    parts.append('    <div id="inventory-table"></div>\n')
    parts.append('  </div>\n')

    parts.append('  <footer>generated by Torch Hammer reporter (interactive mode) '
                 '| plotly.js embedded</footer>\n')
    parts.append('</div>\n')

    # Embedded data + JS
    # Split data and code into separate script blocks to avoid
    # browser inline-script parser limits with large datasets.
    parts.append('<script>\n')
    parts.append('var _RCOLS = {};\n'.format(results_cols_json))
    parts.append('var _RDATA = {};\n'.format(results_json))
    parts.append('var RESULTS = _RDATA.map(function(r){var o={};_RCOLS.forEach('
                 'function(k,i){o[k]=r[i];});return o;});\n')
    parts.append('_RDATA=null;\n')  # free memory
    parts.append('var STATS = {};\n'.format(stats_json))
    parts.append('var OUTLIERS = {};\n'.format(outliers_json))
    parts.append('var NODE_MAP = {};\n'.format(node_map_json))
    parts.append('var THRESHOLD = {};\n'.format(threshold))
    parts.append('var THEME_MODE = {};\n'.format(_safe_json(theme_mode)))
    parts.append('</script>\n')
    # Iteration data in its own block (can be very large).
    # Compress with gzip+base64 to reduce file size (typically 6-8x smaller)
    # and avoid browser JS parser limits with giant JSON literals.
    if has_iteration_data:
        _iter_bytes = iteration_json.encode('utf-8')
        _iter_gz = gzip.compress(_iter_bytes, compresslevel=6)
        _iter_b64 = base64.b64encode(_iter_gz).decode('ascii')
        parts.append('<script id="iter-data-gz" type="application/gzip">'
                     '{}</script>\n'.format(_iter_b64))
        parts.append('<script>\n')
        parts.append('var ITER_DATA = null;\n')
        parts.append('(function(){\n')
        parts.append('  var src=document.getElementById("iter-data-gz").textContent;\n')
        parts.append('  var bin=atob(src);\n')
        parts.append('  var bytes=new Uint8Array(bin.length);\n')
        parts.append('  for(var i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);\n')
        parts.append('  var ds=new DecompressionStream("gzip");\n')
        parts.append('  var writer=ds.writable.getWriter();\n')
        parts.append('  writer.write(bytes);writer.close();\n')
        parts.append('  new Response(ds.readable).text().then(function(text){\n')
        parts.append('    ITER_DATA=JSON.parse(text);\n')
        parts.append('    document.getElementById("iter-data-gz").textContent="";\n')
        parts.append('    if(typeof renderTimeSeries==="function") renderTimeSeries();\n')
        parts.append('    if(typeof renderWaterfall==="function") renderWaterfall();\n')
        parts.append('  });\n')
        parts.append('})();\n')
        parts.append('</script>\n')
    else:
        parts.append('<script>var ITER_DATA = {};</script>\n')
    # Application code in a separate block
    parts.append('<script>\n')
    parts.append("""
function esc(s) {
  if (typeof s !== "string") return String(s);
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}
var plotConfig = { responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d","select2d"] };

function getFiltered() {
  var host = document.getElementById("filter-host").value;
  var bench = document.getElementById("filter-bench").value;
  return RESULTS.filter(function(r) {
    if (host !== "all" && r.hostname !== host) return false;
    var lbl = r.dtype ? r.benchmark + " (" + r.dtype + ")" : r.benchmark;
    if (bench !== "all" && lbl !== bench) return false;
    return true;
  });
}
function getMetric() { return document.getElementById("metric-select").value; }
function metricLabel(m) {
  return {"mean_val":"Performance","power_avg_w":"Power (W)","temp_max_c":"Temperature (\\u00b0C)",
    "sm_util_mean":"SM Utilization (%)","mem_bw_util_mean":"Mem BW Util (%)","gpu_clock_mean":"GPU Clock (MHz)"}[m] || m;
}
function metricUnit(data, m) {
  if (m === "mean_val" && data.length > 0) return data[0].unit || "";
  return {"power_avg_w":"W","temp_max_c":"\\u00b0C","sm_util_mean":"%","mem_bw_util_mean":"%","gpu_clock_mean":"MHz"}[m] || "";
}
// Vocabulary (T7, T11): labels name the statistic and its population.
// metricNoun is the sentence-case metric name; metricStatistic is the per-GPU
// statistic behind each summary column (temperature is the run max).
function metricNoun(m) {
  return {"mean_val":"performance","power_avg_w":"power","temp_max_c":"temperature",
    "sm_util_mean":"SM utilization","mem_bw_util_mean":"memory BW utilization","gpu_clock_mean":"GPU clock"}[m] || m;
}
function metricStatistic(m) { return m==="temp_max_c"?"max":"mean"; }
function isLowerBetter(m) { return m==="temp_max_c"||m==="power_avg_w"||m==="temp_gpu_C"||m==="power_W"; }
// T13: every value-ordered view sorts ascending left-to-right regardless of
// metric direction; colour and status carry good/bad. isLowerBetter above is
// kept for outlier direction and legend wording only (T14), never for order.
function ascByMean(getMean) { return function(a,b){ return getMean(a)-getMean(b); }; }
function arrStats(arr) {
  if (!arr.length) return {mean:0,std:0,min:0,max:0,median:0};
  var n=arr.length, mean=arr.reduce(function(s,v){return s+v;},0)/n;
  var std=Math.sqrt(arr.reduce(function(s,v){return s+(v-mean)*(v-mean);},0)/n);
  var sorted=arr.slice().sort(function(a,b){return a-b;});
  var median=n%2?sorted[(n-1)/2]:(sorted[Math.floor(n/2)-1]+sorted[Math.floor(n/2)])/2;
  return {mean:mean,std:std,min:sorted[0],max:sorted[n-1],median:median};
}
function quantile(sortedArr,q) {
  if (!sortedArr.length) return null;
  var pos=(sortedArr.length-1)*q, lo=Math.floor(pos), hi=Math.ceil(pos);
  return sortedArr[lo]+(sortedArr[hi]-sortedArr[lo])*(pos-lo);
}
// T21: colour domain per metric. Utilisation is a bounded percentage, so its
// scale is fixed 0-100; open-ended metrics use the p2-p98 of the non-null
// cells so a handful of zeros or spikes cannot flatten the rest.
var BOUNDED_PERCENT_METRICS={sm_util:true,mem_bw_util:true};
function colourDomain(metricKey,samples) {
  if (BOUNDED_PERCENT_METRICS[metricKey]) return [0,100];
  if (!samples.length) return [0,1];
  var sorted=samples.slice().sort(function(a,b){return a-b;});
  var lo=quantile(sorted,0.02), hi=quantile(sorted,0.98);
  if (hi<=lo) { lo=sorted[0]; hi=sorted[sorted.length-1]; }
  if (hi<=lo) hi=lo+1;
  return [lo,hi];
}
function mkOutlierSet() {
  // Sigma-based outlier detection for interactive exploration.
  // Complements the percentage-based detect_outliers() in Python
  // which drives CLI exit codes and static reports.
  var sigma=parseFloat(document.getElementById("outlier-sigma").value)||2;
  var metric=getMetric();
  var vals=RESULTS.map(function(r){return r[metric];}).filter(function(v){return v>0;});
  if(!vals.length) return {};
  var st=arrStats(vals);
  if(st.std<=0) return {};
  var lo=st.mean-sigma*st.std, hi=st.mean+sigma*st.std;
  // Floor: require at least 5% deviation from mean to be an outlier.
  // Prevents flagging normal variation in tight fleets.
  var minPct=5;
  var loFloor=st.mean*(1-minPct/100), hiFloor=st.mean*(1+minPct/100);
  lo=Math.min(lo,loFloor); hi=Math.max(hi,hiFloor);
  var lb=isLowerBetter(metric);
  var s={};
  RESULTS.forEach(function(r){
    var v=r[metric]; if(v<=0) return;
    if(v<lo) s[r.hostname+":"+r.gpu]="low";
    else if(v>hi) s[r.hostname+":"+r.gpu]="high";
  });
  return s;
}
function mkThrottleSet() {
  var s={};RESULTS.forEach(function(r){if(r.throttled)s[r.hostname+":"+r.gpu]=true;});return s;
}
function gpuStatus(r,oSet,tSet) {
  var k=r.hostname+":"+r.gpu;
  if(oSet[k])return "outlier-"+oSet[k];if(tSet[k])return "throttled";return "normal";
}
function statusColor(st) {
  return st==="outlier-low"?"#f09595":st==="outlier-high"?"#a78bfa":st==="throttled"?"#fbbf24":"#0072B2";
}
function axisTitle(m, unit) {
  var lbl=metricLabel(m);
  if(lbl.indexOf("(")!==-1) return lbl;
  return lbl+" ("+unit+")";
}
var baseLayout = {
  paper_bgcolor:"#0f1419",plot_bgcolor:"#1a1f26",
  font:{color:"#e8eaed",family:"system-ui, sans-serif",size:12},
  margin:{t:36,b:48,l:64,r:24},showlegend:false
};
function themeColors() {
  var s=getComputedStyle(document.documentElement);
  return {paper:s.getPropertyValue("--plot-paper").trim(),
    plot:s.getPropertyValue("--plot-bg").trim(),
    font:s.getPropertyValue("--plot-font").trim(),
    axis:s.getPropertyValue("--axis-font").trim(),
    hint:s.getPropertyValue("--hint").trim(),
    grid:s.getPropertyValue("--grid-color").trim(),
    zero:s.getPropertyValue("--zero-color").trim()};
}
function applyTheme() {
  var tc=themeColors();
  baseLayout.paper_bgcolor=tc.paper;
  baseLayout.plot_bgcolor=tc.plot;
  baseLayout.font.color=tc.font;
}
function tAxis(extra) {
  var tc=themeColors();
  var a={gridcolor:tc.grid,zerolinecolor:tc.zero};
  if(extra) for(var k in extra) a[k]=extra[k];
  return a;
}
function tAxisTitle(text) {
  return {text:text,font:{size:12,color:themeColors().axis}};
}
function initTheme() {
    var root=document.documentElement;
    var btn=document.getElementById("theme-toggle");
    var useLight=false;
    if (THEME_MODE === "light") {
        useLight=true;
    } else if (THEME_MODE === "auto") {
        var prefersDark=window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        useLight=!prefersDark;
    }
    root.classList.toggle("light", useLight);
    if (btn) {
        btn.textContent=useLight?"dark mode":"light mode";
    }
}
function toggleTheme() {
  var root=document.documentElement;
  var isLight=root.classList.toggle("light");
  document.getElementById("theme-toggle").textContent=isLight?"dark mode":"light mode";
  applyTheme();
  renderAll();
}

// ---- Chart 1: Fleet Map ----
function parseXname(hostname) {
  var m=hostname.match(/x(\\d+)c(\\d+)s(\\d+)b(\\d+)n(\\d+)/);
  if(!m) return null;
  return {cabinet:parseInt(m[1]),chassis:parseInt(m[2]),slot:parseInt(m[3]),board:parseInt(m[4]),node:parseInt(m[5])};
}
function valToColor(val, st, lowerBetter) {
  if (val===null||val===undefined) return "#1a1f26";
  // Map deviation from mean to a diverging blue (good) / red (bad) scale.
  // Use a tighter range so even small differences within the fleet are visible.
  var d = (val-st.mean)/(st.std||1);
  if (lowerBetter) d = -d;
  d = Math.max(-3, Math.min(3, d));
  // Neutral midpoint (near mean) is a visible dark slate, not invisible
  var midR=69,midG=75,midB=83;
  if (d >= 0) {
    var t=Math.min(d/3,1);
    return "rgb("+Math.round(midR-24*t)+","+Math.round(midG+55*t)+","+Math.round(midB+117*t)+")";
  } else {
    var t2=Math.min((-d)/3,1);
    return "rgb("+Math.round(midR+146*t2)+","+Math.round(midG-27*t2)+","+Math.round(midB-44*t2)+")";
  }
}
function renderFleetMap() {
  var container=document.getElementById("fleetmap-chart");
  var legendEl=document.getElementById("fleetmap-legend");
  var data=getFiltered(),metric=getMetric();
  if(!data.length){container.innerHTML="";legendEl.innerHTML="";return;}
  var unit=metricUnit(data,metric);
  var lowerBetter=isLowerBetter(metric);
  var oSet=mkOutlierSet(),tSet=mkThrottleSet();
  // Aggregate per-node mean
  var nodeVals={},nodeStatus={};
  data.forEach(function(r){
    var v=r[metric]; if(v<=0) return;
    if(!nodeVals[r.hostname]) nodeVals[r.hostname]={sum:0,n:0};
    nodeVals[r.hostname].sum+=v; nodeVals[r.hostname].n+=1;
    // Throttle status propagates from any GPU to the node
    var sk=r.hostname+":"+r.gpu;
    if(tSet[sk]&&nodeStatus[r.hostname]!=="outlier-low"&&nodeStatus[r.hostname]!=="outlier-high") nodeStatus[r.hostname]="throttled";
  });
  var allVals=[];
  Object.keys(nodeVals).forEach(function(h){allVals.push(nodeVals[h].sum/nodeVals[h].n);});
  var st=arrStats(allVals);
  function nodeMeanOf(h){ var nv=nodeVals[h]; return nv?nv.sum/nv.n:0; }
  // Propagate per-GPU outlier status to nodes (consistent with header card and other charts)
  data.forEach(function(r){
    var sk=r.hostname+":"+r.gpu;
    if(oSet[sk]){
      var cur=nodeStatus[r.hostname];
      if(cur!=="outlier-low"&&cur!=="outlier-high") nodeStatus[r.hostname]="outlier-"+oSet[sk];
    }
  });
  // Legend
  var lgHtml='<span style="display:inline-flex;align-items:center;gap:8px;flex-wrap:wrap">';
  lgHtml+='<span style="color:#d73027">'+(lowerBetter?'high (bad)':'low (bad)')+'</span>';
  lgHtml+='<span style="display:inline-block;width:100px;height:8px;border-radius:3px;background:linear-gradient(to right,#d73027,#454b53,#4575b4)"></span>';
  lgHtml+='<span style="color:#4575b4">'+(lowerBetter?'low (good)':'high (good)')+'</span>';
  lgHtml+='<span style="margin-left:8px;color:var(--muted)">mean: '+st.mean.toFixed(1)+' '+unit+'</span>';
  lgHtml+='<span style="display:inline-flex;align-items:center;gap:4px;margin-left:8px"><span style="display:inline-block;width:10px;height:10px;border:2px solid #f09595;border-radius:2px"></span><span style="color:var(--muted);font-size:10px">outlier \u2193</span></span>';
  lgHtml+='<span style="display:inline-flex;align-items:center;gap:4px"><span style="display:inline-block;width:10px;height:10px;border:2px solid #a78bfa;border-radius:2px"></span><span style="color:var(--muted);font-size:10px">outlier \u2191</span></span>';
  lgHtml+='<span style="display:inline-flex;align-items:center;gap:4px"><span style="display:inline-block;width:10px;height:10px;border:2px solid #fbbf24;border-radius:2px"></span><span style="color:var(--muted);font-size:10px">throttled</span></span>';
  lgHtml+='</span>';
  legendEl.innerHTML=lgHtml;
  // Determine grouping
  var allHosts=Object.keys(nodeVals);
  var xnames={},hasXnames=false;
  allHosts.forEach(function(h){var p=parseXname(h);if(p){xnames[h]=p;hasXnames=true;}});
  // Also check NODE_MAP for location grouping
  var hasNodeMap=NODE_MAP&&Object.keys(NODE_MAP).length>0;
  function mkSquare(host,size) {
    var nv=nodeVals[host]; var val=nv?nv.sum/nv.n:null;
    var color=valToColor(val,st,lowerBetter);
    var devPct=val!==null&&st.mean>0?((val-st.mean)/st.mean*100).toFixed(1):"0.0";
    var status=nodeStatus[host]||"normal";
    var border=status==="outlier-low"?"2px solid #f09595":status==="outlier-high"?"2px solid #a78bfa":status==="throttled"?"2px solid #fbbf24":"1px solid rgba(255,255,255,0.06)";
    var tip=esc(host)+(hasNodeMap&&NODE_MAP[host]?" ["+esc(NODE_MAP[host])+"]":"")+
      "\\n"+(val!=null?val.toFixed(1)+" "+unit:"N/A")+
      "\\n"+devPct+"% vs fleet"+
      (status!=="normal"?"\\n"+status.toUpperCase():"");
    return '<div title="'+tip+'" class="nc" style="width:'+size+'px;height:'+size+'px;background:'+color+';border:'+border+'"></div>';
  }
  // Topology-faithful cabinet layout: mirrors physical Cray EX front view.
  // Adapts to any chassis/slot/board counts found in the data.
  // Paired chassis (even=left, odd=right) detected automatically.
  function buildCabTopo(cabNodes, xnMap, sqSz) {
    var chSet={},slSet={},bdSet={};
    cabNodes.forEach(function(h){
      var x=xnMap[h]; if(!x) return;
      chSet[x.chassis]=true; slSet[x.slot]=true; bdSet[x.board]=true;
    });
    var chs=Object.keys(chSet).map(Number).sort(function(a,b){return a-b;});
    var sls=Object.keys(slSet).map(Number).sort(function(a,b){return a-b;});
    var bds=Object.keys(bdSet).map(Number).sort(function(a,b){return a-b;});
    if(!chs.length) return '';
    // Build lookup: "chassis,slot,board" -> hostname
    var lk={};
    cabNodes.forEach(function(h){
      var x=xnMap[h]; if(!x) return;
      lk[x.chassis+','+x.slot+','+x.board]=h;
    });
    // Detect paired chassis layout (Cray EX: even/odd pairs share rectifier shelf)
    var paired=chs.length>=2&&chs.length%2===0;
    if(paired){ for(var pi=0;pi<chs.length;pi+=2){ if(chs[pi+1]-chs[pi]!==1||chs[pi]%2!==0){paired=false;break;} } }
    var g=sqSz>12?3:2; // gap between squares
    var sg=g*3;         // gap between left/right chassis pair
    function chassisGrid(ch){
      var r='<div style="display:grid;grid-template-columns:repeat('+sls.length+','+sqSz+'px);gap:'+g+'px">';
      for(var bi=bds.length-1;bi>=0;bi--){ var b=bds[bi]; sls.forEach(function(s){
        var host=lk[ch+','+s+','+b];
        if(host) r+=mkSquare(host,sqSz);
        else r+='<div style="width:'+sqSz+'px;height:'+sqSz+'px"></div>';
      }); }
      r+='</div>';
      return r;
    }
    var out='';
    if(paired){
      // Shelves top-to-bottom matching physical front view (highest pair first)
      var nSh=chs.length/2;
      for(var sh=nSh-1;sh>=0;sh--){
        out+='<div style="display:flex;gap:'+sg+'px;margin-bottom:'+g+'px">';
        out+=chassisGrid(chs[sh*2]);
        out+=chassisGrid(chs[sh*2+1]);
        out+='</div>';
      }
    } else {
      // Unpaired: stack chassis rows top-to-bottom
      for(var ci=chs.length-1;ci>=0;ci--){
        out+='<div style="margin-bottom:'+g+'px">';
        out+=chassisGrid(chs[ci]);
        out+='</div>';
      }
    }
    return out;
  }
  var html='';
  if (hasXnames) {
    // Cray EX cabinet view: group by cabinet, arrange nodes inside
    var byCabinet={};
    allHosts.forEach(function(h){
      var x=xnames[h]; if(!x) return;
      if(!byCabinet[x.cabinet]) byCabinet[x.cabinet]=[];
      byCabinet[x.cabinet].push(h);
    });
    var cabList=Object.keys(byCabinet).map(Number).sort(function(a,b){return a-b;});
    var totalNodes=allHosts.length;
    var sqSize=totalNodes>500?10:totalNodes>200?14:totalNodes>80?18:22;
    html+='<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center">';
    cabList.forEach(function(cab){
      html+='<div style="background:var(--surface2);border-radius:8px;padding:8px 10px">';
      html+='<div style="font-size:10px;color:var(--muted);margin-bottom:5px;font-weight:600;text-align:center">x'+cab+'</div>';
      html+=buildCabTopo(byCabinet[cab], xnames, sqSize);
      html+='</div>';
    });
    // Non-xname stragglers
    var nonXname=allHosts.filter(function(h){return !xnames[h];});
    if(nonXname.length){
      nonXname.sort();
      var cols=Math.ceil(Math.sqrt(nonXname.length));
      html+='<div style="background:var(--surface2);border-radius:8px;padding:8px 10px">';
      html+='<div style="font-size:10px;color:var(--muted);margin-bottom:5px;font-weight:600;text-align:center">other</div>';
      html+='<div style="display:grid;grid-template-columns:repeat('+cols+','+sqSize+'px);gap:3px;justify-content:center">';
      nonXname.forEach(function(h){ html+=mkSquare(h,sqSize); });
      html+='</div></div>';
    }
    html+='</div>';
  } else if (hasNodeMap) {
    // Parse xname structure from NODE_MAP location values for cabinet grouping
    var locXnames={};
    allHosts.forEach(function(h){
      var loc=NODE_MAP[h]||"";
      var px=parseXname(loc);
      if(px) locXnames[h]=px;
    });
    var hasLocXnames=Object.keys(locXnames).length>0;
    if(hasLocXnames) {
      // Group by cabinet from parsed location xnames
      var byCab={};
      allHosts.forEach(function(h){
        var x=locXnames[h];
        var cab=x?x.cabinet:"other";
        if(!byCab[cab]) byCab[cab]=[];
        byCab[cab].push(h);
      });
      var cabKeys=Object.keys(byCab).sort(function(a,b){
        if(a==="other") return 1; if(b==="other") return -1;
        return parseInt(a)-parseInt(b);
      });
      var totalNodes=allHosts.length;
      var sqSize=totalNodes>500?10:totalNodes>200?14:totalNodes>80?18:22;
      html+='<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center">';
      cabKeys.forEach(function(cab){
        var cabHosts=byCab[cab];
        html+='<div style="background:var(--surface2);border-radius:8px;padding:8px 10px">';
        html+='<div style="font-size:10px;color:var(--muted);margin-bottom:5px;font-weight:600;text-align:center">x'+cab+'</div>';
        if(cab==="other"){
          var cols=Math.ceil(Math.sqrt(cabHosts.length));
          cabHosts.sort();
          html+='<div style="display:grid;grid-template-columns:repeat('+cols+','+sqSize+'px);gap:3px;justify-content:center">';
          cabHosts.forEach(function(h){ html+=mkSquare(h,sqSize); });
          html+='</div>';
        } else {
          html+=buildCabTopo(cabHosts, locXnames, sqSize);
        }
        html+='</div>';
      });
      html+='</div>';
    } else {
      // NODE_MAP without parseable xnames: flat grid, ascending node mean (T15)
      var totalNodes=allHosts.length;
      var sqSize=totalNodes>500?10:totalNodes>200?14:totalNodes>80?18:22;
      allHosts.sort(ascByMean(nodeMeanOf));
      var cols=Math.min(20,Math.ceil(Math.sqrt(totalNodes)*1.5));
      html+='<div style="display:grid;grid-template-columns:repeat('+cols+','+sqSize+'px);gap:3px;justify-content:center">';
      allHosts.forEach(function(h){ html+=mkSquare(h,sqSize); });
      html+='</div>';
    }
  } else {
    // No grouping info: flat grid, ascending node mean (T15)
    var totalNodes=allHosts.length;
    var sqSize=totalNodes>500?10:totalNodes>200?14:totalNodes>80?18:22;
    allHosts.sort(ascByMean(nodeMeanOf));
    var cols=Math.min(20,Math.ceil(Math.sqrt(totalNodes)*1.5));
    html+='<div style="display:grid;grid-template-columns:repeat('+cols+','+sqSize+'px);gap:3px;justify-content:center">';
    allHosts.forEach(function(h){ html+=mkSquare(h,sqSize); });
    html+='</div>';
  }
  container.innerHTML=html;
}

// ---- Chart 2: Fleet Distribution (Histogram) ----
function renderDistribution() {
  var data = getFiltered(), metric = getMetric();
  var vals = data.map(function(r){return r[metric];}).filter(function(v){return v>0;});
  if (!vals.length) { Plotly.purge("distribution-chart"); return; }
  var st = arrStats(vals);
  var oSet = mkOutlierSet(), tSet = mkThrottleSet();
  var unit = metricUnit(data, metric);
  var histTrace = {
    x:vals, type:"histogram",
    marker:{color:"rgba(0,114,178,0.35)",line:{color:"#0072B2",width:1}},
    hovertemplate:metricLabel(metric)+": %{x:.1f} "+unit+"<br>Count: %{y}<extra></extra>",
    name:"histogram",showlegend:false,yaxis:"y"
  };
  var stripX=[],stripColors=[],stripHover=[];
  data.forEach(function(r) {
    var v=r[metric]; if(v<=0)return;
    stripX.push(v);
    var status=gpuStatus(r,oSet,tSet);
    stripColors.push(statusColor(status));
    var dev=st.mean>0?((v-st.mean)/st.mean*100).toFixed(1):"0.0";
    var statusLabel=status==="outlier-low"?"OUTLIER \u2193":status==="outlier-high"?"OUTLIER \u2191":status==="throttled"?"THROTTLED":"";
    stripHover.push(esc(r.hostname)+" GPU"+r.gpu+"<br>"+v.toFixed(1)+" "+unit+
      "<br>"+dev+"% vs fleet"+(statusLabel?"<br><b>"+statusLabel+"</b>":""));
  });
  var stripTrace = {
    x:stripX,y:stripX.map(function(){return 0;}),
    type:"scatter",mode:"markers",
    marker:{color:stripColors,size:8,symbol:"line-ns",line:{width:2,color:stripColors}},
    hovertext:stripHover,hoverinfo:"text",
    yaxis:"y2",showlegend:false,cliponaxis:false
  };
  var shapes = [
    {type:"line",x0:st.mean,x1:st.mean,y0:0,y1:1,yref:"paper",line:{color:"#5dcaa5",width:2}},
    {type:"line",x0:st.mean-st.std,x1:st.mean-st.std,y0:0,y1:1,yref:"paper",line:{color:"rgba(251,191,36,0.5)",width:1,dash:"dash"}},
    {type:"line",x0:st.mean+st.std,x1:st.mean+st.std,y0:0,y1:1,yref:"paper",line:{color:"rgba(251,191,36,0.5)",width:1,dash:"dash"}},
    {type:"line",x0:st.mean-2*st.std,x1:st.mean-2*st.std,y0:0,y1:1,yref:"paper",line:{color:"rgba(240,149,149,0.4)",width:1,dash:"dot"}},
    {type:"line",x0:st.mean+2*st.std,x1:st.mean+2*st.std,y0:0,y1:1,yref:"paper",line:{color:"rgba(240,149,149,0.4)",width:1,dash:"dot"}}
  ];
  var ann = [
    {x:st.mean,y:1,yref:"paper",text:st.mean.toFixed(1),
     showarrow:false,font:{size:10,color:"#5dcaa5"},yanchor:"bottom",xanchor:"left"}
  ];
  // Compute excess kurtosis (0 = normal)
  var kurt=0;
  if (st.std > 0 && vals.length > 3) {
    var m4=vals.reduce(function(s,v){var d=v-st.mean;return s+d*d*d*d;},0)/vals.length;
    kurt=m4/(st.std*st.std*st.std*st.std)-3;
  }
  // Stats legend: upper-right, compact
  var tc=themeColors();
  var legendLines=[
    "mean = "+st.mean.toFixed(2)+" "+unit,
    "\\u03c3 = "+st.std.toFixed(2)+" "+unit,
    "kurtosis = "+kurt.toFixed(2),
    "n = "+vals.length
  ];
  ann.push({
    x:1,y:1,xref:"paper",yref:"paper",xanchor:"right",yanchor:"top",
    text:legendLines.join("<br>"),showarrow:false,
    font:{size:9,color:tc.axis,family:"monospace"},
    align:"right",
    bgcolor:tc.plot,borderpad:4,opacity:0.85
  });
  var layout = Object.assign({},baseLayout,{
    shapes:shapes,annotations:ann,
    xaxis:tAxis({title:{text:axisTitle(metric,unit),font:{size:10,color:themeColors().hint},standoff:2}}),
    yaxis:tAxis({title:tAxisTitle("GPU Count"),domain:[0.18,1]}),
    yaxis2:{domain:[0,0.08],showticklabels:false,showgrid:false,zeroline:false,fixedrange:true},
    bargap:0.04,height:400,margin:{t:36,b:48,l:64,r:24}
  });
  Plotly.newPlot("distribution-chart",[histTrace,stripTrace],layout,plotConfig);
}

// ---- Chart 3: Power vs Performance ----
function renderEfficiency() {
  var data = getFiltered(), metric = getMetric();
  var oSet = mkOutlierSet(), tSet = mkThrottleSet();
  var hasPower = data.some(function(r){return r.power_avg_w>0;});
  var sec = document.getElementById("efficiency-section");
  if (!hasPower || data.length < 2) { sec.style.display = "none"; return; }
  sec.style.display = "block";
  var unit = metricUnit(data, metric);
  var effTitle=document.getElementById("efficiency-title");
  if(effTitle) effTitle.textContent="Power vs "+metricLabel(metric);
  var xs=[],ys=[],colors=[],texts=[],sizes=[];
  data.forEach(function(r) {
    if (r.power_avg_w<=0) return;
    var v=r[metric]; if(v<=0) return;
    xs.push(r.power_avg_w); ys.push(v);
    colors.push(r.temp_max_c>0?r.temp_max_c:null);
    var status=gpuStatus(r,oSet,tSet);
    var sLabel=status==="outlier-low"?"OUTLIER \u2193":status==="outlier-high"?"OUTLIER \u2191":status==="throttled"?"THROTTLED":"";
    texts.push(esc(r.hostname)+" GPU"+r.gpu+
      "<br>Power: "+r.power_avg_w.toFixed(0)+" W"+
      "<br>"+metricLabel(metric)+": "+v.toFixed(1)+" "+unit+
      "<br>Temp: "+(r.temp_max_c>0?r.temp_max_c.toFixed(0)+"\\u00b0C":"N/A")+
      (sLabel?"<br><b>"+sLabel+"</b>":""));
    sizes.push(status!=="normal"?10:6);
  });
  var hasTemp = colors.some(function(c){return c!==null;});
  var trace = {
    x:xs,y:ys,type:"scatter",mode:"markers",
    marker:{
      color:hasTemp?colors:"#0072B2",
      colorscale:hasTemp?[[0,"#4575b4"],[0.5,"#fbbf24"],[1,"#d73027"]]:undefined,
      showscale:hasTemp,
      colorbar:hasTemp?{title:{text:"Temp (\\u00b0C)",side:"right"},thickness:12,len:0.5,
        tickfont:{size:10,color:themeColors().axis},titlefont:{size:10,color:themeColors().axis}}:undefined,
      size:sizes,opacity:0.85
    },
    hovertext:texts,hoverinfo:"text",showlegend:false
  };
  var pSt=arrStats(xs),mSt=arrStats(ys);
  var shapes = [
    {type:"line",x0:pSt.mean,x1:pSt.mean,y0:0,y1:1,yref:"paper",line:{color:"rgba(255,255,255,0.15)",width:1,dash:"dash"}},
    {type:"line",y0:mSt.mean,y1:mSt.mean,x0:0,x1:1,xref:"paper",line:{color:"rgba(255,255,255,0.15)",width:1,dash:"dash"}}
  ];
  var ann = [
    {x:pSt.mean,y:1.02,yref:"paper",text:"mean power",showarrow:false,font:{size:9,color:"#5f6368"},yanchor:"bottom"},
    {x:0,xref:"paper",y:mSt.mean,text:" mean "+metricNoun(metric),showarrow:false,font:{size:9,color:"#5f6368"},xanchor:"left"}
  ];
  var layout = Object.assign({},baseLayout,{
    shapes:shapes,annotations:ann,
    xaxis:tAxis({title:tAxisTitle("Mean power (W)")}),
    yaxis:tAxis({title:tAxisTitle(axisTitle(metric,unit))}),
    height:420,margin:{t:36,b:56,l:72,r:80}
  });
  Plotly.newPlot("efficiency-chart",[trace],layout,plotConfig);
}

// ---- Chart 4: Node Variability (Strip) ----
function renderStrip() {
  var data = getFiltered(), metric = getMetric();
  var oSet = mkOutlierSet(), tSet = mkThrottleSet();
  if (!data.length) { Plotly.purge("strip-chart"); return; }
  var unit = metricUnit(data, metric);
  var byNode = {};
  data.forEach(function(r) {
    var v=r[metric]; if(v<=0) return;
    if(!byNode[r.hostname]) byNode[r.hostname]=[];
    byNode[r.hostname].push(r);
  });
  var nodeNames = Object.keys(byNode);
  var nodeMeans = {};
  nodeNames.forEach(function(h) {
    var vals=byNode[h].map(function(r){return r[metric];});
    nodeMeans[h]=arrStats(vals).mean;
  });
  // T16: ascending node mean; truncation below keeps both ends + abnormal nodes.
  nodeNames.sort(ascByMean(function(h){ return nodeMeans[h]; }));
  var MAX_DISPLAY = 60;
  var truncated=false, displayNodes=nodeNames;
  if (nodeNames.length > MAX_DISPLAY) {
    var essential={};
    data.forEach(function(r){var s=gpuStatus(r,oSet,tSet);if(s!=="normal")essential[r.hostname]=true;});
    var nEnds = Math.min(15, Math.floor(MAX_DISPLAY/3));
    var top=nodeNames.slice(0,nEnds), bottom=nodeNames.slice(-nEnds);
    var remain=MAX_DISPLAY-top.length-bottom.length;
    var mid=nodeNames.filter(function(h){return essential[h]&&top.indexOf(h)===-1&&bottom.indexOf(h)===-1;}).slice(0,remain);
    displayNodes=[].concat(top,mid,bottom);
    var seen={};
    displayNodes=displayNodes.filter(function(h){if(seen[h])return false;seen[h]=true;return true;});
    truncated=true;
  }
  var categories={normal:[],"outlier-low":[],"outlier-high":[],throttled:[]};
  displayNodes.forEach(function(host,idx) {
    (byNode[host]||[]).forEach(function(r) {
      var v=r[metric]; if(v<=0) return;
      var st=gpuStatus(r,oSet,tSet);
      if(!categories[st]) categories[st]=[];
      categories[st].push({x:idx,y:v,host:host,gpu:r.gpu});
    });
  });
  var allVals=data.map(function(r){return r[metric];}).filter(function(v){return v>0;});
  var fleetMean=arrStats(allVals).mean;
  var traces = [];
  [["normal","#0072B2","Normal"],["throttled","#fbbf24","Throttled"],["outlier-low","#f09595","Outlier \u2193"],["outlier-high","#a78bfa","Outlier \u2191"]].forEach(function(cfg) {
    var pts=categories[cfg[0]];
    if(!pts.length)return;
    traces.push({
      x:pts.map(function(p){return p.x;}),y:pts.map(function(p){return p.y;}),
      type:"scatter",mode:"markers",
      marker:{color:cfg[1],size:7,opacity:0.8,line:{width:0.5,color:"rgba(255,255,255,0.3)"}},
      name:cfg[2],
      hovertext:pts.map(function(p){
        var dev=fleetMean>0?((p.y-fleetMean)/fleetMean*100).toFixed(1):"0.0";
        return esc(p.host)+" GPU"+p.gpu+"<br>"+p.y.toFixed(1)+" "+unit+"<br>"+dev+"% vs fleet";
      }),
      hoverinfo:"text",showlegend:true
    });
  });
  var shapes=[{type:"line",y0:fleetMean,y1:fleetMean,x0:-0.5,x1:displayNodes.length-0.5,
    line:{color:"#5dcaa5",width:1.5,dash:"dash"}}];
  var ann=[{x:displayNodes.length-0.5,y:fleetMean,text:"fleet mean: "+fleetMean.toFixed(1)+" "+unit,
    showarrow:false,font:{size:10,color:"#5dcaa5"},xanchor:"left"}];
  if (truncated) {
    ann.push({x:0.5,y:1.02,xref:"paper",yref:"paper",
      text:"showing "+displayNodes.length+" of "+nodeNames.length+" nodes (lowest/highest + outliers)",
      showarrow:false,font:{size:10,color:"#5f6368"}});
  }
  var maxHostLen=displayNodes.reduce(function(mx,n){return Math.max(mx,n.length);},0);
  var dn=displayNodes.length;
  var layout = Object.assign({},baseLayout,{
    shapes:shapes,annotations:ann,showlegend:true,
    legend:{bgcolor:"rgba(0,0,0,0)",font:{color:themeColors().axis,size:11},orientation:"h",x:0,y:1.12},
    xaxis:tAxis({title:{text:"Node, ascending node mean ("+displayNodes.length+(truncated?" of "+nodeNames.length:"")+" nodes)",standoff:4,font:{size:11,color:themeColors().axis}},
      tickmode:"array",tickvals:dn>50?[]:displayNodes.map(function(_,i){return i;}),
      ticktext:dn>50?[]:displayNodes,tickangle:dn>25?-60:-45,
      tickfont:{size:dn>30?9:10}}),
    yaxis:tAxis({title:tAxisTitle(axisTitle(metric,unit))}),
    height:420,margin:{t:52,b:dn>50?56:Math.min(maxHostLen*5+40,140),l:72,r:80}
  });
  Plotly.newPlot("strip-chart",traces,layout,plotConfig);
}

// Rolling mean smoother for individual traces (removes iteration noise after downsampling).
// T19: null samples are gaps. They are skipped inside the window; a window
// with no samples stays null so Plotly breaks the line instead of drawing 0.
function smoothArr(arr,w){
  if(!w||w<2||arr.length<=w) return arr;
  var out=[],hw=Math.floor(w/2);
  for(var i=0;i<arr.length;i++){
    var lo=Math.max(0,i-hw),hi=Math.min(arr.length-1,i+hw),s=0,n=0;
    for(var j=lo;j<=hi;j++){ if(arr[j]!=null){ s+=arr[j]; n++; } }
    out.push(n?s/n:null);
  }
  return out;
}
// Downsample iteration-level arrays when there are too many points.
// Bins adjacent iterations; uses minFn/maxFn/medFn per bin for envelopes vs lines.
var DS_MAX=600;
function dsIters(iters,arrs,modes){
  var n=iters.length;
  if(n<=DS_MAX) return {x:iters,ya:arrs};
  var binSz=Math.ceil(n/DS_MAX);
  var ox=[],oa=arrs.map(function(){return[];});
  for(var b=0;b<n;b+=binSz){
    var end=Math.min(b+binSz,n);
    ox.push(iters[b]);
    for(var ai=0;ai<arrs.length;ai++){
      var sl=arrs[ai].slice(b,end).filter(function(v){return v!=null;});
      if(!sl.length){ oa[ai].push(null); continue; }  // T19: all-gap bin stays a gap
      sl.sort(function(a,b){return a-b;});
      var m=modes[ai]||"median";
      if(m==="min") oa[ai].push(sl[0]);
      else if(m==="max") oa[ai].push(sl[sl.length-1]);
      else oa[ai].push(sl.length%2?sl[(sl.length-1)/2]:(sl[Math.floor(sl.length/2)-1]+sl[Math.floor(sl.length/2)])/2);
    }
  }
  return {x:ox,ya:oa};
}
// ---- Chart 5: Iteration Time Series ----
function renderTimeSeries() {
  var sec = document.getElementById("timeseries-section");
  if (!ITER_DATA || Object.keys(ITER_DATA).length === 0) { sec.style.display="none"; return; }
  sec.style.display = "block";
  var host=document.getElementById("filter-host").value;
  var bench=document.getElementById("filter-bench").value;
  var metric=getMetric();
  var unit = metricUnit(RESULTS, metric);
  var iterMetric={"mean_val":"performance","power_avg_w":"power_W","temp_max_c":"temp_gpu_C",
    "sm_util_mean":"sm_util","mem_bw_util_mean":"mem_bw_util","gpu_clock_mean":"gpu_clock"};
  var mKey=iterMetric[metric]||"performance";
  var oSet=mkOutlierSet(),tSet=mkThrottleSet();
  var allSeries=[];
  Object.keys(ITER_DATA).forEach(function(key) {
    var p=key.split(":"),h=p[0],g=p[1],b=p[2],d=p[3];
    var lbl=d?b+" ("+d+")":b;
    if(host!=="all"&&h!==host) return;
    if(bench!=="all"&&lbl!==bench) return;
    var rows=ITER_DATA[key];
    var ys=rows.map(function(r){return r[mKey];}).filter(function(v){return v!=null;});  // T19
    if(!ys.length) return;
    var status="normal";
    if(oSet[h+":"+g])status="outlier-"+oSet[h+":"+g]; else if(tSet[h+":"+g])status="throttled";
    allSeries.push({host:h,gpu:g,label:lbl,rows:rows,mKey:mKey,status:status});
  });
  if (!allSeries.length) { sec.style.display="none"; return; }
  var traces=[];
  if (allSeries.length > 15) {
    var iterBuckets={};
    allSeries.forEach(function(s){
      s.rows.forEach(function(r){
        var v=r[mKey]; if(v==null) return;  // T19: a measured 0 is data
        if(!iterBuckets[r.iteration]) iterBuckets[r.iteration]=[];
        iterBuckets[r.iteration].push(v);
      });
    });
    var rawIters=Object.keys(iterBuckets).map(Number).sort(function(a,b){return a-b;});
    var rawMed=[],rawP10=[],rawP90=[];
    rawIters.forEach(function(i){
      var vals=iterBuckets[i].slice().sort(function(a,b){return a-b;});
      var n=vals.length;
      rawMed.push(n%2?vals[(n-1)/2]:(vals[Math.floor(n/2)-1]+vals[Math.floor(n/2)])/2);
      rawP10.push(vals[Math.max(0,Math.floor(n*0.1))]);
      rawP90.push(vals[Math.min(n-1,Math.floor(n*0.9))]);
    });
    var ds=dsIters(rawIters,[rawP10,rawMed,rawP90],["min","median","max"]);
    var iters=ds.x,p10s=ds.ya[0],medians=ds.ya[1],p90s=ds.ya[2];
    // Adaptive envelope opacity: more opaque for smaller fleets where band is narrow
    var envAlpha=allSeries.length>500?0.12:allSeries.length>100?0.18:0.28;
    traces.push({
      x:iters.concat(iters.slice().reverse()),
      y:p90s.concat(p10s.slice().reverse()),
      type:"scatter",fill:"toself",mode:"none",
      fillcolor:"rgba(0,114,178,"+envAlpha+")",name:"p10\\u2013p90 envelope ("+allSeries.length+" GPUs)",showlegend:true,hoverinfo:"skip"
    });
    // Envelope edge lines for visibility
    traces.push({x:iters,y:p90s,type:"scatter",mode:"lines",line:{color:"rgba(0,114,178,0.35)",width:0.5},showlegend:false,hoverinfo:"skip"});
    traces.push({x:iters,y:p10s,type:"scatter",mode:"lines",line:{color:"rgba(0,114,178,0.35)",width:0.5},showlegend:false,hoverinfo:"skip"});
    traces.push({
      x:iters,y:medians,type:"scatter",mode:"lines",
      line:{color:"#0072B2",width:2},name:"fleet median across GPUs",
      hovertemplate:"Iteration %{x}<br>fleet median: %{y:.1f} "+unit+"<extra></extra>"
    });
    // Only show individual traces for outlier/throttled GPUs
    var abnormal=allSeries.filter(function(s){return s.status!=="normal";});
    var nOut=abnormal.filter(function(s){return s.status.indexOf("outlier")===0;}).length;
    var nThr=abnormal.filter(function(s){return s.status==="throttled";}).length;
    // Compute fleet mean for deviation labels
    var fleetSum=0,fleetN=0;
    allSeries.forEach(function(s){
      s.rows.forEach(function(r){var v=r[mKey];if(v!=null){fleetSum+=v;fleetN++;}});
    });
    var fleetMean=fleetN>0?fleetSum/fleetN:0;
    if (abnormal.length <= 20) {
      abnormal.forEach(function(s){
        var sx=s.rows.map(function(r){return r.iteration;});
        var sy=s.rows.map(function(r){return r[mKey];});
        if(sx.length>DS_MAX){var sd=dsIters(sx,[sy],["median"]);sx=sd.x;sy=sd.ya[0];}
        sy=smoothArr(sy,15);
        // Compute GPU mean deviation for legend label
        var gSum=0,gN=0;
        s.rows.forEach(function(r){var v=r[mKey];if(v!=null){gSum+=v;gN++;}});
        var gMean=gN>0?gSum/gN:0;
        var devPct=fleetMean>0?((gMean-fleetMean)/fleetMean*100).toFixed(1):"?";
        var devStr=(parseFloat(devPct)>0?"+":"")+devPct+"%";
        var statusTag=s.status==="outlier-low"?"outlier \u2193":s.status==="outlier-high"?"outlier \u2191":"throttled";
        traces.push({
          x:sx,y:sy,
          type:"scatter",mode:"lines",
          line:{color:statusColor(s.status),width:1.5,dash:s.status==="throttled"?"dash":"solid"},
          name:esc(s.host)+":GPU"+s.gpu+" ("+statusTag+", "+devStr+")",
          hovertemplate:esc(s.host)+" GPU"+s.gpu+"<br>"+metricLabel(metric)+": %{y:.1f} "+unit+"<extra></extra>"
        });
      });
    } else {
      // Too many abnormal traces - show percentile bands instead of individual lines
      var abnBuckets={out:{},thr:{}};
      abnormal.forEach(function(s){
        var bucket=s.status.indexOf("outlier")===0?abnBuckets.out:abnBuckets.thr;
        s.rows.forEach(function(r){
          var v=r[mKey]; if(v==null) return;  // T19
          if(!bucket[r.iteration]) bucket[r.iteration]=[];
          bucket[r.iteration].push(v);
        });
      });
      // cfg: [bucket, line colour, fill colour, population noun, GPU count]
      [["out","#f09595","rgba(240,149,149,0.12)","outlier",nOut],
       ["thr","#fbbf24","rgba(251,191,36,0.12)","throttled",nThr]].forEach(function(cfg){
        var bk=abnBuckets[cfg[0]],population=cfg[3],nGpus=cfg[4];
        var rawBi=Object.keys(bk).map(Number).sort(function(a,b){return a-b;});
        if(!rawBi.length) return;
        var rawBp10=[],rawBp50=[],rawBp90=[];
        rawBi.forEach(function(i){
          var vs=bk[i].slice().sort(function(a,b){return a-b;});
          var n=vs.length;
          rawBp10.push(vs[Math.max(0,Math.floor(n*0.1))]);
          rawBp50.push(n%2?vs[(n-1)/2]:(vs[Math.floor(n/2)-1]+vs[Math.floor(n/2)])/2);
          rawBp90.push(vs[Math.min(n-1,Math.floor(n*0.9))]);
        });
        var abd=dsIters(rawBi,[rawBp10,rawBp50,rawBp90],["min","median","max"]);
        var bi=abd.x,bp10=abd.ya[0],bp50=abd.ya[1],bp90=abd.ya[2];
        traces.push({
          x:bi.concat(bi.slice().reverse()),
          y:bp90.concat(bp10.slice().reverse()),
          type:"scatter",fill:"toself",mode:"none",
          fillcolor:cfg[2],
          name:nGpus+" "+population+" GPU(s) p10\\u2013p90",showlegend:true,hoverinfo:"skip"
        });
        traces.push({
          x:bi,y:bp50,type:"scatter",mode:"lines",
          line:{color:cfg[1],width:1.5},name:population+" GPU median ("+nGpus+" GPUs)",
          hovertemplate:"Iteration %{x}<br>"+population+" GPU median: %{y:.1f} "+unit+"<extra></extra>"
        });
      });
    }
  } else {
    allSeries.forEach(function(s){
      var sx=s.rows.map(function(r){return r.iteration;});
      var sy=s.rows.map(function(r){return r[mKey];});
      if(sx.length>DS_MAX){var sd=dsIters(sx,[sy],["median"]);sx=sd.x;sy=sd.ya[0];}
      traces.push({
        x:sx,y:sy,
        type:"scatter",mode:"lines",
        line:{color:statusColor(s.status),width:s.status!=="normal"?2:1.2},
        name:esc(s.host)+":GPU"+s.gpu,
        hovertemplate:esc(s.host)+" GPU"+s.gpu+"<br>"+metricLabel(metric)+": %{y:.1f} "+unit+"<extra></extra>"
      });
    });
  }
  var layout=Object.assign({},baseLayout,{
    showlegend:true,legend:{bgcolor:"rgba(0,0,0,0)",font:{color:themeColors().axis,size:11},orientation:"h",x:0,y:1.12},
    xaxis:tAxis({title:tAxisTitle("Iteration")}),
    yaxis:tAxis({title:tAxisTitle(axisTitle(metric,unit))}),
    height:400,margin:{t:52,b:56,l:72,r:24}
  });
  Plotly.newPlot("timeseries-chart",traces,layout,plotConfig);
}

// ---- Chart 6: Fleet Waterfall ----
function renderWaterfall() {
  var sec = document.getElementById("waterfall-section");
  if (!ITER_DATA || Object.keys(ITER_DATA).length === 0) { sec.style.display="none"; return; }
  sec.style.display = "block";
  var host=document.getElementById("filter-host").value;
  var bench=document.getElementById("filter-bench").value;
  // Waterfall has its own metric selector (iter-level keys), independent of summary metric
  var wfMetric=document.getElementById("waterfall-metric").value;
  // Sentence-case nouns: they are composed into title/hover/axis text (T10).
  var wfMetricLabels={"performance":"performance","temp_gpu_C":"temperature",
    "gpu_clock":"GPU clock","sm_util":"SM utilization",
    "mem_bw_util":"memory BW utilization","power_W":"power"};
  var wfMetricUnits={"performance":"","temp_gpu_C":"\u00b0C",
    "gpu_clock":"MHz","sm_util":"%","mem_bw_util":"%","power_W":"W"};
  var mKey=wfMetric;
  var mLabel=wfMetricLabels[wfMetric]||wfMetric;
  // For performance, derive unit from summary data
  var metric=getMetric();
  var unit=wfMetric==="performance"?metricUnit(RESULTS,metric):wfMetricUnits[wfMetric];
  var oSet=mkOutlierSet(),tSet=mkThrottleSet();
  var wfTitle=document.getElementById("waterfall-title");
  if(wfTitle) wfTitle.textContent="Fleet Waterfall \u2014 mean "+mLabel+" per iteration";
  // Gather all GPU series
  var gpuSeries=[];
  Object.keys(ITER_DATA).forEach(function(key) {
    var p=key.split(":"),h=p[0],g=parseInt(p[1]),b=p[2],d=p[3];
    var lbl=d?b+" ("+d+")":b;
    if(host!=="all"&&h!==host) return;
    if(bench!=="all"&&lbl!==bench) return;
    var rows=ITER_DATA[key];
    var vals=rows.map(function(r){return r[mKey];});
    var valid=vals.filter(function(v){return v!=null;});  // T19: zeros count, nulls are gaps
    if(!valid.length) return;
    var mean=valid.reduce(function(s,v){return s+v;},0)/valid.length;
    var status="normal";
    if(oSet[h+":"+g])status="outlier-"+oSet[h+":"+g]; else if(tSet[h+":"+g])status="throttled";
    var loc=NODE_MAP&&NODE_MAP[h]?NODE_MAP[h]:"";
    var xn=parseXname(h)||parseXname(loc);
    gpuSeries.push({host:h,gpu:g,key:key,rows:rows,vals:vals,mean:mean,status:status,
      xname:xn,loc:loc});
  });
  if (!gpuSeries.length) { sec.style.display="none"; return; }
  // Sort
  var sortMode=document.getElementById("waterfall-sort").value;
  function topoSort(a,b){
    if(a.xname&&b.xname){
      if(a.xname.cabinet!==b.xname.cabinet) return a.xname.cabinet-b.xname.cabinet;
      if(a.xname.chassis!==b.xname.chassis) return a.xname.chassis-b.xname.chassis;
      if(a.xname.slot!==b.xname.slot) return a.xname.slot-b.xname.slot;
      if(a.xname.board!==b.xname.board) return a.xname.board-b.xname.board;
      if(a.xname.node!==b.xname.node) return a.xname.node-b.xname.node;
      return a.gpu-b.gpu;
    }
    if(a.loc&&b.loc&&a.loc!==b.loc) return a.loc<b.loc?-1:1;
    if(a.host!==b.host) return a.host<b.host?-1:a.host>b.host?1:0;
    return a.gpu-b.gpu;
  }
  if (sortMode==="topology") { gpuSeries.sort(topoSort); }
  else if (sortMode==="performance") {
    // T17: ascending column mean of the selected waterfall metric
    gpuSeries.sort(ascByMean(function(s){ return s.mean; }));
  } else {
    gpuSeries.sort(function(a,b){
      if(a.host!==b.host) return a.host<b.host?-1:a.host>b.host?1:0;
      return a.gpu-b.gpu;
    });
  }
  // Adaptive aggregation: >100 GPUs -> aggregate to node level
  var columns=[]; // {label, series:[gpuSeries entries], group}
  var useNodeAgg=gpuSeries.length>100;
  var entityLabel=useNodeAgg?"node":"GPU";
  var sub=document.getElementById("waterfall-subtitle");
  if(sub) sub.textContent="Vertical stripes = consistent per-"+entityLabel+" difference. Horizontal bands = all "+entityLabel+"s affected simultaneously. Uniform color = healthy fleet. Cells show the measured (or downsampled mean) value; the top line shows the latest iteration per column.";
  if(useNodeAgg){
    var byNode={};
    gpuSeries.forEach(function(s){
      if(!byNode[s.host]) byNode[s.host]={label:s.host,series:[],group:"",xname:s.xname,loc:s.loc};
      byNode[s.host].series.push(s);
      if(s.xname) byNode[s.host].group="x"+s.xname.cabinet;
      else if(s.loc) byNode[s.host].group=s.loc;
      else byNode[s.host].group=s.host;
    });
    // Sort nodes using first GPU's sort key
    var nodeKeys=Object.keys(byNode);
    if(sortMode==="topology"){
      nodeKeys.sort(function(a,b){return topoSort(byNode[a].series[0],byNode[b].series[0]);});
    } else if(sortMode==="performance"){
      // T18: ascending node mean of the constituent GPU means
      var nodeMeanOfGpuMeans=function(h){
        var series=byNode[h].series;
        return series.reduce(function(s,g){return s+g.mean;},0)/series.length;
      };
      nodeKeys.sort(ascByMean(nodeMeanOfGpuMeans));
    } else { nodeKeys.sort(); }
    nodeKeys.forEach(function(h){columns.push(byNode[h]);});
  } else {
    gpuSeries.forEach(function(s){
      var grp=s.xname?"x"+s.xname.cabinet:(s.loc||s.host);
      columns.push({label:s.host+":GPU"+s.gpu,series:[s],group:grp});
    });
  }
  // Build iteration grid
  var maxIter=0;
  gpuSeries.forEach(function(s){
    s.rows.forEach(function(r){if(r.iteration>maxIter)maxIter=r.iteration;});
  });
  var step=1;
  var MAX_ROWS=2000;
  if(maxIter>MAX_ROWS) step=Math.ceil(maxIter/MAX_ROWS);
  var iterCount=Math.ceil((maxIter+1)/step);
  var nCols=columns.length;
  // Build z-matrix (rows=iterations, cols=columns)
  var xLabels=columns.map(function(c){return c.label;});
  var yLabels=[];
  for(var yi=0;yi<iterCount;yi++) yLabels.push(yi*step);
  var zMatrix=new Array(iterCount);
  var zCounts=new Array(iterCount);
  for(var yi=0;yi<iterCount;yi++){
    zMatrix[yi]=new Float64Array(nCols);
    zCounts[yi]=new Uint16Array(nCols);
  }
  // Fill z-matrix (accumulate for averaging when aggregating or downsampling).
  // T19/T20: a measured 0 is a sample and is averaged in; only null is a gap.
  columns.forEach(function(col,xi){
    col.series.forEach(function(s){
      s.rows.forEach(function(r){
        var yi=Math.floor(r.iteration/step);
        if(yi>=iterCount) return;
        var v=r[mKey];
        if(v!=null){
          zMatrix[yi][xi]+=v;
          zCounts[yi][xi]++;
        }
      });
    });
  });
  // Convert sums to averages; cells with no samples become null (a gap, T20)
  var zOut=[],zSamples=[];
  for(var yi=0;yi<iterCount;yi++){
    var row=new Array(nCols);
    for(var xi=0;xi<nCols;xi++){
      if(zCounts[yi][xi]>0){ row[xi]=zMatrix[yi][xi]/zCounts[yi][xi]; zSamples.push(row[xi]); }
      else row[xi]=null;
    }
    zOut.push(row);
  }
  // T21: metric-specific colour domain. Utilisation is bounded 0-100; other
  // metrics span the p2-p98 of non-null cells so a few zeros or spikes do not
  // flatten the colour structure of the rest.
  var zRange=colourDomain(wfMetric,zSamples);
  // SDR-style colorscale: dark blue -> cyan -> yellow -> red -> white.
  // Null cells render transparent (page background), which is outside this
  // palette, so "no data" is visually distinct from the lowest value (T20).
  var colorscale=[
    [0.00,"#000033"],[0.15,"#0000aa"],[0.30,"#0066cc"],
    [0.45,"#00cccc"],[0.60,"#66cc00"],[0.75,"#cccc00"],
    [0.88,"#cc3300"],[0.95,"#ff3300"],[1.00,"#ffffff"]
  ];
  // Latest-iteration value per column for the spectrum trace (top panel).
  // In waterfall convention the spectrum shows the most recent sweep.
  // T22: a column whose GPUs have no non-null sample yields null, which
  // breaks the line there instead of plotting a fictitious 0.
  var colLatest=columns.map(function(c){
    var sum=0,cnt=0;
    c.series.forEach(function(s){
      var latestIter=-1,latestVal=null,hasLatest=false;
      s.rows.forEach(function(r){
        var v=r[mKey]; if(v!=null&&r.iteration>latestIter){latestIter=r.iteration;latestVal=v;hasLatest=true;}
      });
      if(hasLatest){sum+=latestVal;cnt++;}
    });
    return cnt>0?sum/cnt:null;
  });
  var latestPresent=colLatest.filter(function(v){return v!=null;});
  var fleetMean=latestPresent.length?latestPresent.reduce(function(s,v){return s+v;},0)/latestPresent.length:0;
  var spectrumTrace={
    x:xLabels,y:colLatest,
    type:"scatter",
    mode:"lines",
    connectgaps:false,
    line:{color:"#0072B2",width:1.5},
    fill:"tozeroy",
    fillcolor:"rgba(0,114,178,0.15)",
    yaxis:"y2",
    hovertemplate:"%{x}<br>Latest iteration "+mLabel+": %{y:.1f} "+unit+"<extra></extra>",
    showlegend:false
  };
  // Fleet mean line on spectrum panel
  var spectrumShapes=[
    {type:"line",x0:-0.5,x1:nCols-0.5,y0:fleetMean,y1:fleetMean,yref:"y2",
     line:{color:"#5dcaa5",width:1.5,dash:"dash"}}
  ];
  // Always use heatmap with zsmooth for reliable rendering (heatmapgl has
  // issues with autorange:"reversed" and sparse data at scale)
  var trace={
    z:zOut,x:xLabels,y:yLabels,
    type:"heatmap",colorscale:colorscale,
    zmin:zRange[0],zmax:zRange[1],
    hoverongaps:false,
    hovertemplate:"%{x}<br>Iter %{y}<br>"+mLabel+": %{z:.1f} "+unit+"<extra></extra>",
    colorbar:{title:{text:mLabel,side:"right"},thickness:14,len:0.6,y:0.3,
      tickfont:{size:10,color:themeColors().axis},titlefont:{size:11,color:themeColors().axis}},
    zsmooth:"fast",
    showscale:true,
    yaxis:"y"
  };
  // Group separator lines + group tick annotations
  var shapes=spectrumShapes.slice(),annotations=[];
  if(sortMode==="topology"&&nCols>1){
    var prevGroup="",groupStart=0;
    columns.forEach(function(c,xi){
      if(prevGroup&&c.group!==prevGroup){
        var mid=(groupStart+xi-1)/2;
        annotations.push({x:mid,y:-0.02,yref:"paper",text:prevGroup,
          showarrow:false,font:{size:nCols>200?7:nCols>80?8:9,color:themeColors().axis},yanchor:"top"});
        groupStart=xi;
      }
      prevGroup=c.group;
    });
    if(prevGroup){
      var mid=(groupStart+nCols-1)/2;
      annotations.push({x:mid,y:-0.02,yref:"paper",text:prevGroup,
        showarrow:false,font:{size:nCols>200?7:nCols>80?8:9,color:themeColors().axis},yanchor:"top"});
    }
  }
  // Adaptive tick display: hide or thin out labels to prevent overlapping
  var xAxisCfg=tAxis({
    side:"bottom",showgrid:false,zeroline:false,showline:false,ticks:""
  });
  // T13/T17/T18: the "performance" sort mode is an ascending sort on the
  // selected metric's column mean, so name it that way on the axis.
  var sortLabel=sortMode==="performance"?"ascending "+mLabel:sortMode;
  if(sortMode==="topology"&&annotations.length>0){
    // Topology mode with group labels: hide per-GPU ticks, group annotations provide context
    xAxisCfg.showticklabels=false;
  } else if(nCols<=24){
    // Small fleet: show all labels
    xAxisCfg.showticklabels=true;
    xAxisCfg.tickangle=-45;
    xAxisCfg.tickfont={size:10};
    xAxisCfg.title={text:(useNodeAgg?"Node":"GPU")+" (sorted by "+sortLabel+")",font:{size:11,color:themeColors().axis}};
  } else {
    // Large fleet, non-topology: show every Nth label
    var every=Math.ceil(nCols/20);
    var tVals=[],tText=[];
    xLabels.forEach(function(lbl,i){
      if(i%every===0){tVals.push(lbl);tText.push(lbl);}
    });
    xAxisCfg.tickvals=tVals;
    xAxisCfg.ticktext=tText;
    xAxisCfg.tickangle=-45;
    xAxisCfg.tickfont={size:9};
    xAxisCfg.title={text:(useNodeAgg?"Node":"GPU")+" ("+nCols+(useNodeAgg?" nodes":" GPUs")+", sorted by "+sortLabel+")",
      font:{size:11,color:themeColors().axis}};
  }
  // T22: the spectrum range spans the columns that have a latest sample;
  // nulls are gaps and must not drag the axis floor to 0.
  var y2Axis=tAxis({title:{text:"latest iteration "+mLabel+" ("+unit+")",font:{size:10,color:themeColors().axis}},showgrid:false,zeroline:false,showline:false,ticks:"",
      domain:[0.78,1],anchor:"x"});
  if(latestPresent.length){
    y2Axis.range=[Math.min.apply(null,latestPresent)*0.95,Math.max.apply(null,latestPresent)*1.02];
  }
  var layout=Object.assign({},baseLayout,{
    shapes:shapes,annotations:annotations,
    xaxis:xAxisCfg,
    yaxis:tAxis({title:tAxisTitle("Iteration"),showgrid:false,zeroline:false,showline:false,ticks:"",
      range:[yLabels[0],yLabels[yLabels.length-1]],
      domain:[0,0.72]}),
    yaxis2:y2Axis,
    height:Math.max(600,Math.min(iterCount*0.5+200,1000)),
    margin:{t:36,b:nCols<=24?80:sortMode==="topology"?52:90,l:72,r:60}
  });
  Plotly.newPlot("waterfall-chart",[spectrumTrace,trace],layout,plotConfig);
}

// ---- Fleet Inventory Table ----
var _invSortCol=-1,_invSortAsc=true;
function renderInventory() {
  var data=getFiltered(),metric=getMetric();
  var oSet=mkOutlierSet(),tSet=mkThrottleSet();
  var unit=metricUnit(data,metric);
  var allVals=data.map(function(r){return r[metric];}).filter(function(v){return v>0;});
  var fleetMean=arrStats(allVals).mean;
  var seen={},rows=[];
  data.forEach(function(r){var k=r.hostname+":"+r.gpu;if(seen[k])return;seen[k]=true;rows.push(r);});
  // Build sortable row data
  var rowData=rows.map(function(r){
    var val=r[metric];
    var dev=fleetMean>0?((val-fleetMean)/fleetMean*100):0;
    var status=gpuStatus(r,oSet,tSet);
    return {r:r,val:val,dev:dev,status:status,
      sortKeys:[r.hostname,r.gpu,r.gpu_model,r.serial,val,dev,
                r.power_avg_w||0,r.temp_max_c||0,status==="normal"?0:status==="throttled"?1:2]};
  });
  // Filter: show only problems unless "Show all" is checked
  var showAll=document.getElementById("inv-show-all").checked;
  var displayData=showAll?rowData:rowData.filter(function(d){return d.status!=="normal";});
  var countEl=document.getElementById("inv-count");
  if(countEl){
    var nProb=rowData.filter(function(d){return d.status!=="normal";}).length;
    countEl.textContent=showAll?"showing all "+rowData.length+" GPUs":"showing "+nProb+" of "+rowData.length+" GPUs (outliers + throttled)";
  }
  if(_invSortCol>=0){
    displayData.sort(function(a,b){
      var va=a.sortKeys[_invSortCol],vb=b.sortKeys[_invSortCol];
      var cmp=typeof va==="string"?(va<vb?-1:va>vb?1:0):va-vb;
      return _invSortAsc?cmp:-cmp;
    });
  } else {
    displayData.sort(function(a,b){return a.r.hostname<b.r.hostname?-1:a.r.hostname>b.r.hostname?1:a.r.gpu-b.r.gpu;});
  }
  var cols=["Hostname","GPU","Model","Serial","GPU "+metricStatistic(metric)+" "+metricNoun(metric)+" ("+unit+")","vs fleet mean","Power","Temp","Status"];
  var aligns=["left","right","left","left","right","right","right","right","center"];
  var h='<table><thead><tr>';
  cols.forEach(function(c,i){
    var cls=_invSortCol===i?(_invSortAsc?" asc":" desc"):"";
    h+='<th class="sortable'+cls+'" data-col="'+i+'" style="text-align:'+aligns[i]+'">'+c+'</th>';
  });
  h+='</tr></thead><tbody>';
  displayData.forEach(function(d){
    var r=d.r,val=d.val,dev=d.dev,status=d.status;
    var icon=status==="outlier-low"?"\\u2716":status==="outlier-high"?"\\u25B2":status==="throttled"?"\\u26a0":"\\u2714";
    var sc=status==="outlier-low"?"var(--danger)":status==="outlier-high"?"#a78bfa":status==="throttled"?"var(--warn)":"var(--success)";
    var dc=Math.abs(dev)>10?"var(--danger)":Math.abs(dev)>5?"var(--warn)":"var(--muted)";
    h+='<tr>';
    h+='<td style="text-align:left">'+esc(r.hostname)+'</td>';
    h+='<td>'+r.gpu+'</td>';
    h+='<td style="text-align:left">'+esc(r.gpu_model)+'</td>';
    h+='<td style="text-align:left;font-size:11px">'+esc(r.serial)+'</td>';
    h+='<td>'+(val>0?val.toFixed(1):"--")+'</td>';
    h+='<td style="color:'+dc+'">'+(dev>=0?"+":"")+dev.toFixed(1)+'%</td>';
    h+='<td>'+(r.power_avg_w>0?r.power_avg_w.toFixed(0)+"W":"--")+'</td>';
    h+='<td>'+(r.temp_max_c>0?r.temp_max_c.toFixed(0)+"\\u00b0C":"--")+'</td>';
    h+='<td style="text-align:center;color:'+sc+'">'+icon+'</td>';
    h+='</tr>';
  });
  h+='</tbody></table>';
  document.getElementById("inventory-table").innerHTML=h;
  // Attach sort handlers
  document.querySelectorAll("#inventory-table th.sortable").forEach(function(th){
    th.addEventListener("click",function(){
      var col=parseInt(this.getAttribute("data-col"));
      if(_invSortCol===col){_invSortAsc=!_invSortAsc;}else{_invSortCol=col;_invSortAsc=true;}
      renderInventory();
    });
  });
}

function renderAll() {
  // Update outlier card dynamically based on sigma slider
  var oSet=mkOutlierSet();
  var oCount=Object.keys(oSet).length;
  var oCard=document.getElementById("card-outliers");
  if(oCard){
    oCard.querySelector(".value").textContent=oCount;
    oCard.className="metric-card"+(oCount>0?" bad":" ok");
  }
  renderFleetMap();
  renderDistribution();
  renderEfficiency();
  renderStrip();
  renderTimeSeries();
  renderWaterfall();
  renderInventory();
}
document.getElementById("filter-host").addEventListener("change",renderAll);
document.getElementById("filter-bench").addEventListener("change",renderAll);
document.getElementById("metric-select").addEventListener("change",function(){
  // Sync waterfall metric dropdown to match top-level metric
  var summaryToWf={"mean_val":"performance","power_avg_w":"power_W","temp_max_c":"temp_gpu_C",
    "sm_util_mean":"sm_util","mem_bw_util_mean":"mem_bw_util","gpu_clock_mean":"gpu_clock"};
  var wfSel=document.getElementById("waterfall-metric");
  var mapped=summaryToWf[getMetric()];
  if(wfSel&&mapped){wfSel.value=mapped;}
  renderAll();
});
document.getElementById("outlier-sigma").addEventListener("input",function(){
  document.getElementById("sigma-label").textContent=this.value+" sigma";
  renderAll();
});
document.getElementById("waterfall-sort").addEventListener("change",function(){renderWaterfall();});
document.getElementById("waterfall-metric").addEventListener("change",function(){renderWaterfall();});
document.getElementById("inv-show-all").addEventListener("change",function(){renderInventory();});
initTheme();
applyTheme();
renderAll();
""")
    parts.append('</script>\n</body>\n</html>\n')
    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # type: () -> int
    p = argparse.ArgumentParser(
        description="Fleet report from torch-hammer benchmark output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s results/                          CLI summary (default)\n"
            "  %(prog)s results.csv -o report.html        CLI summary + HTML report\n"
            "  %(prog)s results.json --benchmark GEMM     filter to GEMM benchmarks\n"
            "  %(prog)s dump.txt --shell-output           parse shell dump\n"
            "  %(prog)s results/ --quiet                  exit code only (CI mode)\n"
        ),
    )
    p.add_argument("source", help="CSV file, JSON file, directory, or shell dump")
    p.add_argument("-o", "--output", help="HTML output path")
    p.add_argument("-b", "--benchmark", help="Filter benchmark (substring match)")
    p.add_argument("--dtype", help="Filter dtype (exact match)")
    p.add_argument("--shell-output", action="store_true",
                   help="Force shell-dump parse mode")
    p.add_argument("--outlier-threshold", type=float, default=15.0,
                   help="Outlier deviation threshold %% (default: 15)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress summary; exit 0=pass, 1=outliers found")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI color output")
    p.add_argument("--system-name", default="",
                   help="System name for report title (e.g. 'AI-P2')")
    p.add_argument("--job-name", default="",
                   help="Job/run name for report title (e.g. '2026-04-20 Maintenance')")
    p.add_argument("--node-map", default="",
                   help="CSV file mapping hostname,location for topology grouping")
    p.add_argument("--dot-plot", action="store_true",
                   help="Use dot plots instead of histograms for distribution charts")
    p.add_argument("--interactive", action="store_true",
                   help="Generate interactive Plotly dashboard (requires: pip install plotly)")
    p.add_argument("--theme", choices=["auto", "light", "dark"], default="auto",
                   help="HTML color theme: auto (follow OS), light, or dark")
    args = p.parse_args()

    if args.no_color:
        global RED, YELLOW, GREEN, BOLD, RESET
        RED = YELLOW = GREEN = BOLD = RESET = ""

    source = Path(args.source)
    if not source.exists():
        print("Error: {} does not exist".format(source), file=sys.stderr)
        return 2

    # Load
    results = load_input(source, shell_mode=args.shell_output)

    if _parse_warnings > 0:
        print("  Warning: {} row(s) skipped (malformed)".format(
            _parse_warnings), file=sys.stderr)

    if not results:
        print("Error: no data found", file=sys.stderr)
        return 2

    # Filter
    if args.benchmark:
        needle = args.benchmark.lower()
        results = [r for r in results if needle in r.benchmark.lower()]
    if args.dtype:
        results = [r for r in results if r.dtype == args.dtype]

    if not results:
        print("Error: no rows after filtering", file=sys.stderr)
        return 2

    # Node map
    node_map = None  # type: Optional[Dict[str, str]]
    if args.node_map:
        nm_path = Path(args.node_map)
        if not nm_path.exists():
            print("Error: node map {} does not exist".format(nm_path), file=sys.stderr)
            return 2
        node_map = load_node_map(nm_path)

    # Compute
    bench_stats = compute_benchmark_stats(results)
    _auto_scale_units(results, bench_stats, _iteration_data)
    outliers = detect_outliers(bench_stats, threshold_pct=args.outlier_threshold)
    bad_outliers = [o for o in outliers if o["severity"] == "bad"]
    exit_code = 1 if bad_outliers else 0

    # CLI summary
    if not args.quiet:
        print_summary(results, bench_stats, outliers, args.outlier_threshold,
                      system_name=args.system_name, job_name=args.job_name,
                      node_map=node_map)

    # Interactive mode (Plotly)
    if args.interactive:
        if not _PLOTLY_AVAILABLE:
            print("Warning: --interactive requires plotly. "
                  "Install with: pip install plotly", file=sys.stderr)
            print("Falling back to static SVG report.", file=sys.stderr)
        else:
            html_out = _render_interactive_html(
                results, bench_stats, outliers,
                source_name=source.name,
                threshold=args.outlier_threshold,
                system_name=args.system_name,
                job_name=args.job_name,
                node_map=node_map,
                theme_mode=args.theme,
            )
            if args.output:
                Path(args.output).write_text(html_out, encoding='utf-8')
                size_kb = len(html_out) / 1024
                print("Interactive report: {} ({:.0f} KB)".format(
                    args.output, size_kb), file=sys.stderr)
            return exit_code

    # HTML
    if args.output:
        html_content = render_html(
            results, bench_stats, outliers,
            source_name=source.name,
            threshold=args.outlier_threshold,
            system_name=args.system_name,
            node_map=node_map,
            job_name=args.job_name,
            dot_plot=args.dot_plot,
            theme_mode=args.theme,
        )
        out_path = Path(args.output)
        out_path.write_text(html_content, encoding="utf-8")
        print("  Report written to: {}".format(out_path), file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
