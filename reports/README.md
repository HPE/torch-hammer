# torch-hammer-reporter

Fleet report generator for `torch-hammer` benchmark output. Produces a CLI
summary (default), a self-contained static HTML report, or an interactive
Plotly dashboard. No dependencies beyond the Python standard library for core
functionality. Plotly is required only for `--interactive` mode.

## Usage

```bash
# CLI summary (default -- works over SSH, meaningful exit codes)
python hammer_report.py results/
python hammer_report.py results.csv
python hammer_report.py results.json

# CLI summary + static HTML report
python hammer_report.py results/ -o report.html

# Interactive Plotly dashboard (requires: pip install plotly)
python hammer_report.py results/ --interactive -o dashboard.html

# Filter to a specific benchmark / dtype
python hammer_report.py results.csv --benchmark "Batched GEMM" --dtype float32

# Shell dump from HPC
python hammer_report.py dump.txt --shell-output

# Custom report title
python hammer_report.py results/ -o report.html \
    --system-name "Tahiti" --job-name "2026-04-20 Maintenance"

# Topology-aware grouping via node map
python hammer_report.py results/ -o report.html --node-map locations.csv

# Dot plots instead of histograms
python hammer_report.py results/ -o report.html --dot-plot

# Quiet mode for CI (exit 0 = pass, exit 1 = outliers detected)
python hammer_report.py results/ --quiet --outlier-threshold 10
```

The script **auto-detects** the input format (compact CSV, summary CSV,
JSON, verbose log-dir files, or shell dump).

## Input Formats

### Compact CSV (`--compact`)

The primary format. One row per (GPU, benchmark, dtype):

```
hostname,gpu,gpu_model,serial,benchmark,dtype,iterations,runtime_s,min,mean,max,unit,power_avg_w,temp_max_c
nid005193,0,NVIDIA GH200 120GB,165402507401,...
```

Pass a single file or a directory of per-node CSV files.

### Summary CSV (`--summary-csv`)

One row per (GPU, benchmark). Columns: `test,dtype,gpu,serial,performance,unit,...`

### JSON (`--json-output`)

Full torch-hammer JSON export with `metadata`, `runtime_args`, and `gpus[]`.
Per-iteration telemetry data is extracted when present. In per-iteration
records a telemetry field that is absent, blank or `N/A` is kept as
`null` (a gap), while a measured `0` stays `0` -- the two are never conflated.

### Verbose log-dir files

Per-GPU timestamp-prefixed CSV files produced by `--log-dir --verbose`.
Auto-detected when a directory contains `.csv` or `.log` files with the
verbose column layout. Missing telemetry columns follow the same
`null`-for-missing rule as JSON.

### Shell dump

Raw output of `for f in *; do echo "file: $f"; cat $f; done`.
Use `--shell-output` to force this mode, or let auto-detection handle it.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `source` | (required) | CSV file, JSON file, directory, or shell dump |
| `-o`, `--output` | -- | Write HTML report to this path |
| `-b`, `--benchmark` | -- | Filter: only benchmarks matching this substring |
| `--dtype` | -- | Filter: only this exact dtype |
| `--shell-output` | off | Force shell-dump parse mode |
| `--outlier-threshold` | 15.0 | Deviation % to flag as outlier |
| `--quiet` | off | Suppress CLI summary; exit code only |
| `--no-color` | off | Disable ANSI color output |
| `--system-name` | -- | System name for report title (e.g. `Tahiti`) |
| `--job-name` | -- | Job/run name for report title (e.g. `2026-04-20 Maintenance`) |
| `--node-map` | -- | CSV file mapping `hostname,location` for topology grouping |
| `--dot-plot` | off | Use dot plots instead of histograms for distribution charts |
| `--interactive` | off | Generate interactive Plotly dashboard (requires `pip install plotly`) |
| `--theme` | light | Default theme for generated HTML (`light`, `dark`, `auto`) |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No outliers detected |
| 1 | Outliers detected (below-fleet GPUs) |
| 2 | Input error (missing file, no data, empty after filter) |

## Report Contents

### CLI Summary (stderr)

- **Fleet overview**: nodes, GPUs, GPU model, benchmark count
- **Per-benchmark table**: fleet mean, CV%, mean power, max temp, throttle flag ("Thrt" column)
- **Per-node health**: GPU count, test count, mean power, max temp, status (PASS/WARN/THRT)
- **Outlier list** with deviation % from fleet mean, direction (below/above fleet)
- **Thermal throttling section**: lists affected GPUs when throttling is detected
- **Location cluster summary**: groups outliers by location when `--node-map` is provided
- **Fleet verdict**: PASS / WARN / FAIL / THROTTLED counts

### Unit Auto-Scaling

When the minimum GPU mean exceeds 1000, units are automatically scaled up:

| Original | Scaled |
|----------|--------|
| GFLOP/s | TFLOP/s |
| GB/s | TB/s |
| MLUP/s | GLUP/s |
| Mops/s | Gops/s |

### Outlier Detection

Two algorithms, depending on output mode:

- **CLI and static HTML**: percentage-of-mean deviation, configurable via `--outlier-threshold` (default 15%)
- **Interactive dashboard**: sigma-based (mean ± N×sigma), adjustable via slider with a 5% floor

Outliers are classified by direction: "bad" (below fleet mean) and "good" (above fleet mean).
Only "bad" outliers trigger a non-zero exit code.

### Thermal Throttling

Throttle flags are parsed from compact CSV, JSON (benchmark-level and per-iteration
telemetry), and verbose log files.

- **CLI**: "Thrt" column in the per-benchmark table, dedicated throttling section
  listing affected GPUs, THRT status in per-node health, throttled count in verdict
- **Static HTML**: warning icon in the throttle column, stats line with count
- **Interactive**: throttled metric card, header warning banner, throttle status
  coloring across all charts

## Static HTML Report (`-o`)

Self-contained HTML with server-side SVG charts. Zero external dependencies, fully
offline.

- **Metric cards**: nodes, GPUs, benchmarks, outlier count
- **Scale-adaptive charts**: per-node SVG bar charts for ≤50 nodes; SVG histograms
  for >50 nodes (fleet mean bin highlighted)
- **Dot plot mode** (`--dot-plot`): sigma-band colored dot plots with KDE density
  curve and rug plot, as an alternative to histograms
- **Multi-metric panels**, each titled "Fleet distribution of GPU mean/max &lt;metric&gt;":
  performance, power (W), temperature (°C, per-GPU max), SM utilization (%),
  memory BW utilization (%), GPU clock (MHz). The performance chart is
  full-width (960-unit viewBox); the diagnostic panels are compact small
  multiples (360-unit viewBox, three per row) with fewer ticks, no legend and
  no per-point labels, so their text renders at the same size as the main
  chart instead of shrinking with the grid cell
- **Chart zoom button** for closer inspection
- **Sortable tables**: click any column header to sort ascending/descending
  (via `data-sort` attributes)
- **"node mean" and "vs fleet" columns**: per-node mean of GPU means, and its
  signed % deviation from the fleet mean
- **Truncated tables** for large fleets: bottom 5, outliers, top 5 with section labels
  and "Show all rows" toggle
- **Percentile stats** (p5, median, p95) for large fleets
- **Outlier section** with per-GPU deviation details and location cluster summary
  (when `--node-map` is provided)
- **Dark mode** via `prefers-color-scheme`
- **Colorblind-safe** Okabe-Ito palette

## Interactive Plotly Dashboard (`--interactive -o`)

Requires `plotly` (`pip install plotly`). Embeds Plotly.js inline for a fully
self-contained HTML file. Falls back to static SVG if Plotly is not installed.

### Controls

- Dark/light theme toggle
- Filter controls: hostname, benchmark, metric dropdown, sigma slider (1--3, step 0.5)
- Sigma slider sets outlier bounds with a 5% floor

### Charts

**Fleet Map**: Topology-aware grid layout. Auto-detects Cray EX xname format
(`xNNNNcCsCbB`) and groups by cabinet with chassis/slot/board positioning.
Falls back to a flat grid in ascending node mean. Supports `--node-map`
location grouping.

**Fleet Distribution**: Histogram with per-GPU rug plot, sigma-band coloring
(1/2/3 sigma), fleet mean and sigma lines, stats legend with kurtosis.

**Power vs Performance**: Scatter plot with temperature color gradient.
Outlier and throttled GPUs get distinct sizing. Power limit detection.

**Node Variability**: Strip plot in ascending node mean, left to right, for
every metric; colour carries good/bad (outlier and throttle coding). Adaptive
truncation for >60 nodes keeps both ends plus abnormal nodes.

**Iteration Trace**: p10--p90 envelope for large fleets, individual lines for
outlier/throttled GPUs, rolling mean smoothing, adaptive downsampling.
Per-iteration data from JSON or verbose logs.

**Fleet Waterfall**: Heatmap with spectrum trace. Multiple sort modes
(topology / by value ascending / hostname), per-metric dropdown, node-level
aggregation for >100 GPUs, SDR-style colorscale. Utilization metrics use a
fixed 0--100 colour domain; other metrics span the p2--p98 of the cells. A
cell with no sample is a transparent gap with no hover; a measured `0` is
coloured and hoverable like any other value.

**Fleet Inventory**: Filterable, sortable table with show-all toggle and
status icons.

### Data Compression

Iteration-level data is gzip+base64 compressed in the HTML output and
decompressed client-side via `DecompressionStream`.

## Node Map Format

A two-column CSV with `hostname,location`:

```csv
hostname,location
nid005193,row-A-rack-1
nid005194,row-A-rack-1
nid005195,row-B-rack-3
```

Used for topology grouping in the CLI location cluster summary, static HTML
outlier section, and interactive Fleet Map.

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| Python ≥3.8 | yes | Standard library only for core + static HTML |
| `plotly` | no | `--interactive` dashboard mode |

## Tests

```bash
pytest tests/test_report.py -v
```

218 tests covering: compact/summary/JSON/verbose/shell parsing, multi-benchmark
grouping, outlier detection, thermal throttle parsing, unit auto-scaling, HTML
XSS safety, CLI exit codes, edge cases, scale-adaptive rendering, sortable table
markup, vs-fleet column, SVG charts, multi-metric panels, sort JS correctness,
CLI truncation, dot plots, interactive report escaping, and node-map grouping.
