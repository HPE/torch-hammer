# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for telemetry classes and data structures.
"""
import pytest
from unittest.mock import MagicMock, patch
import time


class TestTelemetryBase:
    """Tests for the TelemetryBase abstract class."""
    
    def test_base_supported_fields(self, th):
        """TelemetryBase should have minimal supported fields."""
        assert "vendor" in th.TelemetryBase.supported
        assert "model" in th.TelemetryBase.supported
        assert "device_id" in th.TelemetryBase.supported
    
    def test_schema_returns_supported(self, th):
        """schema() should return the supported fields list."""
        tel = th.TelemetryBase()
        schema = tel.schema()
        
        assert schema == th.TelemetryBase.supported
    
    def test_read_not_implemented(self, th):
        """read() should raise NotImplementedError in base class."""
        tel = th.TelemetryBase()
        
        with pytest.raises(NotImplementedError):
            tel.read()
    
    def test_get_stats_returns_empty(self, th):
        """get_stats() should return empty dict in base class."""
        tel = th.TelemetryBase()
        stats = tel.get_stats()
        
        assert stats == {}
    
    def test_reset_stats_is_noop(self, th):
        """reset_stats() should be a no-op in base class."""
        tel = th.TelemetryBase()
        # Should not raise
        tel.reset_stats()
    
    def test_shutdown_is_noop(self, th):
        """shutdown() should be a no-op in base class."""
        tel = th.TelemetryBase()
        # Should not raise
        tel.shutdown()


class TestCpuTelemetry:
    """Tests for the CPU fallback telemetry class."""
    
    def test_cpu_telemetry_init(self, th):
        """CpuTelemetry should initialize correctly."""
        tel = th.CpuTelemetry(index=0)
        
        assert tel.idx == 0
        assert hasattr(tel, 'hostname')
        assert hasattr(tel, '_model')
    
    def test_cpu_telemetry_read(self, th):
        """CpuTelemetry.read() should return valid data."""
        tel = th.CpuTelemetry(index=0)
        data = tel.read()
        
        assert "vendor" in data
        assert "model" in data
        assert "device_id" in data
        assert data["device_id"] == 0
        assert "hostname" in data
    
    def test_cpu_telemetry_different_index(self, th):
        """CpuTelemetry should respect device index."""
        tel = th.CpuTelemetry(index=3)
        data = tel.read()
        
        assert data["device_id"] == 3


class TestIntelTelemetry:
    """Tests for the Intel GPU telemetry class (placeholder)."""
    
    def test_intel_telemetry_init(self, th):
        """IntelTelemetry should initialize correctly."""
        tel = th.IntelTelemetry(index=0)
        
        assert tel.idx == 0
        assert hasattr(tel, 'hostname')
    
    def test_intel_telemetry_read(self, th):
        """IntelTelemetry.read() should return valid data."""
        tel = th.IntelTelemetry(index=1)
        data = tel.read()
        
        assert data["vendor"] == "Intel"
        assert data["model"] == "Intel_GPU"
        assert data["device_id"] == 1
        assert "hostname" in data


class TestNVMLTelemetryFields:
    """Tests for NVIDIA telemetry field definitions (no actual GPU needed)."""
    
    def test_nvml_supported_fields(self, th):
        """NVMLTelemetry should define comprehensive supported fields."""
        expected_fields = [
            "hostname", "vendor", "model", "device_id", "serial",
            "sm_util", "mem_bw_util", "mem_util",
            "gpu_clock", "mem_clock",
            "power_W", "temp_gpu_C", "temp_hbm_C",
            "mem_used_MB", "mem_total_MB", "mem_free_MB",
            "hw_slowdown", "sw_slowdown", "power_limit", "throttled",
        ]
        
        for field in expected_fields:
            assert field in th.NVMLTelemetry.supported, f"Missing field: {field}"
    
    def test_nvml_throttle_fields(self, th):
        """NVML telemetry should include throttle detection fields."""
        throttle_fields = ["hw_slowdown", "sw_slowdown", "power_limit", "throttled"]
        
        for field in throttle_fields:
            assert field in th.NVMLTelemetry.supported


class TestRocmTelemetryFields:
    """Tests for AMD ROCm telemetry field definitions (no actual GPU needed)."""
    
    def test_rocm_gpu_fields(self, th):
        """RocmTelemetry should define GPU-specific fields."""
        # Check GPU fields (class-level constant)
        expected_fields = [
            "vendor", "model", "device_id", "serial",
            "gpu_clock", "mem_clock", "power_W",
        ]
        
        for field in expected_fields:
            assert field in th.RocmTelemetry.GPU_FIELDS, f"Missing GPU field: {field}"
    
    def test_rocm_cpu_fields(self, th):
        """RocmTelemetry should define CPU-specific fields."""
        # Check CPU fields (for APU mode)
        expected_fields = [
            "vendor", "model", "device_id", "serial",
        ]
        
        for field in expected_fields:
            assert field in th.RocmTelemetry.CPU_FIELDS, f"Missing CPU field: {field}"


class TestRocmPowerFallback:
    """Tests for AMD ROCm power reading across GPU generations."""
    
    def _make_rocm_telemetry_stub(self, th, metrics_dict=None):
        """Create a RocmTelemetry instance with mocked internals for power testing."""
        tel = object.__new__(th.RocmTelemetry)
        tel.hostname = "testhost"
        tel._model = "MI250"
        tel._serial = "TEST-SERIAL"
        tel.idx = 0
        tel._is_gpu = True
        tel.supported = th.RocmTelemetry.GPU_FIELDS
        tel._cpu_power_warned = False
        tel._cpu_temp_warned = False
        tel.power_limit_W = None
        tel.throttle_detected = False
        tel.throttle_count = 0
        tel.power_throttle_count = 0
        tel.thermal_throttle_count = 0
        tel.readings = {
            'temp_gpu_C': [], 'power_W': [], 'gpu_clock': [],
            'sm_util': [], 'mem_util': [], 'mem_used_MB': []
        }
        
        mock_amdsmi = MagicMock()
        tel.amdsmi = mock_amdsmi
        tel.handle = MagicMock()
        
        default_metrics = {
            'average_gfx_activity': 95,
            'average_umc_activity': 50,
            'current_gfxclk': 1700,
            'current_uclk': 1600,
            'current_socket_power': 450,
            'average_socket_power': 450,
            'temperature_edge': 65,
            'throttle_status': 0,
        }
        mock_amdsmi.amdsmi_get_gpu_metrics_info.return_value = (
            metrics_dict if metrics_dict is not None else default_metrics
        )
        mock_amdsmi.amdsmi_get_gpu_memory_total.return_value = 64 * 1024**3
        mock_amdsmi.amdsmi_get_gpu_memory_usage.return_value = 32 * 1024**3
        mock_amdsmi.AmdSmiMemoryType.VRAM = "VRAM"
        
        return tel
    
    def test_mi300_power_from_current(self, th):
        """MI300: current_socket_power is populated, used directly."""
        tel = self._make_rocm_telemetry_stub(th)
        data = tel.read()
        assert data["power_W"] == 450
    
    def test_mi250_primary_gcd(self, th):
        """MI250 primary GCD: current='N/A', average=94 → uses average."""
        tel = self._make_rocm_telemetry_stub(th, metrics_dict={
            'average_gfx_activity': 100, 'average_umc_activity': 10,
            'current_gfxclk': 1605, 'current_uclk': 1600,
            'current_socket_power': 'N/A',
            'average_socket_power': 94,
            'temperature_edge': 44, 'throttle_status': 0,
        })
        data = tel.read()
        assert data["power_W"] == 94
    
    def test_mi250_secondary_gcd(self, th):
        """MI250 secondary GCD: current='N/A', average=0 → N/A (expected)."""
        tel = self._make_rocm_telemetry_stub(th, metrics_dict={
            'average_gfx_activity': 99, 'average_umc_activity': 11,
            'current_gfxclk': 1700, 'current_uclk': 1600,
            'current_socket_power': 'N/A',
            'average_socket_power': 0,
            'temperature_edge': 52, 'throttle_status': 0,
        })
        data = tel.read()
        assert data["power_W"] == "N/A"
    
    def test_power_tracked_in_readings(self, th):
        """Valid power should be tracked in readings list."""
        tel = self._make_rocm_telemetry_stub(th, metrics_dict={
            'average_gfx_activity': 95, 'average_umc_activity': 50,
            'current_gfxclk': 1700, 'current_uclk': 1600,
            'current_socket_power': 'N/A',
            'average_socket_power': 94,
            'temperature_edge': 65, 'throttle_status': 0,
        })
        tel.read()
        assert 94 in tel.readings['power_W']


class TestTelemetryThread:
    """Tests for the background telemetry thread."""
    
    def test_thread_initialization(self, th):
        """TelemetryThread should initialize correctly."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"vendor": "TEST", "device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=100)
        
        assert thread.sample_interval_ms == 100
        assert thread.running is False
        assert thread.active is False
        assert thread.latest_reading == {}
    
    def test_thread_start_stop(self, th):
        """TelemetryThread should start and stop cleanly."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"vendor": "TEST", "device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=50)
        
        thread.start()
        assert thread.running is True
        assert thread.thread is not None
        
        time.sleep(0.1)  # Let thread run briefly
        
        thread.stop()
        assert thread.running is False
    
    def test_get_latest_returns_data(self, th):
        """get_latest() should return telemetry data."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"vendor": "TEST", "model": "TestGPU", "device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=20)
        thread.start()
        
        time.sleep(0.1)  # Let thread collect some data
        
        latest = thread.get_latest()
        
        thread.stop()
        
        assert "vendor" in latest
        assert latest["vendor"] == "TEST"
    
    def test_set_active_flag(self, th):
        """set_active() should update the active flag."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device)
        
        assert thread.active is False
        
        thread.set_active(True)
        assert thread.active is True
        
        thread.set_active(False)
        assert thread.active is False
    
    def test_iteration_marking(self, th):
        """mark_iteration_start/end should track iterations."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"device_id": 0, "power_W": 100}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=10)
        thread.start()
        
        thread.mark_iteration_start(0)
        time.sleep(0.05)
        thread.mark_iteration_end(0)
        
        thread.mark_iteration_start(1)
        time.sleep(0.05)
        thread.mark_iteration_end(1)
        
        time.sleep(0.05)  # Let thread collect
        
        thread.stop()
        
        # Should have iteration data
        assert 0 in thread.iteration_samples or 1 in thread.iteration_samples
    
    def test_get_iteration_telemetry_aggregates(self, th):
        """get_iteration_telemetry() should aggregate samples."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"device_id": 0, "power_W": 100, "temp_gpu_C": 50}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=10)
        thread.start()
        
        thread.mark_iteration_start(5)
        time.sleep(0.05)
        thread.mark_iteration_end(5)
        
        time.sleep(0.02)  # Let thread finish
        
        data = thread.get_iteration_telemetry(5)
        
        thread.stop()
        
        # Should return data (either collected or latest)
        assert data is not None
        assert "device_id" in data


class TestMakeTelemetry:
    """Tests for the telemetry factory function."""
    
    def test_cpu_fallback(self, th):
        """make_telemetry should fall back to CPU when no GPU tools available."""
        import torch
        device = torch.device("cpu")
        
        # Mock shutil.which to return None (no nvidia-smi or rocm-smi)
        with patch('shutil.which', return_value=None):
            tel = th.make_telemetry(0, device)
        
        # Should get CpuTelemetry
        assert isinstance(tel, th.CpuTelemetry)
    
    def test_xpu_returns_intel(self, th):
        """make_telemetry should return IntelTelemetry for XPU devices."""
        import torch
        
        # Create a mock device that identifies as XPU
        mock_device = MagicMock()
        mock_device.type = "xpu"
        
        with patch('shutil.which', return_value=None):
            tel = th.make_telemetry(0, mock_device)
        
        assert isinstance(tel, th.IntelTelemetry)


class TestTelemetryStatAccumulation:
    """Tests for telemetry statistics accumulation."""
    
    def test_cpu_telemetry_no_stats(self, th, tmp_path):
        """CpuTelemetry without Linux sensors (e.g. macOS) has no stats."""
        tel = th.CpuTelemetry(0, root=str(tmp_path))
        
        # Read a few times
        for _ in range(5):
            tel.read()
        
        stats = tel.get_stats()
        assert stats == {}
    
    def test_reset_stats_clears_data(self, th):
        """reset_stats should work without error."""
        tel = th.CpuTelemetry(0)
        
        # Should not raise
        tel.reset_stats()


# ── Linux procfs/sysfs CPU telemetry (fake tree) ───────────────────────

def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _stat_text(ticks):
    """/proc/stat with per-CPU (busy, idle) jiffies, then a long non-cpu line."""
    lines = ["cpu  0 0 0 0 0 0 0 0 0 0"]
    for cpu, (busy, idle) in sorted(ticks.items()):
        lines.append(f"cpu{cpu} {busy} 0 0 {idle} 0 0 0 0 0 0")
    lines.append("intr 1 2 3")
    return "\n".join(lines) + "\n"


def _fake_node(root, sockets=1, cores_per_socket=2, cpuinfo=None):
    """Minimal Linux node: cpuinfo, topology, /proc/stat and /proc/meminfo."""
    _write(root, "proc/cpuinfo", cpuinfo or (
        "processor\t: 0\nvendor_id\t: GenuineIntel\n"
        "model name\t: Intel(R) Xeon(R)   Platinum 8480+\n\nprocessor\t: 1\n"))
    cpus = list(range(sockets * cores_per_socket))
    for cpu in cpus:
        _write(root, f"sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id",
               f"{cpu // cores_per_socket}\n")
    _write(root, "proc/stat", _stat_text({c: (0, 0) for c in cpus}))
    _write(root, "proc/meminfo",
           "MemTotal:       8388608 kB\nMemFree:          1024 kB\nMemAvailable:   6291456 kB\n")
    return cpus


def _hwmon(root, n, name, files):
    _write(root, f"sys/class/hwmon/hwmon{n}/name", name + "\n")
    for fname, text in files.items():
        _write(root, f"sys/class/hwmon/hwmon{n}/{fname}", text)


def _rapl(root, socket, energy_uj, max_range=262143328850):
    zone = f"sys/class/powercap/intel-rapl:{socket}"
    _write(root, f"{zone}/name", f"package-{socket}\n")
    _write(root, f"{zone}/max_energy_range_uj", f"{max_range}\n")
    _write(root, f"{zone}/intel-rapl:{socket}:0/name", "core\n")  # sub-domain, must be ignored
    return _write(root, f"{zone}/energy_uj", f"{energy_uj}\n")


class TestCpuTelemetrySysfs:
    """CpuTelemetry reading a fake /proc and /sys tree."""

    def test_x86_identity(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        d = tel.read()
        assert d["vendor"] == "GenuineIntel"
        assert d["model"] == "Intel(R) Xeon(R) Platinum 8480+"

    def test_arm_identity(self, th, tmp_path):
        cpus = _fake_node(tmp_path, cpuinfo=(
            "processor\t: 0\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\n"
            "CPU architecture: 8\nCPU part\t: 0xd4f\n\n"))
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()
        assert d["vendor"] == "ARM"
        assert d["model"] == "ARM Neoverse-V2"

    def test_arm_unknown_part_and_implementer(self, th, tmp_path):
        cpus = _fake_node(tmp_path, cpuinfo="CPU implementer\t: 0x4e\nCPU part\t: 0x123\n\n")
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()
        assert d["vendor"] == "ARM (NVIDIA)"
        assert d["model"] == "NVIDIA part 0x123"

    def test_utilization_from_proc_stat_deltas(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        _write(tmp_path, "proc/stat", _stat_text({0: (75, 25), 1: (25, 75)}))
        d = tel.read()
        assert d["cpu_util"] == 50.0
        assert "cpu_util" in tel.schema()
        assert "cpu_util_s0" not in d  # single socket: no per-socket fields

    def test_utilization_only_counts_bound_cores(self, th, tmp_path):
        _fake_node(tmp_path, cores_per_socket=4)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1])
        _write(tmp_path, "proc/stat", _stat_text({0: (100, 0), 1: (100, 0), 2: (0, 100), 3: (0, 100)}))
        assert tel.read()["cpu_util"] == 100.0

    def test_clock_mean_and_sample_cap(self, th, tmp_path):
        cpus = _fake_node(tmp_path, cores_per_socket=20)
        for cpu in cpus:
            _write(tmp_path, f"sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq", "2000000\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        assert len(tel._clock_paths[0]) == th.CPU_CLOCK_SAMPLE_CORES_PER_SOCKET
        assert tel.read()["cpu_clock"] == 2000

    def test_coretemp_per_socket_and_max_over_bound_sockets(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        _hwmon(tmp_path, 0, "coretemp", {"temp1_label": "Package id 0\n", "temp1_input": "60000\n",
                                         "temp2_label": "Core 0\n", "temp2_input": "99000\n"})
        _hwmon(tmp_path, 1, "coretemp", {"temp1_label": "Package id 1\n", "temp1_input": "70000\n"})
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1])  # socket 0 only
        d = tel.read()
        assert d["temp_cpu_C"] == 60.0  # core sensors and the unbound socket don't count
        assert d["temp_cpu_C_s0"] == 60.0
        assert d["temp_cpu_C_s1"] == 70.0

    def test_k10temp_maps_by_pci_order_and_prefers_tdie(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        # hwmon numbering deliberately disagrees with PCI order
        for n, pci, tdie in ((0, "0000:00:19.3", "72000"), (1, "0000:00:18.3", "61000")):
            (tmp_path / "devices" / pci).mkdir(parents=True)
            _hwmon(tmp_path, n, "k10temp", {"temp1_label": "Tctl\n", "temp1_input": "90000\n",
                                            "temp2_label": "Tdie\n", "temp2_input": f"{tdie}\n"})
            (tmp_path / f"sys/class/hwmon/hwmon{n}/device").symlink_to(tmp_path / "devices" / pci)
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1, 2, 3]).read()
        assert d["temp_cpu_C_s0"] == 61.0
        assert d["temp_cpu_C_s1"] == 72.0
        assert d["temp_cpu_C"] == 72.0

    def test_k10temp_count_mismatch_is_aggregate_only(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        _hwmon(tmp_path, 0, "k10temp", {"temp1_label": "Tctl\n", "temp1_input": "65000\n"})
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1, 2, 3])
        d = tel.read()
        assert d["temp_cpu_C"] == 65.0
        assert not any(k.startswith("temp_cpu_C_s") for k in d)
        assert not any(k.startswith("temp_cpu_C_s") for k in tel.schema())

    def test_thermal_zone_fallback_filters_non_cpu_zones(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        _write(tmp_path, "sys/class/thermal/thermal_zone0/type", "iwlwifi_1\n")
        _write(tmp_path, "sys/class/thermal/thermal_zone0/temp", "99000\n")
        _write(tmp_path, "sys/class/thermal/thermal_zone1/type", "x86_pkg_temp\n")
        _write(tmp_path, "sys/class/thermal/thermal_zone1/temp", "55000\n")
        assert th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()["temp_cpu_C"] == 55.0

    def test_rapl_wins_over_pm_counters(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        _write(tmp_path, "sys/cray/pm_counters/cpu_power", "312 W 1730000000000000 us\n")
        _rapl(tmp_path, 0, 1000)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        assert tel._power_kind == "rapl"

    def test_pm_counters_used_when_nothing_standard_is_readable(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        _write(tmp_path, "sys/cray/pm_counters/cpu_power", "312 W 1730000000000000 us\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        assert tel._power_kind == "gauge"
        assert tel.read()["power_W"] == 312.0

    def test_pm_counters_node_total_has_no_socket_split(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        _write(tmp_path, "sys/cray/pm_counters/cpu_power", "400 W 1 us\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1])
        d = tel.read()
        assert d["power_W"] == 400.0
        assert not any(k.startswith("power_W_s") for k in d)

    def test_pm_counters_per_socket(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        _write(tmp_path, "sys/cray/pm_counters/cpu_power", "400 W 1 us\n")
        _write(tmp_path, "sys/cray/pm_counters/cpu0_power", "150 W 1 us\n")
        _write(tmp_path, "sys/cray/pm_counters/cpu1_power", "250 W 1 us\n")
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1]).read()  # socket 0 only
        assert d["power_W"] == 150.0
        assert (d["power_W_s0"], d["power_W_s1"]) == (150.0, 250.0)

    def test_rapl_rate_sums_bound_sockets_and_handles_wraparound(self, th, tmp_path, monkeypatch):
        _fake_node(tmp_path, sockets=2)
        clock = iter([100.0, 100.0, 101.0, 102.0])
        monkeypatch.setattr(th.time, "monotonic", lambda: next(clock))
        e0 = _rapl(tmp_path, 0, 1_000_000, max_range=10_000_000)
        e1 = _rapl(tmp_path, 1, 9_500_000, max_range=10_000_000)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1, 2, 3])
        assert tel._power_kind == "rapl"
        e0.write_text("1100000\n")  # +0.1 J
        e1.write_text("300000\n")  # wrapped: +0.8 J
        d = tel.read()  # dt = 1 s
        assert d["power_W_s0"] == pytest.approx(0.1)
        assert d["power_W_s1"] == pytest.approx(0.8)
        assert d["power_W"] == pytest.approx(0.9)

    @pytest.mark.skipif(hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
                        reason="root can read mode-000 files")
    def test_unreadable_rapl_is_skipped(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        _rapl(tmp_path, 0, 1000).chmod(0)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        assert "power_W" not in tel.schema()
        assert "power_W" not in tel.read()

    def test_hwmon_power_uses_cpu_sensors_only(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        _hwmon(tmp_path, 0, "power_meter", {"power1_oem_info": "Module Power Socket 0\n",
                                            "power1_average": "900000000\n",
                                            "power2_oem_info": "CPU Power Socket 0\n",
                                            "power2_average": "200000000\n"})
        _hwmon(tmp_path, 1, "power_meter", {"power1_oem_info": "CPU Power Socket 1\n",
                                            "power1_average": "180000000\n"})
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1, 2, 3]).read()
        assert d["power_W"] == 380.0
        assert (d["power_W_s0"], d["power_W_s1"]) == (200.0, 180.0)

    def test_meminfo(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()
        assert d["mem_total_MB"] == 8192.0
        assert d["mem_free_MB"] == 6144.0
        assert d["mem_used_MB"] == 2048.0

    def test_stats_and_reset(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        temp = _write(tmp_path, "sys/class/thermal/thermal_zone0/temp", "50000\n")
        _write(tmp_path, "sys/class/thermal/thermal_zone0/type", "cpu-thermal\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        tel.read()
        temp.write_text("70000\n")
        tel.read()
        stats = tel.get_stats()
        assert (stats["temp_cpu_C_min"], stats["temp_cpu_C_mean"], stats["temp_cpu_C_max"]) == (50.0, 60.0, 70.0)
        assert "mem_used_MB_mean" in stats
        assert "mem_total_MB_mean" not in stats and "device_id_mean" not in stats
        tel.reset_stats()
        assert tel.get_stats() == {}

    def test_missing_sensor_after_probe_does_not_raise(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        temp = _write(tmp_path, "sys/class/thermal/thermal_zone0/temp", "50000\n")
        _write(tmp_path, "sys/class/thermal/thermal_zone0/type", "cpu-thermal\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        temp.write_text("garbage\n")
        d = tel.read()
        assert "temp_cpu_C" not in d
        assert d["mem_used_MB"] == 2048.0

    def test_gpu_device_fallback_is_identity_only(self, th, tmp_path):
        import torch
        cpus = _fake_node(tmp_path)
        tel = th.CpuTelemetry(0, device=torch.device("cuda"), root=str(tmp_path), cpus=cpus)
        assert tel.schema() == th.CpuTelemetry.IDENTITY_FIELDS
        assert set(tel.read()) == {"vendor", "model", "device_id", "hostname"}

    def test_no_proc_is_identity_only(self, th, tmp_path):
        tel = th.CpuTelemetry(0, root=str(tmp_path))
        assert set(tel.read()) == {"vendor", "model", "device_id", "hostname"}

    def test_make_telemetry_passes_device(self, th):
        import torch
        with patch('shutil.which', return_value=None):
            tel = th.make_telemetry(0, torch.device("cuda"))
        assert isinstance(tel, th.CpuTelemetry)
        assert tel.schema() == th.CpuTelemetry.IDENTITY_FIELDS

    def test_compact_line_shows_cpu_fields(self, th):
        line = th._format_telemetry_compact({
            "vendor": "ARM (NVIDIA)", "model": "ARM Neoverse-V2", "device_id": 0,
            "cpu_util": 97.4, "temp_cpu_C": 61.3, "power_W": 250.0, "cpu_clock": 3100,
        })
        assert line.startswith("ARM Neoverse-V2 | ")
        assert "Util:97%" in line and "Temp:61°C" in line and "Clock:3100MHz" in line

    def test_compact_csv_temp_falls_back_to_cpu(self, th):
        assert th._temp_max_c({"temp_cpu_C_max": 71.6}) == "72"
        assert th._temp_max_c({"temp_gpu_C_max": 80, "temp_cpu_C_max": 71.6}) == "80"
        assert th._temp_max_c({}) == ""


class TestCpuTelemetryTopology:
    """SMT-aware utilization and the benchmark's CPU set on multi-socket nodes."""

    @staticmethod
    def _smt_node(root, sockets=2, cores_per_socket=2):
        """CPUs numbered like Linux on SMT parts: all first threads, then all second threads."""
        n_cores = sockets * cores_per_socket
        for cpu in range(2 * n_cores):
            core = cpu % n_cores
            topo = f"sys/devices/system/cpu/cpu{cpu}/topology"
            _write(root, f"{topo}/physical_package_id", f"{core // cores_per_socket}\n")
            _write(root, f"{topo}/thread_siblings_list", f"{core},{core + n_cores}\n")
        _write(root, "proc/stat", _stat_text({c: (0, 0) for c in range(2 * n_cores)}))
        return list(range(2 * n_cores))

    def test_idle_smt_siblings_do_not_halve_utilization(self, th, tmp_path):
        cpus = self._smt_node(tmp_path, sockets=1)  # cores (0,2) and (1,3)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        _write(tmp_path, "proc/stat", _stat_text({0: (100, 0), 1: (100, 0), 2: (0, 100), 3: (0, 100)}))
        assert tel.read()["cpu_util"] == 100.0

    def test_per_socket_util_covers_unbound_socket(self, th, tmp_path):
        self._smt_node(tmp_path)  # socket 0: cores 0,1; socket 1: cores 2,3
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1, 4, 5])  # socket 0 only
        assert {"cpu_util_s0", "cpu_util_s1"} <= set(tel.schema())
        _write(tmp_path, "proc/stat", _stat_text({0: (10, 90), 1: (10, 90), 2: (100, 0), 3: (100, 0),
                                                  4: (0, 100), 5: (0, 100), 6: (0, 100), 7: (0, 100)}))
        d = tel.read()
        assert d["cpu_util"] == 10.0
        assert (d["cpu_util_s0"], d["cpu_util_s1"]) == (10.0, 100.0)

    def test_cpu_set_is_union_of_thread_masks_and_refreshes(self, th, tmp_path, monkeypatch):
        self._smt_node(tmp_path)
        _hwmon(tmp_path, 0, "coretemp", {"temp1_label": "Package id 0\n", "temp1_input": "50000\n"})
        _hwmon(tmp_path, 1, "coretemp", {"temp1_label": "Package id 1\n", "temp1_input": "80000\n"})
        task = "proc/self/task"
        # OpenMP pins the main thread to core 0 and the telemetry thread to one CPU; workers span socket 0
        _write(tmp_path, f"{task}/100/status", "Name:\tpython\nCpus_allowed_list:\t0,4\n")
        _write(tmp_path, f"{task}/101/status", "Name:\tpython\nCpus_allowed_list:\t5\n")
        worker = _write(tmp_path, f"{task}/102/status", "Name:\tpython\nCpus_allowed_list:\t1\n")
        monkeypatch.setattr(th.os, "sched_getaffinity", lambda pid: {7}, raising=False)
        clock = [1000.0]
        monkeypatch.setattr(th.time, "monotonic", lambda: clock[0])
        tel = th.CpuTelemetry(0, root=str(tmp_path))
        assert tel._cpus == [0, 1, 4, 5]
        assert tel.read()["temp_cpu_C"] == 50.0

        worker.write_text("Name:\tpython\nCpus_allowed_list:\t2-3,6-7\n")  # workers move to socket 1
        assert tel.read()["temp_cpu_C"] == 50.0  # not refreshed yet
        clock[0] += th.CPU_SET_REFRESH_S
        assert tel.read()["temp_cpu_C"] == 80.0
        assert tel._bench_sockets == [0, 1]

    def test_cpu_set_refreshes_immediately_when_threads_appear(self, th, tmp_path, monkeypatch):
        self._smt_node(tmp_path)
        _hwmon(tmp_path, 0, "coretemp", {"temp1_label": "Package id 0\n", "temp1_input": "50000\n"})
        _hwmon(tmp_path, 1, "coretemp", {"temp1_label": "Package id 1\n", "temp1_input": "80000\n"})
        task = "proc/self/task"
        _write(tmp_path, f"{task}/100/status", "Cpus_allowed_list:\t0\n")  # main thread, OMP-pinned
        monkeypatch.setattr(th.time, "monotonic", lambda: 1000.0)  # no time passes
        tel = th.CpuTelemetry(0, root=str(tmp_path))
        assert tel.read()["temp_cpu_C"] == 50.0
        _write(tmp_path, f"{task}/101/status", "Cpus_allowed_list:\t2-3\n")  # OpenMP worker on socket 1
        assert tel.read()["temp_cpu_C"] == 80.0
        assert tel._bench_sockets == [0, 1]

    def test_cpu_set_falls_back_to_process_status(self, th, tmp_path):
        self._smt_node(tmp_path)
        _write(tmp_path, "proc/self/status", "Name:\tpython\nCpus_allowed_list:\t2-3\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path))
        assert tel._cpus == [2, 3] and tel._bench_sockets == [1]

    def test_clock_aggregate_uses_bound_sockets_only(self, th, tmp_path):
        cpus = self._smt_node(tmp_path)
        for cpu in cpus:
            khz = "2000000" if (cpu % 4) < 2 else "3000000"
            _write(tmp_path, f"sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq", khz + "\n")
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1]).read()
        assert d["cpu_clock"] == 2000
        assert (d["cpu_clock_s0"], d["cpu_clock_s1"]) == (2000, 3000)


class TestCpuTelemetryCpuinfoClock:
    """acpi-cpufreq reports the requested P-state; the clock then comes from /proc/cpuinfo."""

    @staticmethod
    def _node(root, mhz):
        cpus = TestCpuTelemetryTopology._smt_node(root)  # 2 sockets x 2 cores x 2 threads
        _write(root, "proc/cpuinfo", "".join(
            f"processor\t: {c}\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC Test CPU\ncpu MHz\t\t: {m}\n\n"
            for c, m in enumerate(mhz)))
        for c in cpus:
            base = f"sys/devices/system/cpu/cpu{c}/cpufreq"
            _write(root, f"{base}/scaling_driver", "acpi-cpufreq\n")
            _write(root, f"{base}/scaling_cur_freq", "2250000\n")
        return cpus

    def test_uses_cpuinfo_with_busiest_sibling(self, th, tmp_path):
        # cores 0,1 on socket 0 (siblings 4,5), cores 2,3 on socket 1 (siblings 6,7)
        cpus = self._node(tmp_path, [3400, 1500, 3000, 3000, 1500, 3600, 3000, 3000])
        d = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()
        assert d["cpu_clock_s0"] == 3500  # core0 3400, core1 max(1500, 3600)
        assert d["cpu_clock_s1"] == 3000
        assert d["cpu_clock"] == 3250

    def test_cpuinfo_refresh_is_rate_limited(self, th, tmp_path, monkeypatch):
        cpus = self._node(tmp_path, [3000] * 8)
        clock = [100.0]
        monkeypatch.setattr(th.time, "monotonic", lambda: clock[0])
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        self._node(tmp_path, [3500] * 8)
        clock[0] += th.CPU_CPUINFO_CLOCK_INTERVAL_S / 2
        assert tel.read()["cpu_clock"] == 3000
        clock[0] += th.CPU_CPUINFO_CLOCK_INTERVAL_S
        assert tel.read()["cpu_clock"] == 3500

    def test_other_drivers_keep_scaling_cur_freq(self, th, tmp_path):
        cpus = self._node(tmp_path, [3400] * 8)
        for c in cpus:
            _write(tmp_path, f"sys/devices/system/cpu/cpu{c}/cpufreq/scaling_driver", "amd-pstate-epp\n")
        assert th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()["cpu_clock"] == 2250


def _hsmp(root, n, acpi, power_uw, cap_uw=500_000_000):
    _hwmon(root, n, "amd_hsmp_hwmon", {"power1_input": f"{power_uw}\n", "power1_cap": f"{cap_uw}\n"})
    (root / "devices" / acpi).mkdir(parents=True, exist_ok=True)
    (root / f"sys/class/hwmon/hwmon{n}/device").symlink_to(root / "devices" / acpi)


class TestCpuTelemetryHsmp:
    """AMD HSMP hwmon: per-socket power and power cap, world-readable on EPYC."""

    def test_per_socket_power_and_cap_in_acpi_order(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        # hwmon numbering deliberately disagrees with ACPI instance order
        _hsmp(tmp_path, 3, "AMDI0097:00", 300_000_000, cap_uw=480_000_000)
        _hsmp(tmp_path, 1, "AMDI0097:01", 250_000_000)
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1])  # socket 0 only
        d = tel.read()
        assert (d["power_W_s0"], d["power_W_s1"]) == (300.0, 250.0)
        assert d["power_W"] == 300.0
        assert (d["power_cap_W_s0"], d["power_cap_W_s1"]) == (480.0, 500.0)
        assert d["power_cap_W"] == 480.0
        assert "power_cap_W" in tel.schema()

    def test_hsmp_wins_over_pm_counters(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        _write(tmp_path, "sys/cray/pm_counters/cpu_power", "312 W 1 us\n")
        _hsmp(tmp_path, 0, "AMDI0097:00", 200_000_000)
        assert th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus).read()["power_W"] == 200.0

    def test_rapl_wins_over_hsmp(self, th, tmp_path):
        cpus = _fake_node(tmp_path)
        _rapl(tmp_path, 0, 1000)
        _hsmp(tmp_path, 0, "AMDI0097:00", 200_000_000)
        assert th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)._power_kind == "rapl"

    def test_socket_count_mismatch_falls_through(self, th, tmp_path):
        _fake_node(tmp_path, sockets=2)
        _hsmp(tmp_path, 0, "AMDI0097:00", 200_000_000)
        _write(tmp_path, "sys/cray/pm_counters/cpu_power", "400 W 1 us\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=[0, 1, 2, 3])
        d = tel.read()
        assert d["power_W"] == 400.0
        assert "power_cap_W" not in d

    def test_cap_is_read_rarely(self, th, tmp_path, monkeypatch):
        cpus = _fake_node(tmp_path)
        _hsmp(tmp_path, 0, "AMDI0097:00", 200_000_000, cap_uw=500_000_000)
        clock = [100.0]
        monkeypatch.setattr(th.time, "monotonic", lambda: clock[0])
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        assert tel.read()["power_cap_W"] == 500.0
        (tmp_path / "sys/class/hwmon/hwmon0/power1_cap").write_text("400000000\n")
        assert tel.read()["power_cap_W"] == 500.0
        clock[0] += th.CPU_SET_REFRESH_S
        assert tel.read()["power_cap_W"] == 400.0


class TestCpuTelemetryRateLimit:
    """/proc/stat is parsed at most once per CPU_STAT_INTERVAL_S; values repeat in between."""

    def test_utilization_refreshes_once_per_interval(self, th, tmp_path, monkeypatch):
        cpus = _fake_node(tmp_path)
        clock = [100.0]
        monkeypatch.setattr(th.time, "monotonic", lambda: clock[0])
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        _write(tmp_path, "proc/stat", _stat_text({0: (75, 25), 1: (25, 75)}))
        assert tel.read()["cpu_util"] == 50.0           # first read computes immediately
        _write(tmp_path, "proc/stat", _stat_text({0: (175, 25), 1: (125, 75)}))
        clock[0] += th.CPU_STAT_INTERVAL_S / 2
        with patch.object(tel, "_read_proc_stat", wraps=tel._read_proc_stat) as rd:
            assert tel.read()["cpu_util"] == 50.0       # repeated, file not parsed
            rd.assert_not_called()
            clock[0] += th.CPU_STAT_INTERVAL_S
            assert tel.read()["cpu_util"] == 100.0      # refreshed over the whole window
            rd.assert_called_once()

    def test_power_still_follows_every_sample(self, th, tmp_path, monkeypatch):
        cpus = _fake_node(tmp_path)
        clock = [100.0]
        monkeypatch.setattr(th.time, "monotonic", lambda: clock[0])
        pm = _write(tmp_path, "sys/cray/pm_counters/cpu_power", "100 W 1 us\n")
        tel = th.CpuTelemetry(0, root=str(tmp_path), cpus=cpus)
        assert tel.read()["power_W"] == 100.0
        pm.write_text("250 W 2 us\n")
        clock[0] += 0.1
        assert tel.read()["power_W"] == 250.0
