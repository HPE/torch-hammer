# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CPU operand fill (random_operand) and CPU memory interleaving.
"""
import logging
import sys
from unittest.mock import patch

import pytest
import torch


LOG = logging.getLogger("test_cpu_fill")


# ── random_operand: non-CPU devices are unchanged ──────────────────────

class TestRandomOperandNonCpu:
    """GPU (and any non-CPU) runs must make exactly the torch.rand call they made before."""

    @pytest.mark.parametrize("dev", [torch.device("cuda", 1), torch.device("mps"), torch.device("meta")])
    def test_identical_torch_rand_call(self, th, dev):
        sentinel = object()
        with patch.object(th.torch, "rand", return_value=sentinel) as rand, \
             patch.object(th, "_parallel_rand_cpu") as cpu_fill:
            out = th.random_operand((4, 8, 16), torch.bfloat16, dev)
        assert out is sentinel
        rand.assert_called_once_with(4, 8, 16, device=dev, dtype=torch.bfloat16)
        cpu_fill.assert_not_called()

    def test_spread_flag_is_ignored_off_cpu(self, th):
        dev = torch.device("cuda", 0)
        with patch.object(th.torch, "rand", return_value="t") as rand:
            th.random_operand((3, 5), torch.float32, dev, spread=False)
        rand.assert_called_once_with(3, 5, device=dev, dtype=torch.float32)

    def test_meta_device_really_runs(self, th):
        out = th.random_operand((2, 3), torch.float32, torch.device("meta"))
        assert out.device.type == "meta" and out.shape == (2, 3)


# ── random_operand: CPU parallel fill ──────────────────────────────────

@pytest.fixture
def small_chunks(th, monkeypatch):
    """Tiny chunks so small test tensors still span many chunks."""
    monkeypatch.setattr(th, "CPU_FILL_CHUNK_ELEMS", 1000)


class TestRandomOperandCpu:

    def test_same_seed_same_values_whatever_the_thread_count(self, th, small_chunks):
        results = []
        for workers in (1, 3, 8):
            torch.manual_seed(1234)
            results.append(th._parallel_rand_cpu((7, 1001), torch.float32, workers=workers))
        assert all(torch.equal(results[0], r) for r in results[1:])

    def test_follows_torch_manual_seed(self, th, small_chunks):
        torch.manual_seed(1)
        a = th.random_operand((50, 100), torch.float32, torch.device("cpu"))
        torch.manual_seed(1)
        b = th.random_operand((50, 100), torch.float32, torch.device("cpu"))
        torch.manual_seed(2)
        c = th.random_operand((50, 100), torch.float32, torch.device("cpu"))
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_consecutive_operands_differ(self, th, small_chunks):
        torch.manual_seed(0)
        a = th.random_operand((40, 100), torch.float32, torch.device("cpu"))
        b = th.random_operand((40, 100), torch.float32, torch.device("cpu"))
        assert not torch.equal(a, b)

    def test_chunks_are_not_copies_of_each_other(self, th, small_chunks):
        torch.manual_seed(0)
        flat = th._parallel_rand_cpu((5000,), torch.float32, workers=4)
        assert not torch.equal(flat[:1000], flat[1000:2000])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                       torch.float64, torch.complex64])
    def test_dtypes_shape_and_range(self, th, small_chunks, dtype):
        out = th.random_operand((3, 4, 555), dtype, torch.device("cpu"))
        assert out.dtype == dtype and out.shape == (3, 4, 555) and out.is_contiguous()
        parts = (out.real, out.imag) if dtype.is_complex else (out,)
        for part in parts:
            assert float(part.min()) >= 0.0 and float(part.max()) < 1.0
            assert 0.4 < float(part.float().mean()) < 0.6

    def test_shape_not_a_multiple_of_chunk(self, th, small_chunks):
        torch.manual_seed(5)
        out = th._parallel_rand_cpu((2501,), torch.float32, workers=3)
        assert float(out[-1]) != 0.0 and float(out[2000:].min()) >= 0.0

    def test_spread_false_fills_on_the_calling_thread(self, th):
        with patch.object(th, "_parallel_rand_cpu") as pool, \
             patch.object(th.torch, "rand", wraps=th.torch.rand) as rand:
            out = th.random_operand((4, 6), torch.float64, torch.device("cpu"), spread=False)
        pool.assert_not_called()
        rand.assert_called_once_with(4, 6, dtype=torch.float64)
        assert out.shape == (4, 6) and out.dtype == torch.float64

    def test_spread_false_follows_seed(self, th):
        torch.manual_seed(11)
        a = th.random_operand((100,), torch.float32, torch.device("cpu"), spread=False)
        torch.manual_seed(11)
        b = th.random_operand((100,), torch.float32, torch.device("cpu"), spread=False)
        assert torch.equal(a, b)

    def test_zero_elements(self, th):
        out = th.random_operand((0, 16), torch.float32, torch.device("cpu"))
        assert out.shape == (0, 16)

    def test_matches_benchmark_call_signature(self, th):
        # Benchmarks pass a tuple shape and a torch.device("cpu"); the result must be a normal CPU tensor
        out = th.random_operand((2, 3), torch.float32, torch.device("cpu"))
        assert out.device.type == "cpu"


# ── CPU memory interleave ──────────────────────────────────────────────

def _node_tree(root, nodes, has_memory):
    """Fake /sys/devices/system/node: nodes = {node: 'cpulist'}."""
    for node, cpulist in nodes.items():
        (root / f"node{node}").mkdir(parents=True)
        (root / f"node{node}" / "cpulist").write_text(cpulist + "\n")
    (root / "has_memory").write_text(has_memory + "\n")
    (root / "possible").write_text("0-15\n")  # non-node entries must be ignored
    return str(root)


# 4 NUMA nodes x 4 cores, 2 hardware threads per core (siblings numbered +16)
FOUR_NUMA_NODES = {0: "0-3,16-19", 1: "4-7,20-23", 2: "8-11,24-27", 3: "12-15,28-31"}


class TestNumaNodesForCpus:

    def test_whole_node(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        assert th.numa_nodes_for_cpus(range(32), node_dir) == [0, 1, 2, 3]

    def test_restricted_allocation(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        assert th.numa_nodes_for_cpus([2, 21], node_dir) == [0, 1]

    def test_cpu_less_memory_node_excluded(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, {**FOUR_NUMA_NODES, 4: ""}, "0-4")  # e.g. CXL expander
        assert th.numa_nodes_for_cpus(range(32), node_dir) == [0, 1, 2, 3]

    def test_memory_less_node_excluded(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0,2-3")
        assert th.numa_nodes_for_cpus(range(32), node_dir) == [0, 2, 3]

    def test_no_sysfs(self, th, tmp_path):
        assert th.numa_nodes_for_cpus(range(8), str(tmp_path / "missing")) == []


class TestConfigureCpuFill:

    @pytest.fixture(autouse=True)
    def _reset(self, th, monkeypatch):
        monkeypatch.setattr(th, "_CPU_FILL_PLACEMENT", None)

    def test_records_cpus_and_nodes(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        with patch.object(th, "_interleave_works", return_value=True):
            assert th.configure_cpu_fill(range(32), True, LOG, node_dir) == [0, 1, 2, 3]
        assert th._CPU_FILL_PLACEMENT == (list(range(32)), [0, 1, 2, 3])

    def test_never_sets_policy_on_the_calling_thread(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        callers = []
        with patch.object(th, "_set_mempolicy_interleave",
                          side_effect=lambda nodes: callers.append(th.threading.get_ident())):
            th.configure_cpu_fill(range(32), True, LOG, node_dir)
        assert callers and th.threading.get_ident() not in callers

    def test_interleave_off(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        with patch.object(th, "_interleave_works") as works:
            assert th.configure_cpu_fill(range(32), False, LOG, node_dir) == []
        works.assert_not_called()
        assert th._CPU_FILL_PLACEMENT == (list(range(32)), [])

    def test_single_node_is_not_interleaved(self, th, tmp_path):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        assert th.configure_cpu_fill([0, 16], True, LOG, node_dir) == []

    def test_policy_failure_is_a_warning_not_an_error(self, th, tmp_path, caplog):
        node_dir = _node_tree(tmp_path, FOUR_NUMA_NODES, "0-3")
        with patch.object(th, "_set_mempolicy_interleave", side_effect=OSError(1, "Operation not permitted")):
            with caplog.at_level(logging.WARNING):
                assert th.configure_cpu_fill(range(32), True, LOG, node_dir) == []
        assert "could not interleave" in caplog.text
        assert th._CPU_FILL_PLACEMENT == (list(range(32)), [])

    @pytest.mark.skipif(sys.platform == "linux", reason="checks the non-Linux refusal")
    def test_syscall_refused_off_linux(self, th):
        with pytest.raises(OSError):
            th._set_mempolicy_interleave([0, 1])


class TestFillThreadPlacement:
    """Only the pool threads are re-pinned and get the policy; the main thread is untouched."""

    def test_pool_threads_get_affinity_and_policy(self, th, monkeypatch):
        monkeypatch.setattr(th, "CPU_FILL_CHUNK_ELEMS", 1000)
        monkeypatch.setattr(th, "_CPU_FILL_PLACEMENT", ([0, 1, 2, 3], [0, 1]))
        main = th.threading.get_ident()
        seen = {"aff": [], "pol": []}
        monkeypatch.setattr(th.os, "sched_setaffinity",
                            lambda pid, cpus: seen["aff"].append((th.threading.get_ident(), list(cpus))), raising=False)
        monkeypatch.setattr(th, "_set_mempolicy_interleave",
                            lambda nodes: seen["pol"].append((th.threading.get_ident(), nodes)))
        th._parallel_rand_cpu((10_000,), torch.float32, workers=4)
        assert len(seen["aff"]) == 4 and len(seen["pol"]) == 4
        assert all(tid != main and cpus == [0, 1, 2, 3] for tid, cpus in seen["aff"])
        assert all(tid != main and nodes == [0, 1] for tid, nodes in seen["pol"])

    def test_no_placement_leaves_threads_alone(self, th, monkeypatch):
        monkeypatch.setattr(th, "CPU_FILL_CHUNK_ELEMS", 1000)
        monkeypatch.setattr(th, "_CPU_FILL_PLACEMENT", None)
        with patch.object(th, "_set_mempolicy_interleave") as pol:
            th._parallel_rand_cpu((10_000,), torch.float32, workers=4)
        pol.assert_not_called()

    def test_placement_failure_still_fills(self, th, monkeypatch):
        monkeypatch.setattr(th, "CPU_FILL_CHUNK_ELEMS", 1000)
        monkeypatch.setattr(th, "_CPU_FILL_PLACEMENT", ([0], [0, 1]))
        monkeypatch.setattr(th, "_set_mempolicy_interleave", lambda nodes: (_ for _ in ()).throw(OSError(1, "nope")))
        monkeypatch.setattr(th.os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
        torch.manual_seed(9)
        out = th._parallel_rand_cpu((10_000,), torch.float32, workers=4)
        assert float(out.min()) >= 0.0 and float(out[-1]) != 0.0


class TestSetupCpuThreadingInterleave:

    @staticmethod
    def _args(parser, *flags):
        return parser.parse_args(list(flags))

    def test_cpu_run_configures_fill_with_interleave(self, th, parser):
        with patch.object(th, "configure_cpu_fill") as cfg, patch.object(th.torch, "set_num_threads"):
            th.setup_cpu_threading(torch.device("cpu"), self._args(parser), LOG)
        assert cfg.call_args.args[1] is True

    def test_flag_disables_interleave(self, th, parser):
        with patch.object(th, "configure_cpu_fill") as cfg, patch.object(th.torch, "set_num_threads"):
            th.setup_cpu_threading(torch.device("cpu"), self._args(parser, "--no-cpu-interleave"), LOG)
        assert cfg.call_args.args[1] is False

    def test_cpu_list_path_uses_its_cpus(self, th, parser):
        with patch.object(th, "configure_cpu_fill") as cfg, patch.object(th.torch, "set_num_threads"), \
             patch.object(th.os, "sched_setaffinity", create=True):
            th.setup_cpu_threading(torch.device("cpu"), self._args(parser, "--cpu-list", "0-3,8"), LOG)
        assert list(cfg.call_args.args[0]) == [0, 1, 2, 3, 8]

    @pytest.mark.parametrize("dev", [torch.device("cuda", 0), torch.device("mps")])
    def test_gpu_runs_never_touch_cpu_fill(self, th, parser, dev):
        with patch.object(th, "configure_cpu_fill") as cfg, patch.object(th.torch, "set_num_threads") as snt:
            th.setup_cpu_threading(dev, self._args(parser), LOG)
        cfg.assert_not_called()
        snt.assert_not_called()


class TestCpuInterleaveFlag:

    def test_default_on(self, parser):
        assert parser.parse_args([]).cpu_interleave is True

    def test_cli_off(self, parser):
        assert parser.parse_args(["--no-cpu-interleave"]).cpu_interleave is False

    def test_yaml_off(self, th, parser):
        orig = sys.argv
        sys.argv = ["torch-hammer.py"]
        try:
            args = th.apply_config_to_args(parser.parse_args([]), {"global": {"no_cpu_interleave": True}})
        finally:
            sys.argv = orig
        assert args.cpu_interleave is False
