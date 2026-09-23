# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Start-up behaviour tests for torch-hammer.

These tests run torch-hammer.py in a fresh subprocess rather than through the
``th`` fixture, because the behaviour under test happens at import time
(before argparse runs) and depends on the interpreter's warning filters and
module search path -- neither of which can be reset cleanly inside the pytest
process once torch has been imported.

NumPy is blocked with a stub module placed on PYTHONPATH rather than with
``sys.modules["numpy"] = None``. The stub raises ModuleNotFoundError exactly
as a genuinely missing package does, which makes PyTorch emit the reporter's
byte-exact message ("Failed to initialize NumPy: No module named 'numpy'")
regardless of whether NumPy happens to be installed on the test machine.
(``sys.modules["numpy"] = None`` produces a different message and a plain
ImportError crashes ``import torch`` outright.)

See GitHub issue #41.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SCRIPT = ROOT_DIR / "torch-hammer.py"

NUMPY_MISSING_WARNING = "Failed to initialize NumPy"
CANARY_WARNING = "torch-hammer-canary"

# Runs the script to argparse's --help SystemExit, then emits an unrelated
# UserWarning in the same process to prove the filter is targeted rather
# than a blanket ignore.
PAYLOAD = textwrap.dedent(
    f"""
    import runpy, sys, warnings
    sys.argv = ["torch-hammer.py", "--help"]
    try:
        runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
    except SystemExit:
        pass
    warnings.warn({CANARY_WARNING!r}, UserWarning)
    """
)


def _environment_without_numpy(stub_dir: Path) -> dict:
    """Return a child environment whose import path shadows numpy with a stub."""
    (stub_dir / "numpy.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'numpy'\")\n"
    )
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(stub_dir) + (os.pathsep + existing_path if existing_path else "")
    env.pop("PYTHONWARNINGS", None)
    return env


class TestStartupWarnings:
    """Warnings emitted (or not) while torch-hammer.py starts up."""

    def test_numpy_warning_is_suppressed_but_others_are_not(self, tmp_path):
        env = _environment_without_numpy(tmp_path)

        result = subprocess.run(
            [sys.executable, "-c", PAYLOAD],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"script did not exit cleanly (rc={result.returncode}); stderr:\n{result.stderr}"
        )
        assert NUMPY_MISSING_WARNING not in result.stderr, (
            f"missing-NumPy warning leaked to the user; stderr:\n{result.stderr}"
        )
        assert CANARY_WARNING in result.stderr, (
            f"filter is too broad -- an unrelated UserWarning was swallowed; stderr:\n{result.stderr}"
        )
        assert "usage:" in result.stdout, (
            f"script never reached argparse --help; stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
