"""Check the documented executable entry points."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script,args,required",
    [
        (
            "run_binary.py",
            ["--method", "monomial"],
            {"score", "companion", "bound", "log_e_value"},
        ),
        ("run_binary.py", ["--method", "pooled"], {"score", "companion", "bound", "log_e_value"}),
        ("run_volume.py", [], {"score", "companion"}),
        ("run_occupancy.py", [], {"score", "companion", "draws", "success"}),
        ("run_degree.py", ["--target", "union"], {"q", "primitive_scores", "companion_score"}),
    ],
)
def test_entrypoint_executes(script, args, required):
    run = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    assert required <= json.loads(run.stdout).keys()


def test_binary_cli_rejects_invalid_inputs_without_traceback():
    run = subprocess.run(
        [sys.executable, str(ROOT / "examples/run_binary.py"), "--counts", "-1", "2", "3", "4"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    assert "error:" in run.stderr and "Traceback" not in run.stderr
