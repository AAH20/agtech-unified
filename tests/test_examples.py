"""Tests for examples/quickstart.py and docs/deployment.md."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_quickstart_runs_without_error():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_quickstart_outputs_all_modules():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "TSP:" in result.stdout
    assert "VRP:" in result.stdout
    assert "Sensors:" in result.stdout
    assert "Twin:" in result.stdout
    assert "Decisions:" in result.stdout


def test_quickstart_under_100_lines():
    lines = (ROOT / "examples" / "quickstart.py").read_text().splitlines()
    assert len(lines) < 100


def test_deployment_doc_exists():
    assert (ROOT / "docs" / "deployment.md").is_file()


def test_deployment_doc_over_100_lines():
    lines = (ROOT / "docs" / "deployment.md").read_text().splitlines()
    assert len(lines) > 100
