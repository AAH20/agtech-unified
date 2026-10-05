"""Tests for the onboarding CLI (src/onboarding/cli.py).

Covers:
- `agtech onboard init` command
- `agtech onboard status` command
- `agtech onboard validate` command
"""

from __future__ import annotations

import json

import pytest

from src.onboarding.cli import main


class TestOnboardInit:
    """Tests for `agtech onboard init`."""

    def test_init_requires_employees(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--revenue", "100000"])
        assert exc_info.value.code == 2

    def test_init_requires_revenue(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "100"])
        assert exc_info.value.code == 2

    def test_init_prints_tier(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "onboard",
                    "init",
                    "--employees",
                    "5",
                    "--revenue",
                    "50000",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "startup" in out

    def test_init_prints_modules(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "5", "--revenue", "50000"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "basics" in out

    def test_init_prints_install_order(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "5", "--revenue", "50000"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "install_order" in out.lower() or "Install order" in out

    def test_init_smb_tier(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "100", "--revenue", "1000000"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "smb" in out

    def test_init_enterprise_tier(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "1000", "--revenue", "100000000"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "enterprise" in out

    def test_init_negative_employees_returns_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "-1", "--revenue", "100000"])
        assert exc_info.value.code == 1

    def test_init_zero_employees_ok(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "init", "--employees", "0", "--revenue", "0"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "startup" in out

    def test_init_with_fields_sensors(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "onboard",
                    "init",
                    "--employees",
                    "8",
                    "--revenue",
                    "200000",
                    "--fields",
                    "30",
                    "--sensors",
                    "50",
                    "--dedicated-it",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "tier" in out.lower() or "Tier" in out

    def test_init_json_output(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "onboard",
                    "init",
                    "--employees",
                    "5",
                    "--revenue",
                    "50000",
                    "--json",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["tier"] == "startup"
        assert "modules" in data
        assert "install_order" in data

    def test_init_json_output_no_color_codes(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "onboard",
                    "init",
                    "--employees",
                    "5",
                    "--revenue",
                    "50000",
                    "--json",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "\033[" not in out


class TestOnboardStatus:
    """Tests for `agtech onboard status`."""

    def test_status_returns_0(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "status"])
        assert exc_info.value.code == 0

    def test_status_prints_not_initialized(self, capsys):
        with pytest.raises(SystemExit):
            main(["onboard", "status"])
        out = capsys.readouterr().out
        assert "not initialized" in out.lower() or "No active" in out

    def test_status_prints_status_line(self, capsys):
        with pytest.raises(SystemExit):
            main(["onboard", "status"])
        out = capsys.readouterr().out
        assert "Status:" in out or "status" in out.lower()


class TestOnboardValidate:
    """Tests for `agtech onboard validate`."""

    def test_validate_returns_0(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard", "validate"])
        assert exc_info.value.code == 0

    def test_validate_prints_success(self, capsys):
        with pytest.raises(SystemExit):
            main(["onboard", "validate"])
        out = capsys.readouterr().out
        assert "valid" in out.lower() or "OK" in out

    def test_validate_with_explicit_config(self, capsys):
        import tempfile
        from pathlib import Path

        config = {
            "tier": "startup",
            "modules": ["basics", "sensors", "alerts"],
            "features": ["sensor_monitoring", "basic_alerts"],
            "database": {"type": "sqlite", "path": "./data/agtech.db", "pool_size": 1},
            "api": {"host": "localhost", "port": 8000, "workers": 1, "auth": "basic"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["onboard", "validate", "--config", config_path])
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            assert "valid" in out.lower() or "OK" in out
        finally:
            Path(config_path).unlink()

    def test_validate_with_invalid_config(self, capsys):
        import tempfile
        from pathlib import Path

        config = {"tier": "nonexistent"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["onboard", "validate", "--config", config_path])
            assert exc_info.value.code == 1
        finally:
            Path(config_path).unlink()


class TestOnboardNoCommand:
    """Tests for running `agtech onboard` without a subcommand."""

    def test_no_subcommand_prints_usage(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["onboard"])
        assert exc_info.value.code == 2

    def test_no_args_prints_usage(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2
