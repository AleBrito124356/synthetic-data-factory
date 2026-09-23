"""Command-line entry points: ``main([...])``, ``python -m factory``, ``sdf``."""
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

from factory.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMAS = os.path.join(ROOT, "schemas")


def _clean_env():
    env = dict(os.environ)
    env.pop("NVIDIA_API_KEY", None)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    return env


def test_main_generate_tabular_writes_files(tmp_path, capsys):
    rc = main([
        "generate", "tabular",
        "--schema", os.path.join(SCHEMAS, "ecommerce.yaml"),
        "--out", str(tmp_path),
        "--format", "csv",
        "--validate",
    ])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert (tmp_path / "customers.csv").exists()
    assert "checks passed (OK)" in out


def test_main_validate_and_report(capsys):
    assert main(["validate", "--schema", os.path.join(SCHEMAS, "healthcare.yaml")]) == 0
    assert main(["report", "--schema", os.path.join(SCHEMAS, "saas-users.yaml")]) == 0
    out = capsys.readouterr().out
    assert "# Distribution report" in out


def test_main_schema_error_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tables: [{name: t, rows: 1, fields: [{name: x, type: wat}]}]\n", encoding="utf-8")
    assert main(["validate", "--schema", str(bad)]) == 1
    assert "unknown type 'wat'" in capsys.readouterr().err


def test_python_dash_m_factory(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "factory", "--help"],
        capture_output=True, text=True, env=_clean_env(), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: sdf" in proc.stdout


def test_root_cli_shim(tmp_path):
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "cli.py"), "--help"],
        capture_output=True, text=True, env=_clean_env(), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: sdf" in proc.stdout


def _installed_sdf():
    scripts = sysconfig.get_path("scripts")
    for name in ("sdf", "sdf.exe"):
        candidate = os.path.join(scripts, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("sdf")


@pytest.mark.skipif(_installed_sdf() is None, reason="package not installed (pip install -e .)")
def test_installed_sdf_entry_point(tmp_path):
    proc = subprocess.run(
        [_installed_sdf(), "generate", "tabular",
         "--schema", os.path.join(SCHEMAS, "saas-users.yaml"),
         "--out", str(tmp_path), "--format", "csv", "--validate"],
        capture_output=True, text=True, env=_clean_env(), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "accounts.csv").exists()


@pytest.mark.skipif(_installed_sdf() is None, reason="package not installed (pip install -e .)")
def test_installed_sdf_infer_and_validate_data(tmp_path):
    env = _clean_env()
    real = tmp_path / "real"
    steps = [
        ["generate", "tabular", "--schema", os.path.join(SCHEMAS, "healthcare.yaml"),
         "--out", str(real), "--format", "sqlite"],
        ["infer", "--data", str(real / "dataset.db"), "--out", str(tmp_path / "inferred.yaml"),
         "--rows-scale", "0.5"],
        ["generate", "tabular", "--schema", str(tmp_path / "inferred.yaml"),
         "--out", str(tmp_path / "synthetic"), "--format", "csv"],
        ["validate", "--schema", str(tmp_path / "inferred.yaml"), "--data", str(tmp_path / "synthetic")],
    ]
    outputs = []
    for step in steps:
        proc = subprocess.run([_installed_sdf()] + step, capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outputs.append(proc.stdout)
    assert "foreign key: encounters.patient_uuid -> patients.patient_uuid" in outputs[1]
    assert "patients: 400 -> 200 rows" in outputs[1]
    assert "checks passed (OK)" in outputs[3]


def test_validate_data_bad_path(tmp_path, capsys):
    assert main(["validate", "--schema", os.path.join(SCHEMAS, "ecommerce.yaml"), "--data", str(tmp_path)]) == 1
    assert "Data error" in capsys.readouterr().err
    assert main(["infer", "--data", str(tmp_path), "--out", str(tmp_path / "x.yaml")]) == 1
