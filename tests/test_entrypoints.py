"""Exercise credential safety in isolated CLI processes without network access."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["main.py", "run_jev_trial.py"])
def test_cli_configures_logging_before_sdk_import_and_hides_unexpected_errors(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    # The fake CLI replaces all simulation/network work. Assert setup was applied
    # before the CLI was imported, then raise an exception containing a fake key.
    code = """
import importlib.abc
import importlib.util
import logging
import os
import runpy
import sys

class FakeCLI(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "jev_rl.cli":
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        assert os.environ["TYPESAFE_LOG_LEVEL"] == "off"
        assert os.environ["TYPESAFE_BASE_URL"] == "https://api.typesafe.ai"
        assert logging.root.manager.disable >= logging.CRITICAL
        print("runtime-configured-before-cli-import")
        def fail(*args, **kwargs):
            logging.critical(os.environ["TYPESAFE_API_KEY"])
            raise RuntimeError(os.environ["TYPESAFE_API_KEY"])
        module.main = fail

sys.meta_path.insert(0, FakeCLI())
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
"""
    environment = os.environ.copy()
    environment.update(
        TYPESAFE_API_KEY="synthetic-cli-test-credential",
        TYPESAFE_LOG_LEVEL="debug",
        TYPESAFE_BASE_URL="https://invalid.example",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root / script)]
        + (["--output-dir", str(tmp_path / "trial")] if script == "run_jev_trial.py" else []),
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "runtime-configured-before-cli-import" in result.stdout
    assert "Raw error details are suppressed" in result.stderr
    assert "synthetic-cli-test-credential" not in output
    assert "Traceback" not in output
