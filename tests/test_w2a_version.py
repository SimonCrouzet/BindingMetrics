"""``binding_metrics.__version__`` comes from the installed package metadata."""

import importlib.metadata
import re
import subprocess
import sys
import textwrap

import binding_metrics


def test_version_matches_installed_metadata():
    assert binding_metrics.__version__ == importlib.metadata.version("binding-metrics")


def test_version_of_an_uninstalled_source_tree_is_a_valid_placeholder(tmp_path):
    """Without metadata the attribute still exists and is a PEP 440 version string."""
    script = textwrap.dedent(
        """
        import importlib.metadata

        def not_installed(name):
            raise importlib.metadata.PackageNotFoundError(name)

        importlib.metadata.version = not_installed

        import binding_metrics

        print(binding_metrics.__version__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert re.fullmatch(r"\d+\.\d+\.\d+\+\w+", result.stdout.strip())
