"""The 1YCR fixture ships with the package, so make_synthetic_poses works outside a checkout.

A wheel holds the Python modules of ``src/binding_metrics`` and the files that
``[tool.setuptools.package-data]`` lists, and nothing from the repository's ``data/`` or
``tests/``. These tests run setuptools' ``build_py`` step, which is the step that decides the
contents of a wheel, into a temporary directory (no wheel is built, nothing is installed, the
working tree is not written to) and use the result as an installed package: first as a
directory on ``PYTHONPATH``, then zipped.
"""

import fnmatch
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
import zipfile
from pathlib import Path

import pytest

from binding_metrics.testing import synthetic

REPO = Path(__file__).resolve().parents[2]
NAME = synthetic.BUNDLED_COMPLEX_NAME
PACKAGE_DIR = Path(synthetic.__file__).resolve().parent
PACKAGED_COPY = PACKAGE_DIR / "data" / NAME
CHECKOUT_COPY = REPO / "data" / NAME

needs_checkout = pytest.mark.skipif(
    not (REPO / "pyproject.toml").is_file() or not CHECKOUT_COPY.is_file(),
    reason="needs a source checkout with pyproject.toml and data/",
)


def run_python(code, *, pythonpath, cwd):
    """Run ``code`` in a fresh interpreter whose imports start at ``pythonpath``."""
    environment = {**os.environ, "PYTHONPATH": str(pythonpath), "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()


@pytest.fixture(scope="module")
def built_layout(tmp_path_factory):
    """The directory that setuptools would zip into a wheel: ``build_py`` output."""
    pytest.importorskip("setuptools")
    if not (REPO / "pyproject.toml").is_file():
        pytest.skip("needs a source checkout with pyproject.toml")
    project = tmp_path_factory.mktemp("project")
    shutil.copy(REPO / "pyproject.toml", project / "pyproject.toml")
    shutil.copy(REPO / "README.md", project / "README.md")
    shutil.copytree(
        REPO / "src" / "binding_metrics",
        project / "src" / "binding_metrics",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    library = tmp_path_factory.mktemp("library")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import setuptools; setuptools.setup()",
            "-q",
            "build_py",
            "--build-lib",
            str(library),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return library


class TestDeclaration:
    @needs_checkout
    def test_pyproject_lists_the_file_as_package_data_of_the_testing_package(self):
        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        package_data = config["tool"]["setuptools"]["package-data"]
        assert "binding_metrics.testing" in package_data
        listed = [
            path.relative_to(PACKAGE_DIR).as_posix()
            for path in (PACKAGE_DIR / "data").iterdir()
            if any(
                fnmatch.fnmatch(path.relative_to(PACKAGE_DIR).as_posix(), pattern)
                for pattern in package_data["binding_metrics.testing"]
            )
        ]
        assert f"data/{NAME}" in listed

    def test_the_package_copy_exists_and_is_1ycr(self):
        text = PACKAGED_COPY.read_text(encoding="utf-8")
        assert text.startswith("HEADER") and " 1YCR" in text.splitlines()[0]

    @needs_checkout
    def test_the_package_copy_is_the_file_of_the_repository_data_directory(self):
        assert PACKAGED_COPY.read_bytes() == CHECKOUT_COPY.read_bytes()


class TestResolution:
    def test_the_package_copy_is_found_first(self):
        assert synthetic.bundled_complex_path() == PACKAGED_COPY
        assert synthetic._candidate_paths()[0] == PACKAGED_COPY

    def test_the_package_copy_wins_over_the_source_tree(self, monkeypatch, tmp_path):
        stand_in = tmp_path / "data" / NAME
        stand_in.parent.mkdir()
        stand_in.write_text("not read here\n", encoding="utf-8")
        monkeypatch.setattr(synthetic.resources, "files", lambda package: tmp_path)
        assert synthetic.bundled_complex_path() == stand_in

    @needs_checkout
    def test_a_source_tree_without_the_package_copy_reads_the_repository_data(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(synthetic.resources, "files", lambda package: tmp_path)
        assert synthetic.bundled_complex_path().resolve() == CHECKOUT_COPY.resolve()

    def test_nothing_anywhere_raises_with_the_way_out(self, monkeypatch, tmp_path):
        monkeypatch.setattr(synthetic.resources, "files", lambda package: tmp_path)
        monkeypatch.setattr(
            synthetic,
            "_candidate_paths",
            lambda: [synthetic._packaged_complex_path() or tmp_path / "absent.pdb"],
        )
        with pytest.raises(FileNotFoundError, match="complex_path"):
            synthetic.bundled_complex_path()

    def test_an_unknown_package_is_none_and_not_an_error(self, monkeypatch):
        def refuse(package):
            raise ModuleNotFoundError(package)

        monkeypatch.setattr(synthetic.resources, "files", refuse)
        assert synthetic._packaged_complex_path() is None


class TestWheelLayout:
    def test_the_built_package_holds_the_modules_and_the_complex_and_nothing_else(
        self, built_layout
    ):
        top = sorted(path.name for path in built_layout.iterdir())
        assert top == ["binding_metrics"]
        testing = built_layout / "binding_metrics" / "testing"
        assert (testing / "synthetic.py").is_file()
        assert (testing / "data" / NAME).read_bytes() == PACKAGED_COPY.read_bytes()
        assert sorted(path.name for path in (testing / "data").iterdir()) == [NAME]

    def test_an_installed_layout_makes_the_synthetic_poses_without_the_repository(
        self, built_layout, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        lines = run_python(
            """
            import binding_metrics
            from importlib import resources
            from binding_metrics.testing import bundled_complex_path, make_synthetic_poses

            path = bundled_complex_path()
            poses = make_synthetic_poses(seeds=(1, 2))
            resource = resources.files("binding_metrics.testing") / "data" / path.name
            print(binding_metrics.__file__)
            print(path)
            print(resource.is_file(), len(poses.poses), poses.reference.pdb_id)
            """,
            pythonpath=built_layout,
            cwd=elsewhere,
        )
        package_file, path, facts = lines[-3:]
        assert Path(package_file).resolve().is_relative_to(built_layout.resolve())
        assert Path(path).resolve() == (built_layout / "binding_metrics/testing/data" / NAME)
        assert facts == "True 4 1YCR"
        assert list(elsewhere.iterdir()) == []

    def test_a_zipped_install_extracts_the_complex_for_the_length_of_the_process(
        self, built_layout, tmp_path
    ):
        archive = tmp_path / "site.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in sorted(built_layout.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(built_layout).as_posix())
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        lines = run_python(
            """
            import binding_metrics
            from binding_metrics.testing import bundled_complex_path, make_synthetic_poses

            path = bundled_complex_path()
            poses = make_synthetic_poses(seeds=(1,))
            print(binding_metrics.__file__)
            print(path)
            print(path.is_file(), len(poses.poses))
            """,
            pythonpath=archive,
            cwd=elsewhere,
        )
        package_file, path, facts = lines[-3:]
        assert package_file.startswith(str(archive))
        assert not Path(path).is_relative_to(archive)  # a real file, not a path inside the zip
        assert facts == "True 2"
        assert not Path(path).exists()  # extracted to a temporary file, removed at exit
