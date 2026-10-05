"""PredictionRecord, the deprecated name of PredictedPose."""

import importlib
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from binding_metrics.predictors.record import PredictedPose

REPO = Path(__file__).resolve().parents[2]

MESSAGE = (
    "PredictionRecord is deprecated and renamed PredictedPose; the old name will be removed "
    "in a future release (not before 0.4.0)"
)

#: Every module that imported or defined the old name before the rename.
MODULES = [
    "binding_metrics.predictors",
    "binding_metrics.predictors.record",
    "binding_metrics.predictors.base",
    "binding_metrics.predictors.af2",
    "binding_metrics.predictors.boltz2",
    "binding_metrics.predictors.of3",
    "binding_metrics.predictors.protenix",
    "binding_metrics.metrics.evobind",
    "binding_metrics.metrics.openfold",
    "binding_metrics.metrics.prediction",
]


@pytest.mark.parametrize("module_name", MODULES)
class TestTheOldNameStillImports:
    def test_attribute_access_warns_with_the_exact_message_and_gives_the_new_class(
        self, module_name
    ):
        module = importlib.import_module(module_name)
        with pytest.warns(DeprecationWarning) as caught:
            old = module.PredictionRecord
        assert old is PredictedPose
        assert [str(w.message) for w in caught] == [MESSAGE]
        assert caught[0].category is DeprecationWarning

    def test_the_warning_points_at_the_line_that_used_the_name(self, module_name):
        module = importlib.import_module(module_name)
        with pytest.warns(DeprecationWarning) as caught:
            module.PredictionRecord  # noqa: B018
        assert Path(caught[0].filename).resolve() == Path(__file__).resolve()

    def test_a_from_import_statement_works_and_points_at_the_importer(self, module_name):
        namespace: dict = {}
        code = compile(f"from {module_name} import PredictionRecord", "<alias-test>", "exec")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(code, namespace)  # noqa: S102 - a fixed import statement
        assert namespace["PredictionRecord"] is PredictedPose
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert [str(w.message) for w in deprecations] == [MESSAGE]
        assert deprecations[0].filename == "<alias-test>"

    def test_every_use_warns_because_nothing_is_cached(self, module_name):
        module = importlib.import_module(module_name)
        for _ in range(2):
            with pytest.warns(DeprecationWarning, match=re.escape(MESSAGE)):
                module.PredictionRecord  # noqa: B018
        assert "PredictionRecord" not in vars(module)

    def test_other_missing_names_are_still_attribute_errors(self, module_name):
        module = importlib.import_module(module_name)
        with pytest.raises(AttributeError, match="no_such_name"):
            module.no_such_name  # noqa: B018
        assert not hasattr(module, "PredictionRecordX")

    def test_the_old_name_is_neither_listed_nor_star_imported(self, module_name):
        module = importlib.import_module(module_name)
        assert "PredictionRecord" not in getattr(module, "__all__", ())
        assert "PredictionRecord" not in dir(module)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exec(f"from {module_name} import *", {})  # noqa: S102 - a fixed star import


class TestTheOldNameIsTheSameClass:
    def test_isinstance_and_issubclass_work_both_ways(self):
        with pytest.warns(DeprecationWarning):
            from binding_metrics.predictors import PredictionRecord

        record = PredictedPose("of3", "q")
        assert isinstance(record, PredictionRecord)
        assert issubclass(PredictedPose, PredictionRecord)
        assert PredictionRecord("of3", "q").model == "of3"
        assert type(PredictionRecord("of3", "q")) is PredictedPose

    def test_a_subclass_of_the_old_name_is_a_subclass_of_the_new_one(self):
        with pytest.warns(DeprecationWarning):
            from binding_metrics.predictors.record import PredictionRecord

        class Mine(PredictionRecord):
            pass

        assert issubclass(Mine, PredictedPose)

    def test_the_new_name_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            import binding_metrics.predictors as package
            import binding_metrics.predictors.record as record_module

            assert package.PredictedPose is record_module.PredictedPose is PredictedPose
            exec("from binding_metrics.predictors import PredictedPose", {})  # noqa: S102


class TestOurOwnCodeUsesTheNewName:
    """A fresh interpreter imports and uses the predictors package with the warning an error."""

    def test_a_representative_import_and_use_raises_no_deprecation_warning(self, tmp_path):
        script = textwrap.dedent(
            f"""
            import warnings
            # third-party libraries may warn on their own import; load them first
            import numpy, biotite.structure, biotite.structure.io.pdbx, biotite.structure.io.pdb
            from tests.predictors import synth, synth_of3

            warnings.simplefilter("error", DeprecationWarning)

            import importlib
            import binding_metrics.predictors as predictors
            for name in predictors.__all__:
                getattr(predictors, name)
            for module in (
                "record", "base", "registry", "session", "store", "af2", "boltz2", "of3",
                "protenix", "of3_runner", "boltz2_runner", "protenix_runner", "af2_runner",
            ):
                importlib.import_module("binding_metrics.predictors." + module)
            for module in ("evobind", "openfold", "prediction"):
                importlib.import_module("binding_metrics.metrics." + module)
            from binding_metrics.predictors import *  # noqa: F403
            from binding_metrics.poses import *  # noqa: F403

            from pathlib import Path
            from binding_metrics.predictors import get_parser
            from binding_metrics.metrics.prediction import summarize_prediction

            work = Path({str(tmp_path)!r})
            synth_of3.write_prediction(work, "q", synth.synthetic_complex())
            record = get_parser("of3").load(work, "q")
            record.validate()
            assert isinstance(record, PredictedPose)
            assert record.kind == "predicted" and record.native_scores()
            summarize_prediction(record)
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("ok")
