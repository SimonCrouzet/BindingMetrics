"""The insertion codes of the reference survive an array without an ins_code annotation (#194)."""

from pathlib import Path

from poses.test_frame import _load, consecutive

from binding_metrics.poses import Reference
from binding_metrics.poses.frame import renumber_to_reference, superpose_receptor

SFTI = Path(__file__).resolve().parent.parent / "data" / "example_bicyclic_sfti1_3P8F.cif"


def _renumbered(drop_annotation: bool):
    reference = Reference(SFTI, "A", "I")
    pose = consecutive(consecutive(_load(SFTI), "A"), "I")
    if drop_annotation:
        pose.del_annotation("ins_code")
    return renumber_to_reference(pose, superpose_receptor(pose, reference), reference)


def test_the_codes_of_the_reference_are_written_to_an_array_that_had_none():
    with_codes = _renumbered(drop_annotation=False)
    without = _renumbered(drop_annotation=True)

    assert set(with_codes.ins_code) - {""}  # the reference has insertion codes on chain A
    assert (without.ins_code == with_codes.ins_code).all()
    assert (without.res_id == with_codes.res_id).all()
