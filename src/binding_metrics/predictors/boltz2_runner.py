"""Boltz-2 as a prediction runner for the prediction store.

``Boltz2Runner`` builds the Boltz-2 YAML input and the template CIFs from a complex structure,
starts ``boltz predict`` in the current environment or in a conda environment, and leaves the
output where ``binding_metrics.predictors.boltz2.Boltz2Parser`` reads it. It is the Boltz-2
counterpart of ``OpenFold3Runner``: the same request, the same store layout, the same failure
reporting.

Public names and signatures (the module imports the standard library, the runner ABC and the
request; gemmi is imported when an input is built, the OpenFold3 residue reader when a chain
is read)::

    class Boltz2Runner(PredictionRunner):
        Boltz2Runner(conda_env: Optional[str] = None)
        name = "boltz2"; capabilities = None
        supports_custom_weights = True; weights_kind = "file"
        run_modes = RUN_MODES      # the modes it builds an input for, ``predict`` included
        .conda_env
        .make_request(input_path, *, name: str, binder_chain: Optional[str] = None,
            receptor_chain: Optional[str] = None, mode: str = "score",
            seeds: Optional[Sequence[int]] = None, num_samples: int = 5,
            use_msa_server: bool = True, on_unmappable_residue: str = "error",
            binder_cyclic: bool | str = "auto",
            lock_threshold_angstrom: Optional[float] = None,
            weights: Optional[str | Path | WeightsRef] = None,
            extra_args: Sequence[str] = (),
            closure_bonds: str | Sequence[Mapping] = "none",
            write_embeddings: bool = False,
            msa_policy: str = "default",
            receptor_msa: Optional[str | Path] = None) -> PredictionRequest
        .prepare(request, work_dir) -> Path          # <work_dir>/input/<name>.yaml
        .run(request, work_dir) -> Path   # <work_dir>/boltz_results_<name>/predictions/<name>
        .is_available() -> bool
        .version() -> Optional[str]

    class Boltz2RunError(subprocess.CalledProcessError)
    RUN_MODES                                        # the five modes
    DEFAULT_LOCK_THRESHOLD_ANGSTROM = 2.0            # both lock modes, this package's choice

Modes. ``input_path`` is the complex structure (PDB or mmCIF); ``binder_chain`` and
``receptor_chain`` are required in every mode. The YAML has one protein entity per chain, with
the chain IDs of the input, the sequence and the modified residues that
``metrics._openfold_run._extract_query_chain`` reads (the same chain reader as the OpenFold3
runner), and these templates:

* ``predict``: no ``templates`` block; the model sees the sequences only.
* ``refold``: the receptor is templated from ``receptor.cif`` (the receptor chain of the input);
  the binder is free.
* ``score``: each chain is templated from its own single-chain CIF (``receptor.cif``,
  ``binder.cif``), unforced. A template row gives the fold of its chain only; the model places
  the chains against each other itself.
* ``score-lock``: one ``lock.cif`` holds both chains in the frame of the input and is listed for
  both chains with ``force: true`` and a ``threshold``. Boltz-2 keeps both chains in one
  template row and the forced template pulls the prediction back to within the threshold of the
  input after a rigid alignment over the templated residues. It is a guidance term of weight
  0.1, not a hard constraint, so the pose can still move.
* ``refold-lock``: ``receptor.cif`` holds the receptor chain alone and is listed for it with
  ``force: true`` and a ``threshold``; the binder has no template and is free. It is ``refold``
  with the receptor pinned to the input by the same guidance term as ``score-lock``, so the
  receptor can still move by about the threshold and the binder is placed by the model.

Any other mode raises ``ValueError``.

The threshold. Boltz-2 requires ``threshold`` when ``force`` is true and documents no default,
so ``lock_threshold_angstrom`` defaults to 2.0 here: that is this package's choice, not
Boltz-2's. It is an option of the request (``options["lock_threshold_angstrom"]``, None outside
the lock modes ``score-lock`` and ``refold-lock``) and so part of the key. It is the upper bound,
in angstrom, on the distance of a residue's representative atom (CB, CA for glycine) from the
aligned template before the guidance term acts; a smaller value pins the pose harder.

Residues. Standard residues and their protonation variants are plain letters; D-amino acids,
N-methylated and other peptide-linking Chemical Component Dictionary residues keep their
chemistry through ``modifications`` (position and CCD code), with the letter of the parent
residue; selenocysteine is sent as ``C`` with ``SEC``, because Boltz-2 reads the letter ``U`` as
an unknown residue. A residue that cannot be expressed raises ``ValueError`` naming the chain and
the residues before any file is written, unless ``on_unmappable_residue="x"`` sends an unknown
residue (``X``) in its place and logs a warning. Waters, ions, ligands and terminal caps are not
sent. Chains with one sequence become one Boltz-2 entity, which keeps the modifications and the
cyclic flag of the first of them, so such chains that differ in either raise ``ValueError``.

Template CIFs. Boltz-2 aligns the template sequence to the query sequence letter by letter, and
reads a modified residue of a template as ``X``, so the chain ``ALLVTAGLVLA`` of cyclosporin
would match its own template over a few residues only. The template CIF therefore names every
residue by its parent (``MLE`` becomes ``LEU``, ``DAL`` becomes ``ALA``, an unknown parent
becomes ``UNK``), as the OpenFold3 template carries the canonical sequence in ``_entity_poly``.
The coordinates are those of the input; hydrogens are dropped. The chain names of a template
are its ``label_asym_id`` values (``Subchain.subchain_id`` in Boltz-2's reader), which are set to
the chain IDs of the input.

Stored input. The YAML lists each template CIF by an absolute path below the ``work_dir`` of the
run. ``PredictionStore`` runs in a temporary directory (``<key>.tmp-<pid>-<random>``) and renames
it when it keeps the run, so the YAML of a stored entry (``outputs/input/<name>.yaml``) names the
temporary path, which no longer exists, while the CIFs are in the entry
(``outputs/input/templates/``). That is by design. The path is not part of the key, and a relative
path would not be simpler: Boltz-2 opens ``cif`` as written (``schema.py:1611-1672``, relative to
its working directory, not to the YAML). A stored YAML records what was run; it cannot be given to
``boltz predict`` as it stands, and ``prepare`` writes a fresh one.

Cyclic binder. ``binder_cyclic`` is ``"auto"``, True or False, with the meaning it has for
OpenFold3: True writes ``cyclic: true`` on the binder entity, False never does, ``"auto"``
writes it when the binder has a head-to-tail bond (``capabilities.detect_closures``). Boltz-2
turns the flag into a cyclic period of the chain length for the relative positions; it does not
enforce the closure bond. ``closure_bonds`` (below) writes the bonds themselves.

Closure bonds. ``closure_bonds`` is ``"none"`` (default), ``"auto"`` or a list of bonds, and is
written as the ``constraints`` of the YAML::

    constraints:
      - bond:
          atom1: ["I", 3, "SG"]       # chain, 1-based position in the sequence sent, atom
          atom2: ["I", 11, "SG"]

``"auto"`` writes one ``bond`` for each closure that ``capabilities.detect_closures`` finds on
the binder of the input, of the families ``head_to_tail`` (C of the last residue to N of the
first), ``disulfide`` (SG to SG) and ``lactam`` (the amide of a side chain or of a terminus).
A staple or another cross-link is not written and a warning names it: Boltz-2 documents the
constraint for CCD ligands and canonical residues only. The position of a closure end is found
by ``predictors/_closures.py``, the mapping that the Protenix runner uses, so a modified residue
in the chain does not shift it, and a closure that cannot be placed raises ``ValueError`` before
any file is written. A list is written as given, each item ``{"atom1": [chain, position, atom],
"atom2": [...]}``; a shape error or a chain or position the input lacks raises ``ValueError``.
A bond is an input feature of the trunk (``token_bonds``, ``boltz2.py:426``). Boltz-2 reads a
bond between the first and the last residue of a chain as a cyclic period too, but it compares
the residue indices of the whole structure with 0 (``featurizerv2.py:1027-1050``), so only for the
first chain of the YAML; the receptor comes first here, and the period of the binder comes from
``cyclic: true``. ``ConnectionsPotential`` skips a bond inside one chain (``featurizerv2.py:2023``),
so the source shows nothing that holds the distance of a disulfide or a lactam. Whether the
feature moves the prediction is a measurement and not a promise. ``"none"`` keeps the YAML and
the key as they were.
``options["closure_bonds"]`` holds the list that is written (an ``"auto"`` request is resolved
when the request is made) and is absent when that list is empty, so two requests that write the
same bonds share a key and a request without the option keeps its key.

Embeddings. ``write_embeddings=True`` passes ``--write_embeddings``: Boltz-2 then writes
``embeddings_<name>.npz`` (arrays ``s`` and ``z``, the single and pair representations of the
trunk) next to the structures, once per input and not per sample (``writer.py:249-257``).
``Boltz2Parser`` registers the file and puts its path in ``extras["embeddings_path"]``; nothing
here loads the arrays. The flag is in ``options["write_embeddings"]`` only when True, so a
request without it keeps its key, and a stored entry made without it is not a hit for a request
that asks for the embeddings.

MSA. ``use_msa_server`` (default True) passes ``--use_msa_server``: the sequences are sent to the
ColabFold server. False writes ``msa: empty`` on each entity, the single-sequence mode of the
documentation, which Boltz-2 itself calls suboptimal.

Receptor MSA and the policy that forbids the server. ``receptor_msa`` is the path of the
receptor's alignment (``.a3m`` or ``.csv``, the two formats Boltz-2 reads; the first sequence is
the query and must be the receptor sequence sent) or the word ``"single"``; ``msa_policy`` is
``"default"`` or ``"forbid-server"`` (``binding_metrics.predictors._msa`` holds the rules, shared
with the other runners). A file is copied to ``<work_dir>/input/msa/receptor.<ext>`` and the YAML
gives that copy to the receptor entity (``msa: "<path>"``). The binder entity is written
``msa: "empty"`` whenever the receptor has a file or ``"single"``: Boltz-2 refuses a custom MSA
next to an automatic one in one input (``schema.py:1311``), and an alignment of 4 to 12 residues
is of no use. Boltz-2 reads chains with one sequence as one entity with one MSA
(``schema.py:1118``), so a receptor and a binder with the same sequence are refused when a file is
given. Under ``forbid-server`` ``use_msa_server`` must be False, the receptor needs a file or
``"single"``, and the command never carries ``--use_msa_server``: ``prepare`` and ``run`` check
the request again (a request built by hand) and refuse the command line they built if it holds a
server flag, before a file is written or a process starts. Boltz-2 raises "Missing MSA's in
input and --use_msa_server flag not set" for an entity without an ``msa`` when the flag is
absent (``main.py:581``), so no entity can reach the server without it. The key:
``options["msa_policy"]`` only when it is ``"forbid-server"``, ``options["receptor_msa"] =
"single"`` only when asked, ``extra_files["receptor_msa"]`` (hashed by content, not by path) only
when a file is given; a request without these keeps its key. With a file and
``use_msa_server=True`` (default policy) the flag is on the command but moot: every entity has an
``msa``, so nothing is generated.

Seeds and samples. ``boltz predict`` takes one ``--seed`` and writes no seed dimension, which
the adapter mirrors (``seed_index`` must be 1), so a request has exactly one seed (default 42)
and another seed is another request. ``num_samples`` (default 5, as for OpenFold3) is
``--diffusion_samples``; the files are ranked by confidence, so sample 1 of the adapter is the
best one. Every run is given ``--seed``, ``--diffusion_samples``, ``--model boltz2``,
``--output_format mmcif``, ``--write_full_pae`` and ``--write_full_pde`` explicitly.

Custom weights. ``weights`` (a checkpoint file, or the ``WeightsRef`` that
``PredictionStore.weights_reference`` made for it) is ``PredictionRequest.weights``: the key holds
the SHA-256 and size of the file, never its path, and the run passes it as ``--checkpoint``. The
runner sets ``supports_custom_weights = True`` and ``weights_kind = "file"``; a directory is
refused. Without weights Boltz-2 loads ``boltz2_conf.ckpt`` of its cache, and the key is what it
would be without the field.

Environment. Boltz-2's cache is ``~/.boltz`` or ``$BOLTZ_CACHE``; it holds the weights and the
Chemical Component Dictionary and is downloaded on first use (network). Boltz-2 prints some of
its failures and exits with status 0 (see below), and takes the model's messages on stdout and
stderr: both are echoed to stderr and the last 8000 characters are kept for the reason.

Version. ``version()`` and ``is_available()`` describe the same installation, the ``boltz`` that
``run`` starts. Without ``conda_env`` the version is read by the interpreter named on the first
line of the ``boltz`` script that ``PATH`` finds (``pip``, ``pipx`` and ``conda`` write that
line), so a ``boltz`` in an environment other than the one that runs this package has its
version in the request key. The version is None, and the key holds an empty version, when
``boltz`` is not on PATH, when the script is not Python (a shell wrapper names no interpreter)
or when the interpreter has no ``boltz`` distribution (the package metadata is read, as for
OpenFold3; the package is not imported). With ``conda_env`` the interpreter is
``conda run -n ENV python``.

Failures. A non-zero exit raises ``Boltz2RunError`` (a ``CalledProcessError`` whose text starts
with the reason, advice for the failures that have a known fix, the last lines of output and the
command). ``boltz predict`` catches an exception in the preparation of an input (a YAML or a
template it cannot read) and in a batch that runs out of memory, prints a message and exits with
status 0 without writing a prediction; ``run`` raises ``RuntimeError`` with that message then,
so that an empty result is never stored as done.

What rests on reading the source. Boltz-2 v2.2.1 (tag ``v2.2.1``, ``pyproject.toml`` 2.2.1;
https://github.com/jwohlwend/boltz) was read on 2026-10-01: ``docs/prediction.md`` (YAML
schema, ``msa: empty``, ``modifications``, ``templates`` with ``chain_id``, ``template_id``,
``force`` and ``threshold``), ``src/boltz/main.py`` (every flag above and the layout
``<out_dir>/boltz_results_<input stem>/predictions/<input stem>/``; ``process_input`` catches
every exception of the preparation, ``main.py:657``), ``src/boltz/data/parse/schema.py`` (entities
are grouped by type and sequence, ``schema.py:1037``; modifications replace the residue by its
CCD code, ``schema.py:1169``; ``threshold`` is required with ``force``, ``schema.py:1698``; the
letter ``U`` is an unknown residue, ``data/const.py:166``), ``src/boltz/data/parse/mmcif.py``
(template chain names, ``mmcif.py:886``), ``src/boltz/data/feature/featurizerv2.py`` (a template
row per template file, ``featurizerv2.py:1784``), ``src/boltz/model/potentials/potentials.py``
(``TemplateReferencePotential`` is on by default through ``contact_guidance_update``, which
``BoltzSteeringParams`` sets to True at ``main.py:156`` and ``use_potentials`` does not touch:
``potentials.py:755-786``), ``src/boltz/model/models/boltz2.py`` (a batch that runs out of memory
is skipped with a printed warning, ``boltz2.py:1125``, and the writer then writes nothing,
``writer.py:60``) and ``src/boltz/data/write/writer.py`` (the files).
``--write_full_pae`` and ``--write_full_pde`` are read by ``boltz1.py`` only: ``boltz2.py`` adds
the PDE to every prediction and the PAE when the PAE head is on, whatever the flags
(``boltz2.py:1083-1101``), so the flags are passed for the documentation's sake and a later
version that reads them.

What was run. On 2026-10-01 the runner was run against the real model once, on one machine: the
v2.2.1 source tree on the interpreter of a conda environment that has Boltz-2's dependencies, the
``boltz2_conf.ckpt`` of its cache, an 8 GB GPU, ``use_msa_server=False`` (``msa: empty``),
``extra_args=["--no_kernels"]`` and two samples, through ``PredictionStore``, ``PredictionSession``
and ``Boltz2Parser``. Modes ``predict``, ``refold``, ``score`` and ``score-lock`` ran on 1YCR, and
``score-lock`` on 1CWA (a cyclic binder with nine modified residues). Boltz-2 accepted every YAML
and template, wrote the layout above, and the parser read it without a reason: 98 tokens for 1YCR
and 176 for 1CWA (a modified residue is one token), PAE and PDE of that size, the structure files
with ``auth_asym_id`` and ``auth_seq_id``. A first run failed in Boltz-2's cuequivariance kernels
(an import error of the installed ``cuequivariance_ops_torch``): its text and the advice about
``--no_kernels`` are what ``Boltz2RunError`` produced. The YAML and the templates of every mode
were also read by Boltz-2's own ``parse_yaml`` (CPU) for 1YCR and 1CWA.

What was not run on 2026-10-01: ``use_msa_server=True`` (network), a conda environment,
``weights``, any version other than 2.2.1, and a run that exits with status 0 without output (the
swallowed preparation error and the batch skipped for memory rest on the source and on the stub of
the tests).

What was measured about the modes. A second set of runs moved the binder of 1YCR as a rigid body
(receptor fixed; 3, 6 and 10 A away from the pocket, and a 30 degree tilt or roll with 4 A of
translation) and ran each pose as the input, with seeds 42 to 44, one diffusion sample and
single-sequence mode. ``score`` returned the same structure for every pose of the binder
(identical coordinates for five decoys). ``score-lock`` kept the prediction within about the
threshold of the supplied pose and not on it: the representative atom of every templated residue
ended within 2.06 A of the input after a rigid alignment in 18 of 18 runs (threshold 2.0 A), so
the binder C-alpha RMSD to the input pose was 1.8 to 2.3 A (means over the seeds) for decoys 3 to
10 A out. A lower threshold held the pose tighter (one seed each), and the same two-chain
template without ``force`` did not carry the pose (one seed, three decoys). This is one complex
that Boltz-2 very likely saw in training and three seeds: nothing follows for a design, another
complex or another version. The table is in ``docs/metrics.md``, "Observed behaviour".

What was measured about the bonds and the embeddings. On 2026-10-13, Boltz-2 v2.2.1 on one
machine, mode ``predict`` (sequences only), single sequence, ``--no_kernels``, one diffusion sample,
seeds 42 to 44, through the store and the adapter. Boltz-2's own ``parse_yaml`` (CPU) accepted the
block for 3P8F chain I, 1CWA chain C and 1XY4 chain A and put each bond on the intended residues
and atoms. Distance of the closure atoms in the prediction, without the bond constraint -> with
``closure_bonds="auto"`` (mean of three seeds, angstrom; the value in the input structure in
brackets):

* 3P8F chain I with ``cyclic: true`` (what ``binder_cyclic="auto"`` writes for a ring): SG-SG
  2.03 -> 1.90 (2.04), head-to-tail C-N 1.33 -> 1.33 (1.44). 1CWA chain C: C-N 1.30 -> 1.30
  (1.33). Both closed without the bond already.
* Without ``cyclic: true`` (``binder_cyclic=False``), so that the bond is the only closure: C-N
  6.68 -> 1.77 on 3P8F (6.3 to 7.0 without), 7.78 -> 1.40 on 1CWA (6.0 to 9.3 without).
* 1XY4 (disulfide and lactam; a chain of 1YCR added as receptor, no template): lactam NZ-CD 5.09
  (3.5 to 6.2) -> 2.15 (1.9 to 2.4; the input has 1.89, an amide bond is 1.33), SG-SG 2.03 -> 1.81.
* Sequences the model cannot have seen, the same atoms in a made-up input: SFTI-1 with nine
  residues replaced (disulfide and ring) without ``cyclic: true``: SG-SG 13.4 -> 1.75, C-N 20.7 ->
  1.37, binder pLDDT 82 -> 57; a p53 peptide with two cysteines (explicit bond, no ring):
  SG-SG 18.1 -> 1.84, ipTM 0.92 -> 0.78 (0.57 to 0.92), binder pLDDT 90 -> 61. One seed gave
  SG-SG 1.46, shorter than a disulfide.

So the constraint moved a closure that the model did not make by itself in every run (12 of 12
for the disulfide and the ring, 3 of 3 for the lactam, which came within 1.9 to 2.4 and not to
the bond length) and changed little where it already did (a 0.1 to 0.2 shorter SG-SG); it can
over-close, and the confidence of the prediction falls with it. Three complexes that the model has
very probably seen, two made-up sequences, three seeds each: no accuracy of the model is claimed.
With ``write_embeddings=True`` on 1YCR (98 tokens) the run wrote ``embeddings_<name>.npz`` with
``s`` of shape (1, 98, 384) and ``z`` of shape (1, 98, 98, 128), float32, 4.6 MB compressed;
the run without the flag wrote no such file, and both runs gave the same ipTM (0.943).

What was measured about ``refold-lock`` and the given MSA. On 2026-10-13, Boltz-2 v2.2.1 on one
machine, one diffusion sample, seeds 42 to 44, ``--no_kernels``, through the store and the adapter.
Boltz-2's own ``parse_yaml`` (CPU) read the YAML of ``refold-lock`` as one template record on the
receptor chain, forced, with the threshold, and none on the binder. On 1YCR in single-sequence
mode (receptor Cα RMSD to the input; largest deviation of a templated residue's representative atom
after one rigid alignment, Boltz-2's own guided quantity; binder Cα RMSD to the crystal peptide
after superposing the receptor; ipTM; mean over three seeds, angstrom): unforced ``refold`` 0.48,
3.90, 1.81, 0.952; ``refold-lock`` at 0.5 A 0.39, 0.49, 1.80, 0.954; at 1.0 A 0.44, 0.99, 1.85,
0.953; at 2.0 A 0.46, 1.98, 1.86, 0.951. The largest deviation ended at the threshold in 9 of 9
runs (one residue, the C-terminal one, moves 3.9 A unforced); the typical residue moves little in
all cells (mean deviation 0.53 unforced, 0.33 to 0.50 locked); the binder, which has no template,
is where the unforced run puts it (1.8 A from the crystal peptide, 2.0 A for ``predict``) and ipTM
does not change beyond the seed spread. The term is flat inside the threshold: on 1CWA (a ring with
nine modified residues) at 2.0 A the output was identical to the unforced run, because the
receptor was within 0.90 A of the input, and at 0.5 A the largest deviation went from 0.90 to 0.49 A
with nothing else changed. The threshold stays 2.0 A: no cost showed at 0.5 or 1.0 A either.
With a receptor alignment of the public MDM2 sequence (1633 sequences, from the ColabFold server
once) under ``msa_policy="forbid-server"`` in a network namespace without an interface, three
seeds ran and no log mentions a server; a control with the server on failed in the server call.
ipTM was 0.969 (0.965 to 0.972) with the given alignment, 0.969 (0.964 to 0.974) with the server
for both entities, and 0.948 (0.943 to 0.954) in single-sequence mode. One public complex that the
model very probably saw in training: nothing here says how the lock behaves for a designed receptor
that the model does not already reproduce.

Boltz-2 licence: MIT; no weights or model code are read or shipped here.
"""

from __future__ import annotations

import codecs
import csv
import dataclasses
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from binding_metrics.capabilities import LOCK_MODES
from binding_metrics.predictors._closures import find_closures, place_closures
from binding_metrics.predictors._msa import (
    DEFAULT_MSA_POLICY,
    FORBID_SERVER,
    RECEPTOR_MSA_ROLE,
    SINGLE_SEQUENCE,
    assert_no_server_flag,
    check_msa_of_request,
    check_msa_request,
)
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import PredictionRequest
from binding_metrics.predictors.weights import WeightsRef

if TYPE_CHECKING:
    import gemmi

logger = logging.getLogger(__name__)

#: The modes this runner builds an input for (the vocabulary of ``store.MODES``).
RUN_MODES: tuple[str, ...] = ("predict", "refold", "score", "score-lock", "refold-lock")

_DEFAULT_SEED = 42
_DEFAULT_NUM_SAMPLES = 5
_DEFAULT_USE_MSA_SERVER = True
_DEFAULT_ON_UNMAPPABLE = "error"
_DEFAULT_BINDER_CYCLIC = "auto"
_DEFAULT_CLOSURE_BONDS = "none"
_DEFAULT_WRITE_EMBEDDINGS = False

#: The alignment formats Boltz-2 reads (``main.py:640-656`` compares the suffix as written).
_MSA_SUFFIXES = (".a3m", ".csv")

#: Letters of an alignment's query that stand for "any residue": they match whatever was sent.
_MSA_WILDCARDS = frozenset("XU")

#: The closure families written as ``bond`` constraints (``closure_bonds="auto"``).
_CLOSURE_BOND_FAMILIES = ("head_to_tail", "disulfide", "lactam")

#: Threshold of the forced template in the lock modes, in angstrom. Boltz-2 gives no default (it
#: raises when ``force`` has none), so this is the choice of this package.
DEFAULT_LOCK_THRESHOLD_ANGSTROM = 2.0

_ON_UNMAPPABLE_CHOICES = ("error", "x")

#: The flags the runner sets itself. A caller's ``extra_args`` may not repeat one: the last
#: occurrence would win on the command line and the request key would describe another run.
_OWNED_FLAGS = (
    "--out_dir",
    "--model",
    "--diffusion_samples",
    "--seed",
    "--output_format",
    "--write_full_pae",
    "--write_full_pde",
    "--use_msa_server",
    "--checkpoint",
    "--write_embeddings",
)

#: The three-letter name a template CIF gives each query letter (the parent residue).
_LETTER_TO_NAME: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}  # fmt: skip

#: How much of Boltz-2's output is kept for the reason of a failure (characters), how many lines
#: of it the error text shows, and the longest line shown.
_OUTPUT_TAIL_CHARS = 8000
_OUTPUT_LINES_IN_MESSAGE = 8
_REASON_CHARS = 300

#: ``(needles, advice)``: advice for the failures of Boltz-2 that have a known fix. A needle is
#: looked up in the lower-cased output.
_KNOWN_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("out of memory", "outofmemoryerror"),
        "The GPU ran out of memory: lower num_samples or use a GPU with more memory. Boltz-2 "
        "skips a batch that does not fit and still exits normally, so such a run ends "
        "without output.",
    ),
    (
        ("cuequivariance",),
        "The cuequivariance kernels failed (an old GPU, or a broken install): pass "
        "extra_args=['--no_kernels'] (docs/prediction.md of Boltz-2, Troubleshooting).",
    ),
    (
        ("no supported gpu backend",),
        "No GPU is visible to Boltz-2, which runs on the GPU by default.",
    ),
    (
        ("no module named 'boltz'",),
        "The boltz command starts but its package cannot be imported (a source tree that was "
        "moved behind an editable install, for example). Reinstall boltz in that environment.",
    ),
    (
        ("max retries exceeded", "connectionerror", "name or service not known"),
        "A network request failed. use_msa_server=True sends the sequences to the ColabFold "
        "server, and Boltz-2 downloads its weights and the Chemical Component Dictionary to "
        "its cache on first use; check the connection, or pass use_msa_server=False for the "
        "single-sequence mode.",
    ),
)

#: Probe that asks an interpreter for the version of the ``boltz`` distribution (its metadata, as
#: the OpenFold3 probe does; importing the package is not needed to read it).
_VERSION_PROBE = "from importlib.metadata import version; print(version('boltz'))"

#: Bytes of the first line of the ``boltz`` script that are read to find its interpreter.
_SHEBANG_BYTES = 4096


# ---------------------------------------------------------------------------- failures


def _output_lines(text: str) -> list[str]:
    """Non-empty lines of ``text``; a progress bar's carriage returns keep only the last state."""
    lines = []
    for raw in text.split("\n"):
        line = raw.split("\r")[-1].strip()
        if line:
            lines.append(line)
    return lines


def _key_line(lines: Sequence[str]) -> str:
    """The line that names the failure: the last exception line, else the last line."""
    for line in reversed(lines):
        if re.match(r"^[\w.]*(Error|Exception|Exit|Interrupt)\b", line):
            return line
    return lines[-1] if lines else ""


def _hint(text: str) -> str:
    """The advice for a known failure message, or an empty string."""
    lowered = text.lower()
    for needles, advice in _KNOWN_FAILURE_HINTS:
        if any(needle in lowered for needle in needles):
            return advice
    return ""


class Boltz2RunError(subprocess.CalledProcessError):
    """``boltz predict`` exited with a non-zero status.

    A ``subprocess.CalledProcessError`` whose message starts with the reason: the line of the
    output that names the failure, advice for the failures that have a known fix, and the last
    lines of output. ``stderr`` holds the last few kilobytes of the output (Boltz-2's stdout and
    stderr together) and ``hint`` the advice.
    """

    def __init__(self, returncode: int, cmd, output_tail: str = ""):
        super().__init__(returncode, cmd, stderr=output_tail)
        self.hint = _hint(output_tail)

    def __str__(self) -> str:
        lines = _output_lines(self.stderr or "")
        head = f"Boltz-2 exited with status {self.returncode}"
        if lines:
            head += f": {_key_line(lines)[:_REASON_CHARS]}"
        parts = [head]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if lines:
            shown = "\n".join(
                f"  {line[:_REASON_CHARS]}" for line in lines[-_OUTPUT_LINES_IN_MESSAGE:]
            )
            parts.append(f"Last lines of output:\n{shown}")
        parts.append(f"Command: {self.cmd}")
        return "\n".join(parts)


def _run_boltz_command(cmd: Sequence[str]) -> str:
    """Run a ``boltz predict`` command line and return the tail of its output.

    Boltz-2 prints some failures on stdout and the progress and the tracebacks on stderr, so the
    two are read as one stream, echoed to ``sys.stderr`` as they arrive (stdout of the caller
    stays free for the caller's own results) and the last ``_OUTPUT_TAIL_CHARS`` characters are
    kept. The child runs unbuffered, so the order of the lines is that of the run.

    Raises:
        Boltz2RunError: The process exited non-zero.
    """
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    try:
        while chunk := process.stdout.read1(4096):
            text = decoder.decode(chunk)
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except (OSError, ValueError):  # a closed or unwritable console must not stop the run
                pass
            tail = (tail + text)[-_OUTPUT_TAIL_CHARS:]
        process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if process.returncode != 0:
        raise Boltz2RunError(process.returncode, list(cmd), tail)
    return tail


def _no_output_message(request: PredictionRequest, directory: Path, tail: str) -> str:
    """Why a run that exited with status 0 wrote nothing: the message Boltz-2 printed."""
    lines = _output_lines(tail)
    reason = next((line for line in reversed(lines) if "Failed to process" in line), "")
    if not reason:
        reason = next((line for line in reversed(lines) if "out of memory" in line.lower()), "")
    reason = reason or _key_line(lines)
    message = f"Boltz-2 exited normally but wrote no output for '{request.name}' in {directory}"
    if reason:
        message += f": {reason[:_REASON_CHARS]}"
    advice = _hint(reason)
    return f"{message}\nHint: {advice}" if advice else message


# ---------------------------------------------------------------------------- the version


def _script_interpreter(script: str) -> Optional[str]:
    """The Python interpreter that the script ``script`` starts with, or None.

    ``pip``, ``pipx`` and ``conda`` write ``boltz`` as a script whose first line names the
    interpreter of the environment that holds the package (``#!/path/to/env/bin/python3.12``, or
    ``#!/usr/bin/env python3``, which is looked up on ``PATH`` as the shell would). A script that
    is not Python (a shell wrapper, a binary) has no interpreter to ask, and the result is None.
    """
    try:
        with open(script, "rb") as handle:
            first = handle.readline(_SHEBANG_BYTES).decode("utf-8", errors="replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    words = first[2:].split()
    if words and Path(words[0]).name == "env":
        words = [word for word in words[1:] if not word.startswith("-") and "=" not in word]
        found = shutil.which(words[0]) if words else None
    else:
        found = words[0] if words else None
    if found is None or not Path(found).name.startswith("python"):
        return None
    return found


def _boltz_python_command(conda_env: Optional[str]) -> Optional[list[str]]:
    """The command that starts the interpreter ``boltz predict`` runs in, or None.

    With ``conda_env`` it is ``conda run -n ENV python``. Without it, it is the interpreter on the
    first line of the ``boltz`` script that ``shutil.which`` finds, which is the installation that
    ``is_available()`` reports and ``run`` starts; the interpreter that runs this package is not
    asked, because ``boltz`` is usually installed in an environment of its own.
    """
    if conda_env:
        return [shutil.which("conda") or "conda", "run", "-n", conda_env, "python"]
    script = shutil.which("boltz")
    interpreter = None if script is None else _script_interpreter(script)
    return None if interpreter is None else [interpreter]


def _installed_boltz_version(conda_env: Optional[str]) -> Optional[str]:
    """The version of the ``boltz`` that a run starts, or None when it cannot be read.

    The interpreter of that installation (``_boltz_python_command``) is started and asked for the
    metadata of the ``boltz`` distribution. None means that ``boltz`` is not on PATH, that its
    script does not name a Python interpreter (a shell wrapper), or that the interpreter cannot be
    started or has no ``boltz`` distribution.
    """
    command = _boltz_python_command(conda_env)
    if command is None:
        return None
    try:
        probe = subprocess.run(
            [*command, "-c", _VERSION_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("could not ask %s for the boltz version: %s", command, exc)
        return None
    if probe.returncode != 0:
        return None
    lines = _output_lines(probe.stdout)
    return lines[-1] if lines and lines[-1][0].isdigit() else None


# ---------------------------------------------------------------------------- the input


@dataclasses.dataclass(frozen=True)
class _Chain:
    """One protein entity of the YAML, and what its template CIF needs.

    ``modifications`` are ``(position, CCD code)`` with 1-based positions, in order.
    ``template_names`` is the parent residue name of each position, for the template CIF.
    """

    chain_id: str
    sequence: str
    modifications: tuple[tuple[int, str], ...]
    cyclic: bool
    template_names: tuple[str, ...]
    residue_indices: tuple[int, ...]  # index in the source chain of each position


@dataclasses.dataclass(frozen=True)
class _Template:
    """One ``templates`` entry: a CIF file and the chains it is listed for."""

    stem: str
    chain_ids: tuple[str, ...]
    force: bool = False
    threshold: Optional[float] = None


#: One ``bond`` constraint: two ``(chain ID, 1-based position, atom name)``.
_Bond = tuple[tuple[str, int, str], tuple[str, int, str]]


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Everything the input files hold, read from the structure before anything is written."""

    chains: tuple[_Chain, ...]  # receptor first, then binder
    templates: tuple[_Template, ...]
    structure: Any  # the gemmi.Structure the sequences and the templates come from
    bonds: tuple[_Bond, ...] = ()  # the ``bond`` constraints, in the order they are written
    #: How the entities get their MSA: ``"default"`` (the server, or ``empty`` without it),
    #: ``"given"`` (the receptor reads ``msa_file``, the binder is ``empty``) or ``"single"``
    #: (both ``empty``).
    msa: str = "default"
    msa_file: Optional[Path] = None  # the caller's file, for ``"given"``


def _quoted(text: str) -> str:
    """``text`` as a double-quoted YAML scalar (JSON strings are valid YAML ones).

    Always quoted, so that a chain named ``N`` or ``Y`` or ``1`` stays a string for the YAML
    loader instead of becoming a boolean or an integer.
    """
    import json

    return json.dumps(str(text))


def _yaml_float(value: float) -> str:
    """A float as a YAML number that a YAML 1.1 loader reads as a float (``2.0``, not ``2e-05``)."""
    text = f"{value:.10f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _yaml_text(
    plan: _Plan,
    template_paths: dict[str, Path],
    *,
    use_msa_server: bool,
    msa_path: Optional[Path] = None,
) -> str:
    """The Boltz-2 input YAML of ``plan`` (the schema is in ``docs/prediction.md`` of Boltz-2).

    ``msa_path`` is the staged alignment of the receptor, required when ``plan.msa`` is
    ``"given"``. The receptor is the first chain of the plan.
    """
    lines = ["version: 1", "sequences:"]
    for index, chain in enumerate(plan.chains):
        lines += [
            "  - protein:",
            f"      id: {_quoted(chain.chain_id)}",
            f"      sequence: {_quoted(chain.sequence)}",
        ]
        if plan.msa == "given" and index == 0:
            lines.append(f"      msa: {_quoted(str(msa_path))}")
        elif plan.msa != "default" or not use_msa_server:
            lines.append('      msa: "empty"')
        if chain.modifications:
            lines.append("      modifications:")
            for position, ccd in chain.modifications:
                lines += [f"        - position: {position}", f"          ccd: {_quoted(ccd)}"]
        if chain.cyclic:
            lines.append("      cyclic: true")
    if plan.bonds:
        lines.append("constraints:")
        for bond in plan.bonds:
            lines.append("  - bond:")
            for number, (chain_id, position, atom) in enumerate(bond, start=1):
                lines.append(
                    f"      atom{number}: [{_quoted(chain_id)}, {position}, {_quoted(atom)}]"
                )
    if plan.templates:
        lines.append("templates:")
        for template in plan.templates:
            ids = "[" + ", ".join(_quoted(chain_id) for chain_id in template.chain_ids) + "]"
            lines += [
                f"  - cif: {_quoted(str(template_paths[template.stem]))}",
                f"    chain_id: {ids}",
                f"    template_id: {ids}",
            ]
            if template.force:
                lines.append("    force: true")
                lines.append(f"    threshold: {_yaml_float(template.threshold)}")
    return "\n".join(lines) + "\n"


def _read_structure(path: Path) -> "gemmi.Structure":
    import gemmi

    try:
        structure = gemmi.read_structure(str(path))
    except (RuntimeError, ValueError, OSError) as exc:
        raise ValueError(f"cannot read {path} as a structure: {exc}") from exc
    if len(structure) == 0:
        raise ValueError(f"cannot read {path} as a structure: it has no model")
    return structure


def _unmappable_message(details: Sequence[tuple[str, str, list[str]]]) -> str:
    """The error text for residues that Boltz-2 cannot take (``UnmappableResidueError.details``)."""
    lines = "\n".join(
        f"  - chain '{chain}': {', '.join(labels)}" for _source, chain, labels in details
    )
    return (
        f"Boltz-2 cannot take these residues:\n{lines}\n"
        "Boltz-2 reads the 20 standard amino acids and an unknown residue (X), and any other "
        "amino acid only as a Chemical Component Dictionary code listed as peptide-linking "
        "(the `modifications` of the YAML); the names above are none of these, so the "
        "prediction would model something else than the input. Remove or replace the "
        "residues, or pass on_unmappable_residue='x' to send an unknown residue instead and "
        "log a warning."
    )


def _read_chain(
    structure: "gemmi.Structure",
    chain_id: str,
    *,
    on_unmappable_residue: str,
    cyclic: bool,
) -> _Chain:
    """The entity of ``chain_id``: sequence, modifications and the residues of its template.

    The chain is read by ``_extract_query_chain``, the reader of the OpenFold3 runner, so both
    models are given the same sequence. The residues that become sequence positions are then
    listed again, with the same rule, for the template CIF; the two counts must agree.

    Raises:
        ValueError: The chain is missing or has no amino acid, or it holds a residue Boltz-2
            cannot take (``UnmappableResidueError`` with Boltz-2's wording).
    """
    from binding_metrics.metrics._openfold_run import (
        UnmappableResidueError,
        _extract_query_chain,
        _residue_letter_and_ccd,
    )

    try:
        sequence, non_canonical = _extract_query_chain(
            structure, chain_id, on_unmappable_residue=on_unmappable_residue
        )
    except UnmappableResidueError as exc:
        raise ValueError(_unmappable_message(exc.details)) from exc

    modifications = dict(non_canonical)
    selenocysteine = [i for i, letter in enumerate(sequence, start=1) if letter == "U"]
    if selenocysteine:
        # Boltz-2 reads the letter U as an unknown residue (data/const.py:166)
        sequence = sequence.replace("U", "C")
        modifications.update({position: "SEC" for position in selenocysteine})

    residue_indices = []
    for chain in structure[0]:
        if chain.name != chain_id:
            continue
        for index, residue in enumerate(chain):
            if _residue_letter_and_ccd(residue.name) is not None or (
                on_unmappable_residue == "x" and {"N", "CA", "C"} <= {a.name for a in residue}
            ):
                residue_indices.append(index)
        break
    if len(residue_indices) != len(sequence):
        raise ValueError(
            f"chain '{chain_id}': the sequence has {len(sequence)} positions but "
            f"{len(residue_indices)} residues were selected for its template; this is a "
            "mismatch between the sequence reader and the template writer"
        )
    return _Chain(
        chain_id=chain_id,
        sequence=sequence,
        modifications=tuple(sorted(modifications.items())),
        cyclic=cyclic,
        template_names=tuple(_LETTER_TO_NAME.get(letter, "UNK") for letter in sequence),
        residue_indices=tuple(residue_indices),
    )


def _check_entities(chains: Sequence[_Chain]) -> None:
    """Chains with one sequence become one Boltz-2 entity: they must not differ otherwise.

    Boltz-2 groups the YAML items by type and sequence and takes the modifications and the
    ``cyclic`` flag of the first item for all of them (``schema.py:1037,1169,1171``), so a
    difference would be dropped without a message.
    """
    first: dict[str, _Chain] = {}
    for chain in chains:
        other = first.setdefault(chain.sequence, chain)
        if other is chain:
            continue
        for what, same in (
            ("modified residues", other.modifications == chain.modifications),
            ("cyclic flag", other.cyclic == chain.cyclic),
        ):
            if not same:
                raise ValueError(
                    f"chains '{other.chain_id}' and '{chain.chain_id}' have the same sequence "
                    f"but not the same {what}. Boltz-2 reads chains with one sequence as one "
                    f"entity and takes the {what} of the first, so the difference would be "
                    "lost."
                )


def _closure_bond_option(value: Any) -> Union[str, list[dict[str, list]]]:
    """``closure_bonds`` as ``"none"``, ``"auto"`` or a list of bonds in canonical form.

    A bond is ``{"atom1": [chain, position, atom], "atom2": [chain, position, atom]}``: the chain
    a non-empty string, the position an integer from 1 (the place in the sequence sent), the atom
    a non-empty string. The list is rebuilt, so the key does not depend on the caller's types.

    Raises:
        TypeError: A string, a mapping or a non-sequence where a list of bonds is expected.
        ValueError: An unknown string, or a bond that does not have the shape above.
    """
    if isinstance(value, str):
        if value not in ("none", "auto"):
            raise ValueError(
                f"closure_bonds must be 'none', 'auto' or a list of bonds, got {value!r}"
            )
        return value
    if isinstance(value, (bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(
            f"closure_bonds must be 'none', 'auto' or a list of bonds, got {type(value).__name__}"
        )
    bonds = []
    for number, bond in enumerate(value, start=1):
        shape = 'each bond is {"atom1": [chain, position, atom], "atom2": [chain, position, atom]}'
        if not isinstance(bond, Mapping) or set(bond) != {"atom1", "atom2"}:
            raise ValueError(f"closure_bonds item {number} is {bond!r}; {shape}")
        entry = {}
        for key in ("atom1", "atom2"):
            end = bond[key]
            if isinstance(end, (str, bytes, Mapping)) or not isinstance(end, Sequence):
                raise ValueError(f"closure_bonds item {number}, {key} is {end!r}; {shape}")
            if len(end) != 3:
                raise ValueError(f"closure_bonds item {number}, {key} is {list(end)!r}; {shape}")
            chain_id, position, atom = end
            if not (isinstance(chain_id, str) and chain_id):
                raise ValueError(
                    f"closure_bonds item {number}, {key}: the chain {chain_id!r} is not a name"
                )
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                raise ValueError(
                    f"closure_bonds item {number}, {key}: the position {position!r} is not an "
                    "integer from 1 (the place of the residue in the sequence sent)"
                )
            if not (isinstance(atom, str) and atom):
                raise ValueError(
                    f"closure_bonds item {number}, {key}: the atom {atom!r} is not a name"
                )
            entry[key] = [chain_id, position, atom]
        if entry["atom1"] == entry["atom2"]:
            raise ValueError(
                f"closure_bonds item {number} joins an atom to itself: {entry['atom1']}"
            )
        bonds.append(entry)
    return bonds


def _closure_bond_entries(
    structure: "gemmi.Structure", input_path: Path, binder: _Chain
) -> list[dict[str, list]]:
    """The ``bond`` constraints of the closures that ``detect_closures`` finds on ``binder``.

    The closure ends are placed in the sequence sent for the binder by ``place_closures``, the
    mapping that the Protenix runner uses. A closure outside the families of
    ``_CLOSURE_BOND_FAMILIES`` is not written and a warning names it.

    Raises:
        ValueError: The closures cannot be searched or a closure cannot be placed.
    """
    chain = next(chain for chain in structure[0] if chain.name == binder.chain_id)
    residues = [(chain[index].name, chain[index].seqid.num) for index in binder.residue_indices]
    way_out = "Pass closure_bonds='none' to write no bond"
    closures = find_closures(
        input_path,
        binder.chain_id,
        fallback_advice="pass closure_bonds='none' to write no bond, or give the bonds as a list",
    )
    placed, not_written = place_closures(
        closures, residues, binder.chain_id, families=_CLOSURE_BOND_FAMILIES, way_out=way_out
    )
    if not_written:
        logger.warning(
            "Chain %s: %s not written as a bond constraint, which Boltz-2 documents for CCD "
            "ligands and canonical residues only (docs/prediction.md, Constraints); the binder "
            "is predicted without it.",
            binder.chain_id,
            "; ".join(not_written),
        )
    entries = [
        {
            "atom1": [binder.chain_id, bond.position1, bond.atom1],
            "atom2": [binder.chain_id, bond.position2, bond.atom2],
        }
        for bond in placed
    ]
    if entries:
        logger.info("Chain %s: %d closure bond(s) written.", binder.chain_id, len(entries))
    return entries


def _planned_bonds(
    entries: Sequence[Mapping[str, Sequence]], chains: Sequence[_Chain]
) -> tuple[_Bond, ...]:
    """The ``entries`` of ``closure_bonds`` as bonds, each end checked against the input.

    Raises:
        ValueError: An end names a chain that is not the receptor or the binder, or a position
            past the end of that chain's sequence.
    """
    length = {chain.chain_id: len(chain.sequence) for chain in chains}
    bonds = []
    for number, entry in enumerate(entries, start=1):
        ends = []
        for key in ("atom1", "atom2"):
            chain_id, position, atom = entry[key]
            if chain_id not in length:
                raise ValueError(
                    f"closure_bonds item {number}, {key}: chain {chain_id!r} is not the receptor "
                    f"or the binder (the chains sent are {', '.join(map(repr, length))})"
                )
            if position > length[chain_id]:
                raise ValueError(
                    f"closure_bonds item {number}, {key}: position {position} is past the end "
                    f"of chain {chain_id!r}, which has {length[chain_id]} residues in the "
                    "sequence sent"
                )
            ends.append((chain_id, int(position), str(atom)))
        bonds.append((ends[0], ends[1]))
    return tuple(bonds)


def _require_chains(
    structure: "gemmi.Structure", input_path: Path, *, receptor: str, binder: str
) -> None:
    """Raise ``ValueError`` when the first model of ``structure`` lacks either chain."""
    chain_ids = [chain.name for chain in structure[0]]
    for role, chain_id in (("receptor", receptor), ("binder", binder)):
        if chain_id not in chain_ids:
            raise ValueError(
                f"{role} chain '{chain_id}' not found in {input_path} "
                f"(chains in its first model: {', '.join(chain_ids) or 'none'})"
            )


def _msa_suffix(path: Path) -> str:
    """The lower-case suffix of a receptor alignment, ``.a3m`` or ``.csv``.

    Boltz-2 compares the suffix of the file it reads as written (``main.py``), so the staged copy
    is named in lower case whatever the caller's file is called.

    Raises:
        ValueError: The file is neither an a3m nor a csv.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _MSA_SUFFIXES:
        raise ValueError(
            f"receptor_msa {Path(path).name} must be a .a3m or .csv file, the two formats "
            "Boltz-2 reads (it fails on any other suffix after printing the error and exits 0)"
        )
    return suffix


def _msa_query(path: Path) -> str:
    """The first sequence of an alignment file: the row Boltz-2 reads as the query.

    A3m: the first line that is not empty, a ``#`` comment or a ``>`` header (Boltz-2 reads one
    sequence per line). Csv: the ``sequence`` column of the first row.

    Raises:
        ValueError: The file holds no sequence, or a csv has no ``sequence`` column.
    """
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        if path.suffix.lower() == ".csv":
            rows = csv.reader(handle)
            header = next(rows, [])
            if "sequence" not in header:
                raise ValueError(
                    f"receptor_msa {path.name} has no 'sequence' column (Boltz-2 expects the "
                    "columns 'key' and 'sequence')"
                )
            first = next(rows, None)
            query = first[header.index("sequence")].strip() if first else ""
        else:
            query = ""
            for line in handle:
                line = line.strip()
                if line and not line.startswith(("#", ">")):
                    query = line
                    break
    if not query:
        raise ValueError(f"receptor_msa {path.name} holds no sequence")
    return query


def _check_msa_of_chains(path: Path, receptor: _Chain, binder: _Chain) -> None:
    """Refuse an alignment that is not the receptor's, or a binder that would share it.

    Boltz-2 reads the first row of an MSA as the query of the entity, with one row position per
    token, and checks neither (a wrong alignment gives wrong features, silently). The first
    sequence is compared with the receptor sequence sent: same length, same letters, an ``X`` or
    ``U`` on either side matching anything. Lower-case letters are insertions and are not counted.
    A binder with the receptor's sequence would be the same entity, with the same MSA.

    Raises:
        ValueError: The first sequence is not the receptor sequence, or the binder has it.
    """
    if binder.sequence == receptor.sequence:
        raise ValueError(
            f"receptor chain '{receptor.chain_id}' and binder chain '{binder.chain_id}' have the "
            "same sequence. Boltz-2 reads chains with one sequence as one entity with one MSA, so "
            "a receptor MSA would be the binder's too, and the binder would not be "
            "single-sequence; receptor_msa cannot be given for this input"
        )
    query = "".join(c for c in _msa_query(Path(path)) if not c.islower())
    sent = receptor.sequence
    if len(query) != len(sent):
        raise ValueError(
            f"receptor_msa {Path(path).name}: its first sequence has {len(query)} residues and is "
            f"not the receptor sequence sent for chain '{receptor.chain_id}' ({len(sent)} "
            "residues); Boltz-2 reads the first row of an MSA as the query of the entity"
        )
    for position, (found, wanted) in enumerate(zip(query.upper(), sent), start=1):
        if found != wanted and not ({found, wanted} & _MSA_WILDCARDS):
            raise ValueError(
                f"receptor_msa {Path(path).name}: its first sequence differs from the receptor "
                f"at position {position} ('{found}' in the file, '{wanted}' in the receptor "
                f"sequence sent for chain '{receptor.chain_id}')"
            )


def _plan(request: PredictionRequest) -> _Plan:
    """Read the structure of ``request`` and decide the entities and the templates.

    Nothing is written; every error about the input is raised here.
    """
    options = request.options
    structure = _read_structure(Path(request.input_path))
    receptor_id, binder_id = request.receptor_chain, request.binder_chain
    _require_chains(structure, request.input_path, receptor=receptor_id, binder=binder_id)

    unmappable = options.get("on_unmappable_residue", _DEFAULT_ON_UNMAPPABLE)
    cyclic = _decide_cyclic(
        request.input_path, binder_id, options.get("binder_cyclic", _DEFAULT_BINDER_CYCLIC)
    )
    receptor = _read_chain(structure, receptor_id, on_unmappable_residue=unmappable, cyclic=False)
    binder = _read_chain(structure, binder_id, on_unmappable_residue=unmappable, cyclic=cyclic)
    chains = (receptor, binder)
    _check_entities(chains)

    templates: tuple[_Template, ...]
    if request.mode == "predict":
        templates = ()
    elif request.mode == "refold":
        templates = (_Template("receptor", (receptor_id,)),)
    elif request.mode == "score":
        templates = (_Template("receptor", (receptor_id,)), _Template("binder", (binder_id,)))
    elif request.mode == "score-lock":
        threshold = float(options["lock_threshold_angstrom"])
        templates = (_Template("lock", (receptor_id, binder_id), force=True, threshold=threshold),)
    elif request.mode == "refold-lock":
        threshold = float(options["lock_threshold_angstrom"])
        templates = (_Template("receptor", (receptor_id,), force=True, threshold=threshold),)
    else:
        raise ValueError(f"the boltz2 runner has no input for mode {request.mode!r}")

    closure_bonds = _closure_bond_option(options.get("closure_bonds", _DEFAULT_CLOSURE_BONDS))
    if closure_bonds == "auto":
        closure_bonds = _closure_bond_entries(structure, Path(request.input_path), binder)
    bonds = () if closure_bonds == "none" else _planned_bonds(closure_bonds, chains)

    msa, msa_file = "default", request.extra_files.get(RECEPTOR_MSA_ROLE)
    if msa_file is not None:
        msa, msa_file = "given", Path(msa_file)
        _msa_suffix(msa_file)
        _check_msa_of_chains(msa_file, receptor, binder)
    elif options.get(RECEPTOR_MSA_ROLE) == SINGLE_SEQUENCE:
        msa = "single"
    return _Plan(
        chains=chains,
        templates=templates,
        structure=structure,
        bonds=bonds,
        msa=msa,
        msa_file=msa_file,
    )


def _decide_cyclic(input_path: Path, binder_chain: str, binder_cyclic: Union[bool, str]) -> bool:
    """Whether the binder entity is written with ``cyclic: true`` (see the module docstring)."""
    if binder_cyclic is True or binder_cyclic is False:
        return binder_cyclic
    from binding_metrics.metrics._openfold_run import _binder_is_head_to_tail

    try:
        head_to_tail = _binder_is_head_to_tail(input_path, binder_chain)
    except Exception as exc:  # noqa: BLE001 - "auto" must not fail a run that works without it
        logger.warning(
            "%s: could not look for a head-to-tail bond in chain %s (%s), so 'cyclic: true' is "
            "not written; binder_cyclic=True writes it regardless.",
            input_path,
            binder_chain,
            exc,
        )
        return False
    if head_to_tail:
        logger.info(
            "Chain %s of %s has a head-to-tail bond: the YAML sets 'cyclic: true' on it.",
            binder_chain,
            input_path,
        )
    return head_to_tail


def _template_structure(plan: _Plan, template: _Template, name: str) -> "gemmi.Structure":
    """The template CIF of ``template`` as a ``gemmi.Structure`` (see the module docstring).

    One polymer entity per chain, named and numbered like the chain; the residues are the ones
    the sequence reader kept, renamed to their parent residue; hydrogens are dropped.
    """
    import gemmi

    by_id = {chain.chain_id: chain for chain in plan.chains}
    source = {chain.name: chain for chain in plan.structure[0]}
    new = gemmi.Structure()
    new.name = name
    model = gemmi.Model("1")
    entities = []
    for chain_id in template.chain_ids:
        planned = by_id[chain_id]
        new_chain = gemmi.Chain(chain_id)
        for position, (index, parent) in enumerate(
            zip(planned.residue_indices, planned.template_names), start=1
        ):
            new_chain.add_residue(source[chain_id][index])
            residue = new_chain[len(new_chain) - 1]
            residue.name = parent
            residue.subchain = chain_id
            residue.label_seq = position
            residue.entity_type = gemmi.EntityType.Polymer
        model.add_chain(new_chain)
        entity = gemmi.Entity(chain_id)
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = [chain_id]
        entity.full_sequence = list(planned.template_names)
        entities.append(entity)
    new.add_model(model)
    for entity in entities:
        new.entities.append(entity)
    new.remove_hydrogens()
    return new


# ---------------------------------------------------------------------------- the runner


def _check_name(name: str) -> None:
    """The name is the stem of the YAML file, so it must be a plain file name."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if "/" in name or "\\" in name or "\0" in name or name in (".", ".."):
        raise ValueError(
            f"name {name!r} must be a plain file name: Boltz-2 names its output folder and "
            "files after the input file"
        )


def _option_name(argument: str) -> str:
    """The name of the option an argument spells, with ``-`` and ``_`` read alike.

    ``--use_msa_server``, ``--use-msa-server`` and ``--use-msa-server=1`` all give
    ``--use_msa_server``; an argument that is not an option (a value, a single-dash word) is
    returned as it is.
    """
    name = argument.split("=", 1)[0]
    return "--" + name[2:].replace("-", "_") if name.startswith("--") else name


def _check_extra_args(extra_args: Sequence[str]) -> list[str]:
    """``extra_args`` as strings; raises when one repeats a flag that the runner sets.

    The flags are compared by name with ``-`` and ``_`` alike, and in both forms (``--seed=3``
    and ``--seed 3``). The installed Boltz-2 (click) accepts the underscore spelling of its own
    options and fails on the hyphen one ("No such option") after the request was stored, so the
    hyphen spelling is refused here as the owned flag it repeats, before anything is written.
    """
    if isinstance(extra_args, (str, bytes)):
        raise TypeError("extra_args must be a sequence of strings, not a string")
    arguments = [str(argument) for argument in extra_args]
    for argument in arguments:
        flag = _option_name(argument)
        if flag in _OWNED_FLAGS:
            written = "" if argument.split("=", 1)[0] == flag else f" (written {argument!r})"
            raise ValueError(
                f"extra_args repeats {flag}{written}, which the runner sets from the request; "
                "change the request instead, or the key would not describe the run"
            )
    return arguments


def _check_write_embeddings(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"write_embeddings must be True or False, got {value!r}")
    return value


def _check_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"lock_threshold_angstrom must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"lock_threshold_angstrom must be a positive number, got {value!r}")
    return float(value)


class Boltz2Runner(PredictionRunner):
    """Runs ``boltz predict`` in the current environment or in a conda environment.

    Args:
        conda_env: Name of the conda environment that has Boltz-2 (``conda run -n <env>``); None
            uses the ``boltz`` on PATH.
    """

    name = "boltz2"
    #: ``boltz predict --checkpoint PATH`` loads one checkpoint file (``main.py:835``).
    supports_custom_weights = True
    weights_kind = "file"
    #: The modes this runner builds an input for. ``predict`` is among them: the YAML is made
    #: from the sequences of the structure, so a caller need not hold a ready input file.
    run_modes = RUN_MODES

    def __init__(self, conda_env: Optional[str] = None):
        self.conda_env = conda_env or None
        self._version: Optional[str] = None
        self._version_probed = False

    # ------------------------------------------------------------------ the machine

    def version(self) -> Optional[str]:
        """The version of the ``boltz`` that ``run`` starts, or None when it cannot be told.

        It describes the installation that ``is_available()`` looks at. With a conda environment
        it is asked of ``conda run -n ENV python``; without one, of the interpreter named on the
        first line of the ``boltz`` script that ``PATH`` finds, so a ``boltz`` installed in an
        environment other than the one that runs this package is still read. Either way a process
        is started, once per runner. It is None when ``boltz`` is not on PATH, when its script is
        not a Python script (a shell wrapper names no interpreter) or when that interpreter cannot
        import ``boltz``; the request key then holds an empty version.
        """
        if not self._version_probed:
            self._version = _installed_boltz_version(self.conda_env)
            self._version_probed = True
        return self._version

    def is_available(self) -> bool:
        """True when ``boltz`` is on PATH, or the conda environment has a ``boltz`` distribution."""
        if self.conda_env is None:
            return shutil.which("boltz") is not None
        return self.version() is not None

    # ------------------------------------------------------------------ the request

    def make_request(
        self,
        input_path: str | Path,
        *,
        name: str,
        binder_chain: Optional[str] = None,
        receptor_chain: Optional[str] = None,
        mode: str = "score",
        seeds: Optional[Sequence[int]] = None,
        num_samples: int = _DEFAULT_NUM_SAMPLES,
        use_msa_server: bool = _DEFAULT_USE_MSA_SERVER,
        on_unmappable_residue: str = _DEFAULT_ON_UNMAPPABLE,
        binder_cyclic: Union[bool, str] = _DEFAULT_BINDER_CYCLIC,
        lock_threshold_angstrom: Optional[float] = None,
        weights: Optional[str | Path | WeightsRef] = None,
        extra_args: Sequence[str] = (),
        closure_bonds: str | Sequence[Mapping] = _DEFAULT_CLOSURE_BONDS,
        write_embeddings: bool = _DEFAULT_WRITE_EMBEDDINGS,
        msa_policy: str = DEFAULT_MSA_POLICY,
        receptor_msa: Optional[str | Path] = None,
    ) -> PredictionRequest:
        """The store request of one Boltz-2 run, with every setting that changes the output.

        The defaults are written out, so two callers that mean the same run get the same key.
        The key holds the Boltz-2 version (``version()``; empty when it cannot be told), the
        mode, the chain roles, the content hash of the input file, the seed, the number of
        samples, the custom weights by content when given, and ``options``: ``use_msa_server`` (what
        the server returns is not recorded), ``binder_cyclic``, ``on_unmappable_residue``,
        ``lock_threshold_angstrom`` (None outside the lock modes) and ``extra_args``. The conda
        environment is not part of it (the version is).

        Args:
            input_path: The complex structure, PDB or mmCIF.
            name: Name of the input; Boltz-2 names its output folder and files after it. A plain
                file name.
            binder_chain, receptor_chain: Chain roles, required and different.
            mode: ``"predict"``, ``"refold"``, ``"score"``, ``"score-lock"`` or ``"refold-lock"``
                (see the module docstring).
            seeds: One seed for ``--seed``; None takes 42. More than one raises, because the
                output has no seed dimension: make one request per seed.
            num_samples: Structures of the run (``--diffusion_samples``).
            use_msa_server: ``--use_msa_server``; False writes ``msa: empty`` (single sequence).
            on_unmappable_residue: ``"error"`` (default) or ``"x"`` (see the module docstring).
            binder_cyclic: ``"auto"`` (default), True or False; whether the binder entity is
                written with ``cyclic: true``.
            lock_threshold_angstrom: The lock modes (``score-lock``, ``refold-lock``) only: the
                threshold of the forced template in angstrom; None takes 2.0, the choice of this
                package.
            weights: A custom (fine-tuned) checkpoint file, or the ``WeightsRef`` that
                ``PredictionStore.weights_reference`` made for it; passed as ``--checkpoint``,
                content in the key. A path is hashed here without a cache; give a
                ``WeightsRef`` to use the store's. None uses Boltz-2's own weights.
            extra_args: Command-line arguments passed to ``boltz predict`` verbatim. They may
                not repeat a flag that the runner sets (``--seed``, ``--diffusion_samples``,
                ``--checkpoint``, ``--use_msa_server`` ...).
            closure_bonds: ``"none"`` (default) writes no ``constraints``; ``"auto"`` writes a
                ``bond`` for each head-to-tail, disulfide and lactam closure that
                ``detect_closures`` finds on the binder of the input, found here and stored in
                the request; a list of ``{"atom1": [chain, position, atom], "atom2": [...]}``
                (positions from 1 in the sequence sent) is written as given. See the module
                docstring.
            write_embeddings: Pass ``--write_embeddings``; the run then leaves
                ``embeddings_<name>.npz`` beside the structures. Part of the key only when True.
            msa_policy: ``"default"`` or ``"forbid-server"``; the second needs
                ``use_msa_server=False`` and a ``receptor_msa``, and keeps ``--use_msa_server``
                off the command. In the key only when it forbids.
            receptor_msa: None, the word ``"single"`` (the receptor is predicted from its own
                sequence) or the path of the receptor's alignment, an ``.a3m`` or ``.csv`` whose
                first sequence is the receptor sequence. The binder is single-sequence in both
                cases. A file is identified by its content in the key, not by its path.

        Raises:
            ValueError: An unknown mode, a missing or equal chain role, a name that is not a plain
                file name, more or fewer than one seed, a choice outside its values, a threshold
                outside the lock modes or not positive, ``extra_args`` that repeat a flag of
                the runner, a ``closure_bonds`` string other than ``"none"`` and ``"auto"`` or a
                bond of the wrong shape; a policy that forbids the server with the server on or
                without a ``receptor_msa``; an alignment that is not an a3m or csv, whose first
                sequence is not the receptor's, or a receptor with the binder's sequence; with
                ``"auto"`` also what ``prepare`` raises about the binder chain and its closures
                (an unreadable structure, a missing chain, a residue Boltz-2 cannot take, a
                closure that cannot be placed).
            TypeError: ``closure_bonds`` is neither a string nor a list, or ``write_embeddings``
                is not a bool.
            OSError: The input file cannot be read.
            FileNotFoundError: ``receptor_msa`` is a path that is not a file.
        """
        if mode not in RUN_MODES:
            raise ValueError(f"mode must be one of {RUN_MODES}, got {mode!r}")
        _check_name(name)
        if not (binder_chain and receptor_chain):
            raise ValueError(f"mode '{mode}' needs binder_chain and receptor_chain")
        if binder_chain == receptor_chain:
            raise ValueError("binder_chain and receptor_chain must be different chains")
        if on_unmappable_residue not in _ON_UNMAPPABLE_CHOICES:
            raise ValueError(
                f"on_unmappable_residue must be one of {_ON_UNMAPPABLE_CHOICES}, "
                f"got {on_unmappable_residue!r}"
            )
        if not (isinstance(binder_cyclic, bool) or binder_cyclic == "auto"):
            raise ValueError(f"binder_cyclic must be True, False or 'auto', got {binder_cyclic!r}")
        if mode in LOCK_MODES:
            threshold: Optional[float] = _check_threshold(
                DEFAULT_LOCK_THRESHOLD_ANGSTROM
                if lock_threshold_angstrom is None
                else lock_threshold_angstrom
            )
        elif lock_threshold_angstrom is not None:
            raise ValueError(
                "lock_threshold_angstrom only applies to the lock modes "
                f"{' and '.join(repr(lock) for lock in LOCK_MODES)}, not to mode '{mode}'"
            )
        else:
            threshold = None
        if seeds is None:
            seed = _DEFAULT_SEED
        else:
            if isinstance(seeds, (str, bytes)):
                raise TypeError("seeds must be a sequence of integers, not a string")
            values = [int(value) for value in seeds]
            if len(values) != 1:
                raise ValueError(
                    "Boltz-2 writes one seed per run and its output has no seed dimension, so "
                    f"a request takes exactly one seed, got {values}; make one request per seed"
                )
            seed = values[0]
        arguments = _check_extra_args(extra_args)
        _check_write_embeddings(write_embeddings)
        fragments = check_msa_request(
            self.name,
            use_msa_server=bool(use_msa_server),
            msa_policy=msa_policy,
            receptor_msa=receptor_msa,
            supports_given_msa=True,
            supports_single=True,
        )
        given = fragments.extra_files.get(RECEPTOR_MSA_ROLE)
        if given is not None:
            _msa_suffix(given)
            structure = _read_structure(Path(input_path))
            _require_chains(structure, input_path, receptor=receptor_chain, binder=binder_chain)
            chains = [
                _read_chain(
                    structure, chain_id, on_unmappable_residue=on_unmappable_residue, cyclic=False
                )
                for chain_id in (receptor_chain, binder_chain)
            ]
            _check_msa_of_chains(given, *chains)
        bonds = _closure_bond_option(closure_bonds)
        if bonds == "auto":
            structure = _read_structure(Path(input_path))
            _require_chains(structure, input_path, receptor=receptor_chain, binder=binder_chain)
            binder = _read_chain(
                structure, binder_chain, on_unmappable_residue=on_unmappable_residue, cyclic=False
            )
            bonds = _closure_bond_entries(structure, Path(input_path), binder)
        options = {
            "use_msa_server": bool(use_msa_server),
            "binder_cyclic": binder_cyclic,
            "on_unmappable_residue": on_unmappable_residue,
            "lock_threshold_angstrom": threshold,
            "extra_args": arguments,
        }
        if bonds and bonds != "none":  # a new field joins the key only when it is not the default
            options["closure_bonds"] = bonds
        if write_embeddings:
            options["write_embeddings"] = True
        options.update(fragments.options)  # the MSA fields are absent at their defaults too

        return PredictionRequest(
            self.name,
            name,
            mode=mode,
            input_path=input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            weights=weights,
            seeds=(seed,),
            num_samples=num_samples,
            model_version=self.version() or "",
            options=options,
            extra_files=fragments.extra_files,
        )

    # ------------------------------------------------------------------ running

    def prepare(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Write the YAML and the template CIFs below ``<work_dir>/input``; return the YAML.

        Raises what ``run`` would raise about the input (an unreadable structure, a chain the
        structure lacks, a residue Boltz-2 cannot take, missing weights) without starting
        Boltz-2. Every check comes before the first file is written.
        """
        self._check_request(request)
        plan = _plan(request)
        input_dir = Path(work_dir).absolute() / "input"
        yaml_path = input_dir / f"{request.name}.yaml"
        # the command of the run is a function of the request alone: refuse a server flag now
        self._assert_no_server_flag(
            request, self._command(request, yaml_path, Path(work_dir).absolute())
        )
        staged_msa = (
            input_dir / "msa" / f"receptor{_msa_suffix(plan.msa_file)}"
            if plan.msa == "given"
            else None
        )
        template_paths = {
            template.stem: input_dir / "templates" / f"{template.stem}.cif"
            for template in plan.templates
        }
        # built in memory first: a template that cannot be made must not leave a half-written input
        structures = {
            template.stem: _template_structure(plan, template, template.stem)
            for template in plan.templates
        }
        text = _yaml_text(
            plan,
            template_paths,
            use_msa_server=bool(request.options.get("use_msa_server", _DEFAULT_USE_MSA_SERVER)),
            msa_path=staged_msa,
        )

        input_dir.mkdir(parents=True, exist_ok=True)
        if structures:
            (input_dir / "templates").mkdir(exist_ok=True)
        for stem, structure in structures.items():
            structure.make_mmcif_document().write_file(str(template_paths[stem]))
        if staged_msa is not None:
            staged_msa.parent.mkdir(exist_ok=True)
            shutil.copyfile(plan.msa_file, staged_msa)
        yaml_path.write_text(text, encoding="utf-8")
        return yaml_path

    def run(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Run Boltz-2 for ``request`` and return the folder its parser loads.

        That folder is ``<work_dir>/boltz_results_<name>/predictions/<name>``. It holds
        ``<name>_model_<r>.cif``, the confidence files and the PAE and PDE arrays that
        ``Boltz2Parser`` reads.

        Raises:
            ValueError, OSError: About the input, as for ``prepare``; ``request.weights`` is a
                directory.
            FileNotFoundError: ``boltz`` is not on PATH and no conda environment is set, or the
                weights file does not exist.
            Boltz2RunError: Boltz-2 exited non-zero.
            RuntimeError: Boltz-2 exited with status 0 but wrote no output of the input (its
                message is part of the text).
        """
        work_dir = Path(work_dir).absolute()
        yaml_path = self.prepare(request, work_dir)
        if self.conda_env is None and shutil.which("boltz") is None:
            raise FileNotFoundError(
                "boltz not found on PATH. Install Boltz-2 here, or pass conda_env='boltz' "
                "(or whichever environment has it)."
            )
        command = self._command(request, yaml_path, work_dir)
        self._assert_no_server_flag(request, command)  # the last check before the process starts
        tail = _run_boltz_command(command)
        predictions = work_dir / f"boltz_results_{request.name}" / "predictions" / request.name
        self._require_output(request, predictions, tail)
        return predictions

    # ------------------------------------------------------------------ arguments

    def _check_request(self, request: PredictionRequest) -> None:
        if request.model != self.name:
            raise ValueError(f"the {self.name} runner cannot run a '{request.model}' request")
        if request.mode not in RUN_MODES:
            raise ValueError(f"mode must be one of {RUN_MODES}, got {request.mode!r}")
        if request.input_path is None:
            raise ValueError("a Boltz-2 request needs an input structure")
        if not (request.binder_chain and request.receptor_chain):
            raise ValueError(f"mode '{request.mode}' needs binder_chain and receptor_chain")
        if len(request.seeds) != 1:
            raise ValueError(f"a Boltz-2 request needs exactly one seed, got {list(request.seeds)}")
        _check_name(request.name)
        options = request.options
        _check_extra_args(options.get("extra_args", ()))
        _closure_bond_option(options.get("closure_bonds", _DEFAULT_CLOSURE_BONDS))
        _check_write_embeddings(options.get("write_embeddings", _DEFAULT_WRITE_EMBEDDINGS))
        check_msa_of_request(self.name, request, supports_given_msa=True, supports_single=True)
        if request.mode in LOCK_MODES:
            _check_threshold(options.get("lock_threshold_angstrom"))
        self.check_weights(request)
        if request.weights is not None and not request.weights.path.is_file():
            raise FileNotFoundError(f"the Boltz-2 weights {request.weights.path} do not exist")

    def _assert_no_server_flag(self, request: PredictionRequest, command: Sequence[str]) -> None:
        """Refuse ``command`` when the request forbids the server and it holds a server flag."""
        if request.options.get("msa_policy", DEFAULT_MSA_POLICY) == FORBID_SERVER:
            assert_no_server_flag(command, self.name)

    def _command(self, request: PredictionRequest, yaml_path: Path, work_dir: Path) -> list[str]:
        """The command line of the run (behind ``conda run`` when an environment is set)."""
        options = request.options
        command = [
            "boltz",
            "predict",
            str(yaml_path),
            "--out_dir",
            str(work_dir),
            "--model",
            "boltz2",
            "--diffusion_samples",
            str(request.num_samples),
            "--seed",
            str(request.seeds[0]),
            "--output_format",
            "mmcif",
            "--write_full_pae",
            "--write_full_pde",
        ]
        if options.get("use_msa_server", _DEFAULT_USE_MSA_SERVER):
            command.append("--use_msa_server")
        if options.get("write_embeddings", _DEFAULT_WRITE_EMBEDDINGS):
            command.append("--write_embeddings")
        if request.weights is not None:
            command += ["--checkpoint", str(request.weights.path)]
        command += [str(argument) for argument in options.get("extra_args", ())]
        if self.conda_env is not None:
            command = ["conda", "run", "-n", self.conda_env, "--no-capture-output", *command]
        return command

    @staticmethod
    def _require_output(request: PredictionRequest, predictions: Path, tail: str) -> None:
        """Raise when ``predictions`` holds no output of the input (see the module docstring)."""
        from binding_metrics.predictors.boltz2 import Boltz2Parser

        if Boltz2Parser().find_files(predictions, request.name).has_output():
            return
        raise RuntimeError(_no_output_message(request, predictions, tail))
