#!/bin/bash
# Entrypoint for binding-metrics :full image.
# Ensures the OpenFold3 default checkpoint is present before running the user command.
#
# openfold3 >= 0.5.0 loads the OpenBind-0 checkpoint (of3-ob-2025-06-30-174k.pt) by default and
# stops with "cowardly refusing to perform inference" when that file is missing; it no longer
# downloads it at first use. Preview2 weights (of3-p2-*.pt) of an older volume do not load into it.
set -e

WEIGHTS_DIR="${HOME:-/root}/.openfold3"
DEFAULT_CHECKPOINT_NAME="openbind-2025-06-30-174k"
DEFAULT_CHECKPOINT_FILE="of3-ob-2025-06-30-174k.pt"
BANNER="============================================================"

# openfold3 reads the folder that holds the checkpoints from <cache>/ckpt_root, and uses the
# cache folder itself when that file is missing.
checkpoint_dir() {
    if [ -s "$WEIGHTS_DIR/ckpt_root" ]; then
        head -n 1 "$WEIGHTS_DIR/ckpt_root"
    else
        echo "$WEIGHTS_DIR"
    fi
}

# Only the current default file counts: any other *.pt (a Preview2 checkpoint from an older
# volume) would not let openfold3 >= 0.5 run.
weights_missing() {
    [ ! -f "$(checkpoint_dir)/$DEFAULT_CHECKPOINT_FILE" ]
}

fail_setup() {
    local reason="$1"
    echo ""
    echo "$BANNER"
    echo "  ERROR: OpenFold3 weights setup failed."
    echo ""
    echo "  Reason: $reason"
    echo ""
    echo "  Likely causes: no network access to s3://openfold3-data, no disk space in"
    echo "  $WEIGHTS_DIR, or a setup_openfold that no longer has the --non-interactive option"
    echo "  (it exists from openfold3 0.4.2)."
    echo ""
    echo "  To see the reason, run setup_openfold once by hand:"
    echo ""
    echo "      docker run -it --rm --gpus all \\"
    echo "          -e BINDING_METRICS_SKIP_WEIGHTS_CHECK=1 \\"
    echo "          -e HOME=/root \\"
    echo "          -v ~/.openfold-weights:/root/.openfold3 \\"
    echo "          simoncrouzet/binding-metrics:full bash"
    echo "      # inside the container:"
    echo "      conda run -n openfold3 --no-capture-output setup_openfold --non-interactive"
    echo "$BANNER"
    exit 1
}

if [ "${BINDING_METRICS_SKIP_WEIGHTS_CHECK:-0}" = "1" ]; then
    exec "$@"
fi

if weights_missing; then
    echo "$BANNER"
    echo "  OpenFold3 default checkpoint not found in $(checkpoint_dir)"
    echo ""
    echo "  One-time setup — downloading the default checkpoint (~2.3 GB)."
    echo "  This will only happen the first time the volume is used."
    echo ""
    echo "    cache dir:   $WEIGHTS_DIR"
    echo "    checkpoint:  $DEFAULT_CHECKPOINT_NAME (OpenBind-0, default only)"
    echo "    integration tests: skipped"

    old_weights=$(ls "$(checkpoint_dir)"/of3-p2-*.pt "$(checkpoint_dir)"/of3_ft3_v1.pt 2>/dev/null || true)
    if [ -n "$old_weights" ]; then
        echo ""
        echo "  Older Preview weights are in that folder. They do not load into"
        echo "  openfold3 >= 0.5, so the OpenBind-0 checkpoint is downloaded next to them."
    fi

    echo ""
    echo "  To skip this check, set BINDING_METRICS_SKIP_WEIGHTS_CHECK=1."
    echo "$BANNER"

    mkdir -p "$WEIGHTS_DIR"

    # --non-interactive takes every default of setup_openfold (cache and download dir
    # ~/.openfold3, the default checkpoint, no integration tests) and asks nothing, whether or
    # not pytest is installed. It exists from openfold3 0.4.2. Files already on disk are kept.
    if ! conda run -n openfold3 --no-capture-output setup_openfold --non-interactive; then
        fail_setup "setup_openfold exited non-zero"
    fi

    if weights_missing; then
        fail_setup "setup_openfold succeeded but $(checkpoint_dir)/$DEFAULT_CHECKPOINT_FILE is still missing"
    fi

    echo "$BANNER"
    echo "  Setup complete. Continuing with your command..."
    echo "$BANNER"
fi

exec "$@"
