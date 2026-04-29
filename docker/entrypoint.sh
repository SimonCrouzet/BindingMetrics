#!/bin/bash
# Entrypoint for binding-metrics :full image.
# Ensures OpenFold3 weights are present before running the user command.
set -e

WEIGHTS_DIR="${HOME:-/root}/.openfold3"
BANNER="============================================================"

weights_missing() {
    [ ! -f "$WEIGHTS_DIR/ckpt_root" ] && return 0
    ls "$WEIGHTS_DIR"/*.pt >/dev/null 2>&1 || return 0
    return 1
}

fail_prompt_drift() {
    local reason="$1"
    echo ""
    echo "$BANNER"
    echo "  ERROR: OpenFold3 weights setup failed."
    echo ""
    echo "  Reason: $reason"
    echo ""
    echo "  Most likely cause: upstream OpenFold3 changed the prompts in"
    echo "  'setup_openfold', so the canned answer sequence '\\n\\n1\\nno\\n'"
    echo "  in docker/entrypoint.sh no longer matches the expected inputs."
    echo ""
    echo "  To diagnose, run setup_openfold interactively once:"
    echo ""
    echo "      docker run -it --rm --gpus all \\"
    echo "          -e BINDING_METRICS_SKIP_WEIGHTS_CHECK=1 \\"
    echo "          -e HOME=/root \\"
    echo "          -v ~/.openfold-weights:/root/.openfold3 \\"
    echo "          simoncrouzet/binding-metrics:full bash"
    echo "      # inside the container:"
    echo "      conda run -n openfold3 --no-capture-output setup_openfold"
    echo ""
    echo "  Then update docker/entrypoint.sh to match the new prompt sequence."
    echo "$BANNER"
    exit 1
}

if [ "${BINDING_METRICS_SKIP_WEIGHTS_CHECK:-0}" = "1" ]; then
    exec "$@"
fi

if weights_missing; then
    echo "$BANNER"
    echo "  OpenFold3 weights not found at $WEIGHTS_DIR"
    echo ""
    echo "  One-time setup — downloading the default checkpoint (~2.3 GB)."
    echo "  This will only happen the first time the volume is used."
    echo ""
    echo "    cache dir:   $WEIGHTS_DIR"
    echo "    checkpoint:  openfold3-p2-155k (default only)"
    echo "    integration tests: skipped"
    echo ""
    echo "  To skip this check, set BINDING_METRICS_SKIP_WEIGHTS_CHECK=1."
    echo "$BANNER"

    mkdir -p "$WEIGHTS_DIR"

    # Feed setup_openfold's interactive prompts non-interactively. The
    # sequence below must match the prompt order in upstream OpenFold3:
    #   <enter>  accept default cache dir
    #   <enter>  accept default download dir
    #   1        download only the default checkpoint (openfold3-p2-155k)
    #   no       skip integration tests
    # If upstream changes this sequence, setup will either fail non-zero
    # or exit "successfully" without downloading weights. Both cases are
    # caught below and surfaced with a clear drift error.
    if ! printf '\n\n1\nno\n' | conda run -n openfold3 --no-capture-output setup_openfold; then
        fail_prompt_drift "setup_openfold exited non-zero"
    fi

    if weights_missing; then
        fail_prompt_drift "setup_openfold succeeded but $WEIGHTS_DIR/ckpt_root and/or the .pt weights are still missing"
    fi

    echo "$BANNER"
    echo "  Setup complete. Continuing with your command..."
    echo "$BANNER"
fi

exec "$@"
