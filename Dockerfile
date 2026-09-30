FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# The -devel base keeps the compiler toolchain and CUDA headers that Triton and
# torch may need to build kernels on first use. Whether the smaller -runtime
# image would do for OpenFold3 0.5 has not been tested.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        wget gcc \
        libxrender1 libxext6 libsm6 libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean

# Install Miniforge
RUN wget -qO /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /opt/binding-metrics

# Full copy needed for the `pip --no-deps .` step in environment.yml.
COPY . /opt/binding-metrics/
# environment.yml pins numpy<2 in the conda solve itself, so mdtraj/openmm/rdkit/
# the openff stack all resolve to numpy-1 builds with a single, consistent ABI.
# Do NOT `pip install "numpy<2"` afterwards — a pip numpy on top of the conda one
# shadows it and segfaults the openff/AmberTools toolkit registry (GAFF2). GPU
# OpenMM is forced by the cuda-version=12.4 pin in environment.yml.
RUN mamba env create -f environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/binding-metrics/bin:$PATH
# Auto-activate the binding-metrics env in interactive shells.
# Source conda.sh first (conda init writes to .bashrc but Docker bash
# may not read /etc/profile.d), then activate the default env.
RUN echo '. /opt/conda/etc/profile.d/conda.sh && conda activate binding-metrics' >> /root/.bashrc

# GPU access is not available during build. After building, verify OpenMM GPU
# support with:
#   docker run --rm --gpus all binding-metrics binding-metrics-check-env

# Full image — adds the OpenFold3 env (~several GB of ML stack).
# Model weights are NOT included; the entrypoint downloads the default checkpoint
# (openbind-2025-06-30-174k, ~2.3 GB) on first use. Bind-mount a host directory so
# weights persist across runs and host reboots (cloud / studio environments wipe
# Docker named volumes on shutdown — bind mounts to ~ survive):
#
#   mkdir -p ~/.openfold-weights ~/.of3-triton-cache
#   docker run -it --gpus all --shm-size=8g \
#       -v ~/.openfold-weights:/root/.openfold3 \
#       -v ~/.of3-triton-cache:/tmp/triton_cache \
#       simoncrouzet/binding-metrics:full bash
#
# The second mount caches the Triton kernels that OpenFold3 compiles on first use
# (TRITON_CACHE_DIR below), so later runs skip the compile step.
FROM base AS full

# Prevent getpass.getuser() from crashing when the container runs with
# --user $(id -u):$(id -g) and the UID has no /etc/passwd entry.
# Python checks LOGNAME / USER env vars before falling back to pwd.getpwuid().
# Cache dirs default to $HOME which may be unwritable for non-root UIDs;
# redirect them to /tmp, and set HOME=/root so ~ resolves correctly
# (OpenFold3 weights are mounted at /root/.openfold3).
ENV USER=user
ENV HOME=/root
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
ENV TRITON_CACHE_DIR=/tmp/triton_cache
ENV XDG_CACHE_HOME=/tmp/.cache
RUN chmod 777 /root

RUN mamba env create -f environment_openfold3.yml && conda clean -afy

# Entrypoint runs `setup_openfold --non-interactive` when the default checkpoint is
# missing from the volume, and prints a clear error if that fails. Opt out with
# BINDING_METRICS_SKIP_WEIGHTS_CHECK=1.
COPY docker/entrypoint.sh /usr/local/bin/binding-metrics-entrypoint
RUN chmod +x /usr/local/bin/binding-metrics-entrypoint
ENTRYPOINT ["/usr/local/bin/binding-metrics-entrypoint"]
CMD ["bash"]
