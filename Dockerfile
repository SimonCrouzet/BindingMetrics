FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Use the -devel base (not -runtime): DeepSpeed's evoformer_attn JIT-builds
# CUDA extensions on first use, which requires CUDA dev headers like
# cusparse.h that the runtime image does not ship. The devel image also
# bundles nvcc, so cuda-nvcc-12-4 no longer needs separate installation.
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

# Full copy needed for pip install .[all,dev] in environment.yml
COPY . /opt/binding-metrics/
RUN mamba env create -f environment.yml && conda clean -afy
# Force numpy<2 after env creation — OpenMM 8.2 (cuda-version=12.4) needs it,
# but mdtraj's conda package can pull numpy 2.x as a hard dependency.
RUN /opt/conda/envs/binding-metrics/bin/pip install "numpy<2"

ENV PATH=/opt/conda/envs/binding-metrics/bin:$PATH
# Auto-activate the binding-metrics env in interactive shells.
# Source conda.sh first (conda init writes to .bashrc but Docker bash
# may not read /etc/profile.d), then activate the default env.
RUN echo '. /opt/conda/etc/profile.d/conda.sh && conda activate binding-metrics' >> /root/.bashrc

# GPU access is not available during build. After building, verify OpenMM GPU
# support with:
#   docker run --rm --gpus all binding-metrics binding-metrics-check-env

# Full image — adds the OpenFold3 env (~several GB of ML stack).
# Model weights are NOT included; the entrypoint auto-downloads them on
# first use. Bind-mount a host directory so weights persist across runs
# and host reboots (cloud / studio environments wipe Docker named volumes
# on shutdown — bind mounts to ~ survive):
#
#   mkdir -p ~/.openfold-weights ~/.of3-jit-cache
#   docker run -it --gpus all --shm-size=8g \
#       -v ~/.openfold-weights:/root/.openfold3 \
#       -v ~/.of3-jit-cache:/tmp/.cache/torch_extensions \
#       simoncrouzet/binding-metrics:full bash
#
# The second mount caches the JIT-compiled DeepSpeed evoformer_attn kernel
# so subsequent runs skip the 1–3 min nvcc rebuild.
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

# DeepSpeed 0.18.x requires CUTLASS headers + ninja to JIT-build the
# evoformer_attn op on first use. Without these, every OF3 inference that
# hits the template pair stack fails with "Unable to JIT load the
# evoformer_attn op". CUTLASS_PATH must be the env prefix (one level above
# include/cutlass/). conda-forge currently ships CUTLASS 4.x; if the JIT
# nvcc build breaks against it, pin to cutlass=3.5.*.
RUN /opt/conda/bin/conda install -n openfold3 -c conda-forge -y cutlass ninja \
    && /opt/conda/bin/conda clean -afy
ENV CUTLASS_PATH=/opt/conda/envs/openfold3

# Entrypoint auto-downloads OpenFold3 weights on first use of an empty
# volume. Prints a clear error if upstream prompts drift. Opt out with
# BINDING_METRICS_SKIP_WEIGHTS_CHECK=1.
COPY docker/entrypoint.sh /usr/local/bin/binding-metrics-entrypoint
RUN chmod +x /usr/local/bin/binding-metrics-entrypoint
ENTRYPOINT ["/usr/local/bin/binding-metrics-entrypoint"]
CMD ["bash"]
