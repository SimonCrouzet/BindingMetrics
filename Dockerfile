FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        wget gcc cuda-nvcc-12-4 \
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
# Model weights are NOT included; download them once using a persistent volume:
#
#   docker run -it --gpus all \
#       -v openfold3-weights:/root/.openfold3 \
#       simoncrouzet/binding-metrics:full \
#       conda run -n openfold3 setup_openfold
#
# The named volume "openfold3-weights" persists across container runs, so
# weights only need to be downloaded once. For subsequent runs:
#
#   docker run -it --gpus all \
#       -v openfold3-weights:/root/.openfold3 \
#       simoncrouzet/binding-metrics:full bash
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
