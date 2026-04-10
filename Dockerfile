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

RUN mamba env create -f environment_openfold3.yml && conda clean -afy
