FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget gcc && apt-get clean

# Install Miniforge
RUN wget -qO /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /opt/binding-metrics

# OpenFold3 env first — heavy, no local source dependency, cache it early.
# Run setup_openfold once inside the container to download model weights:
#   docker run -it --gpus all binding-metrics conda run -n openfold3 setup_openfold
COPY environment_openfold3.yml ./
RUN mamba env create -f environment_openfold3.yml && conda clean -afy

# Full copy needed for pip install .[all,dev] in environment.yml
COPY . /opt/binding-metrics/
RUN mamba env create -f environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/binding-metrics/bin:$PATH

# GPU access is not available during build. After building, verify OpenMM GPU
# support with:
#   docker run --rm --gpus all binding-metrics binding-metrics-check-env
