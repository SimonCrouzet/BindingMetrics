FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget gcc && apt-get clean

# Install Miniforge
RUN wget -qO /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

COPY . /opt/binding-metrics/
WORKDIR /opt/binding-metrics

RUN mamba env create -f environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/binding-metrics/bin:$PATH

# GPU access is not available during build. After building, verify OpenMM GPU
# support with:
#   docker run --rm --gpus all binding-metrics binding-metrics-check-env
