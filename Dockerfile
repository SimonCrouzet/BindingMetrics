FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget && apt-get clean

# Install Miniforge
RUN wget -qO /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# Mirrors environment.yml: only conda-install what needs conda (GPU OpenMM + GAFF2)
RUN mamba install -c conda-forge -y \
    "python>=3.11" \
    "openmm>=8.0" \
    "openmmforcefields>=0.13" \
    "openff-toolkit>=0.14" \
    && conda clean -afy

COPY . /opt/binding-metrics/
RUN pip install "/opt/binding-metrics[all]"
