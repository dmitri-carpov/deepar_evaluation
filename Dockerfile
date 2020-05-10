FROM nvidia/cuda:10.2-cudnn7-devel

ARG EXPERIMENT_ROOT=/experiment
WORKDIR ${EXPERIMENT_ROOT}

# System packages
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      curl \
      git \
      wget \
      unrar \
      unzip && \
    apt-get clean -y

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b  && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Python
RUN conda install -c anaconda -y \
      python=3.7.2 \
      pip

# JupyterLab
RUN conda install -c conda-forge jupyterlab

# Project dependencies and sources
ENV PYTHONPATH=${EXPERIMENT_ROOT}

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -rf /tmp/requirements.txt

ADD . ${EXPERIMENT_ROOT}
