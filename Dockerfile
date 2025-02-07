# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime AS base

# Set the timezone environmental variable
ENV TZ=Europe/London

# Update the apt sources
RUN apt update

# Upgrade pip
RUN pip install --upgrade pip

# Remove the Pytorch version from the image. We'll be installing our own version later
RUN pip uninstall -y torch torchvision torchaudio

# Unminimize Ubunutu, and install a bunch of necessary/helpful packages
RUN yes | unminimize
RUN DEBIAN_FRONTEND=noninteractive apt install -y ubuntu-server openssh-server python-is-python3 git python3-venv build-essential curl git gnupg2 make cmake g++ python-dev-is-python3 libprimesieve-dev

# Move to the root home directory
WORKDIR /root

# Install Weights & Biases now so we we can log in
RUN pip install wandb

# Invalidate the cache if this argument is different from the last build. Convention:
# use: --build-arg CACHE_BUST=`git rev-parse main`
ARG CACHE_BUST=0
RUN echo "$CACHE_BUST"

# Do all the things which require secrets: set up git, login to Weights &
# Biases and clone the repo
RUN --mount=type=secret,id=my_env,mode=0444 /bin/bash -c 'source /run/secrets/my_env \
    && git config --global user.name "${GIT_NAME}" \
    && git config --global user.email "${GIT_EMAIL}" \
    && wandb login "${WANDB_KEY}" \
    && git clone "${GIT_REPO_URI}" pvg-experiments \
    && mkdir -p .ssh \
    && echo "${SSH_PUBKEY}" > .ssh/authorized_keys'

# Add /root/.local/bin to the path
ENV PATH=/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Copy the scripts to the /usr/local/bin directory
COPY docker/bin/* /usr/local/bin/

# Copy .env file to the project directory
COPY .env /root/pvg-experiments

# Move to the repo directory
WORKDIR /root/pvg-experiments

# Download the source code for PyTorch Image Models (timm), so we can use the training
# scripts
RUN mkdir -p vendor
RUN grep timm== requirements.txt \
    | sed -E --expression='s#timm==(.*)#https://github.com/huggingface/pytorch-image-models/archive/refs/tags/v\1.tar.gz#' \
    | xargs wget -qO- \
    | tar -xzC /root/pvg-experiments/vendor

# Install all the required packages
RUN pip install wheel cython \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements_dev.txt \
    && pip install -e . \
    && pip install nvitop


# The default target doesn't do much else
FROM base AS default

# Go back to the root
WORKDIR /root

# Expose the default SSH port (inside the container)
EXPOSE 22


# The datasets target downloads all the datasets used in the project. This is slower to
# download from the hub, but faster overall if you're using a large dataset
FROM base AS datasets

# Download all the datasets used in the project
RUN python scripts/download_all_datasets.py

# Go back to the root
WORKDIR /root

# Expose the default SSH port (inside the container)
EXPOSE 22
