# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.1-devel-ubuntu20.04 AS base

# Ports for the language model server and vLLM server
ARG LM_SERVER_PORT=5000
ARG VLLM_SERVER_PORT=8000

# Set the timezone environmental variable
ENV TZ=Europe/London

# Update the apt sources
RUN apt update

# Unminimize Ubunutu, and install a bunch of necessary/helpful packages
RUN yes | unminimize
RUN DEBIAN_FRONTEND=noninteractive apt install -y ubuntu-server openssh-server python-is-python3 git build-essential curl git gnupg2 make cmake g++ python-dev-is-python3

# Install uv version 0.7.13
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

# Install nvitop for monitoring GPU usage
RUN uv tool install nvitop

# Move to the root home directory
WORKDIR /root

# Add /root/.local/bin to the path
ENV PATH=/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install Weights & Biases now so we we can log in
RUN uv tool install wandb

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
    && git clone "${GIT_REPO_URI}" neural-interactive-proofs \
    && mkdir -p .ssh \
    && echo "${SSH_PUBKEY}" > .ssh/authorized_keys'

# Copy the scripts to the /usr/local/bin directory
COPY docker/bin/* /usr/local/bin/

# Copy the home config files to the home directory
COPY docker/home/* /root/

# Copy .env file to the project directory
COPY .env /root/neural-interactive-proofs

# Move to the repo directory
WORKDIR /root/neural-interactive-proofs

# Download the source code for PyTorch Image Models (timm), so we can use the training
# scripts
RUN mkdir -p vendor
RUN grep timm== pyproject.toml \
    | sed -E --expression='s#\s*"timm==(.*)\s*",#https://github.com/huggingface/pytorch-image-models/archive/refs/tags/v\1.tar.gz#' \
    | xargs wget -qO- \
    | tar -xzC /root/neural-interactive-proofs/vendor

# Install all the required packages
RUN uv sync --locked
RUN uv sync --locked --extra lm-server

# The default target doesn't do much else
FROM base AS default

# Go back to the root
WORKDIR /root

# Expose the default SSH port (inside the container)
EXPOSE 22

# The default target doesn't do much else
FROM base AS lm-server

# Make sure the log directory exists
RUN mkdir -p /root/neural-interactive-proofs/log

# Expose the default SSH port and ports for the language model server (inside the
# container)
EXPOSE 22 ${LM_SERVER_PORT} ${VLLM_SERVER_PORT}

# Run the language model server
ENTRYPOINT uv run python scripts/run_lm_server.py --external 2>&1 | tee /root/neural-interactive-proofs/log/lm_server.log
