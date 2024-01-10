# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Set the timezone environmental variable
ENV TZ=Europe/London

# Update the apt sources
RUN apt update

# Install pip so that we can install PyTorch
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3.10 python3-pip

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
    && wandb login ${WANDB_KEY} \
    && git clone https://$GITHUB_USER:$GITHUB_PAT@github.com/SamAdamDay/pvg-experiments.git pvg-experiments \
    && mkdir -p .ssh \
    && echo ${SSH_PUBKEY} > .ssh/authorized_keys'

# Add /root/.local/bin to the path
ENV PATH=/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Move to the repo directory
WORKDIR /root/pvg-experiments

# Install all the required packages
RUN pip install --upgrade pip \
    && pip install wheel cython \
    && pip install torch==2.3.0.dev20240102+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 \
    && pip install -r requirements.txt \
    && pip install -e . \
    && pip install nvitop

# Apparently this is necessary to fully install primesieve
RUN yes | pip uninstall primesieve && pip install --no-cache-dir primesieve

# Go back to the root
WORKDIR /root

# Expose the default SSH port (inside the container)
EXPOSE 22