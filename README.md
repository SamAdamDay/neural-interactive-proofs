# PVG Experiments

## Installation

1. Clone the repo
2. Create a virtual environment
3. Install prerequisites for [`primesieve`](https://pypi.org/project/primesieve/). On
   Ubuntu/Debian this looks like:
   ```
   sudo apt install g++ python-dev-is-python3 libprimesieve-dev
   ```
4. Install the requirements:

   ```
   pip install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu121
   ```

5. Log in to Weights and Biases: `wandb login` (you'll need an [account and API
   key](https://wandb.ai/settings#dangerzone))
6. Install the `pvg` package locally in edit mode: `pip install -e .`


## Using Docker

A docker file is available which allows for iterative development and running
experiments. To build a new image and use it, follow the proceeding steps.

1. Create GitHub personal access token. Ideally use a fine-grained one which has access
   only to the contents of this repository.

2. Create a [Weights and Biases](https://wandb.ai) account and generate an [API
   key](https://wandb.ai/settings#dangerzone)

3. Create a file named `.env` with the following contents

```bash
GITHUB_USER=
GITHUB_PAT=
GIT_NAME=""
GIT_EMAIL=""
SSH_PUBKEY=""
WANDB_KEY=""
```

4. Fill in the details with your GitHub username, your GitHub PAT, your name as you'd
   like it to appear in git commit messages, the email you'd like to use for git
   commits, the SSH public key you'd like to use to access the container and your
   Weight's and Biases API key.

5. Build the image using the following command:

```
docker build -t DOCKER_REPO:DOCKER_TAG --secret id=my_env,src=.env --build-arg CACHE_BUST=`git rev-parse main` .
```

replacing `DOCKER_REPO` and `DOCKER_TAG` with the appropriate details.

5. Push the image to the Docker Hub, ready for use.


## TODOs

- [ ] Currently when doing a rollout the prover and verifier are run each round, even
  though it's only one of their turns. This is because TorchRL passes the value net
  through vmap, and [vmap can't do data-dependent control
  flow](https://github.com/pytorch/functorch/issues/257). This makes the rollout twice a
  slow.

  It's possible to overcome this using an 'interleaved' execution. Under this each
  rollout is actually two rollouts. At the beginning we sample two datapoints in a new
  batch dimension, and each round we swap these two. In the first round we ignore the
  verifier output. This way we use the output of both agents in every round.