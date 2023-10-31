# PVG Experiments

## Installation

1. Clone the repo
2. Create a virtual environment
3. Install perquisites for [`primesieve`](https://pypi.org/project/primesieve/). On
   Ubunut/Debian this looks like:
   ```
   sudo apt install g++ python-dev-is-python3 libprimesieve-dev
   ```
4. Install the requirements: `pip install -r requirements`
5. Install the `pvg` package locally in edit mode: `pip install -e .`


## Using Docker

A docker file is available which allows for iterative development and running
experiments. To build a new image and use it, follow the proceeding steps.

1. Create GitHub personal access token. Ideally use a fine-grained one which has access
   only to the contents of this repository.

2. Create a file named `.env` with the following contents

```bash
GITHUB_USER=
GITHUB_PAT=
GIT_NAME=""
GIT_EMAIL=""
SSH_PUBKEY=""
WANDB_KEY=""
```

3. Fill in the details with your GitHub username, your GitHub PAT, your name as you'd
   like it to appear in git commit messages, the email you'd like to use for git commits
   and the SSH public key you'd like to use to access the container.

4. Build the image using the following command:

```
docker build -t DOCKER_REPO:DOCKER_TAG --secret id=my_env,src=.env --build-arg CACHE_BUST=`git rev-parse main` .
```

replacing `DOCKER_REPO` and `DOCKER_TAG` with the appropriate details.

5. Push the image to the Docker Hub, ready for use.