# PVG Experiments

## Installation (development)

1. Clone the repo
2. Create a virtual environment
3. Install prerequisites for [`primesieve`](https://pypi.org/project/primesieve/). On
   Ubuntu/Debian this looks like:
   ```
   sudo apt install g++ python-dev-is-python3 libprimesieve-dev
   ```
4. Install the requirements (we use a nightly version of TorchRL, which requires a
   nightly version of PyTorch; also installing torch-scatter is a pain):

   ```
   pip install torch==2.3.0.dev20240102+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
   pip install -r requirements_dev.txt --find-links https://download.pytorch.org/whl/nightly/cu121
   ```

5. Log in to Weights and Biases: `wandb login` (you'll need an [account and API
   key](https://wandb.ai/settings#dangerzone))
6. Install the `pvg` package locally in edit mode: `pip install -e .`


## Running an experiment

The `Parameters` class contains all experiment parameters needed for a reproducible
experiment. Running an experiment looks like:

```python
from pvg import Parameters, run_experiment
params = Parameters(
    "{scenario_name}", 
    "{trainer_name}", 
    "{dataset_name}", 
    **additional_parameters,
)
run_experiment(params)
```

- `additional_parameters` can include nested dictionaries. See the `parameters` module
  for details.
- The first three arguments are string enums, so you can use e.g.
  `ScenarioType.SCENARIO_NAME` instead.
- `run_experiment` takes additional parameters, like the device and whether to log to
  Weights and Biases.
- The `pvg.utils.experiment` contains utility classes for running hyperparameter
  experiments in sequence or in parallel.


## Testing

There is a small suite of tests, to catch some bugs. Run them with:

```bash
python -m pytest
```

Make sure all tests pass before committing.


## Style guide

- The code is formatted using [`black`](https://black.readthedocs.io/en/stable/).
   * To format the whole repository, use `black .`
   * Make sure you do this before committing
   * On VS Code use `Ctrl+Shift+I` ("Format Document") to format the current file. It's
     useful to do this regularly.
- The line length is 88 (see
  [rationale](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#line-length))
   * Black does its best to format to this line length
   * Docstrings and comments are wrapped to this length
   * Use the [Rewrap VS Code
     extension](https://marketplace.visualstudio.com/items?itemName=stkb.rewrap) bound
     to `Alt+Q` to re-wrap any comment or docstring to the line-length. Make sure to set
     the ruler to 88 for this project.
- All classes, functions and modules should have a docstring (Copilot helps with this)
   * Docstrings are formatting using the [Numpydoc Style
     Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- Please add type hints to all functions
   * [Jaxtyping](https://docs.kidger.site/jaxtyping/) is used for annotating tensors
     with their shapes.
      + E.g. `fun(arg: Float[Tensor, "batch feature"])`
   * Strict type-checking is not enforced


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


## Wishlist TODOs

These are things which would be nice to have (e.g. for efficiency reasons) but which
aren't essential.

- [ ] Currently when doing a rollout the prover and verifier are run each round, even
  though it's only one of their turns. This is because TorchRL passes the value net
  through vmap, and [vmap can't do data-dependent control
  flow](https://github.com/pytorch/functorch/issues/257). This makes the rollout twice a
  slow.

  It's possible to overcome this using an 'interleaved' execution. Under this each
  rollout is actually two rollouts. At the beginning we sample two datapoints in a new
  batch dimension, and each round we swap these two. In the first round we ignore the
  verifier output. This way we use the output of both agents in every round.
