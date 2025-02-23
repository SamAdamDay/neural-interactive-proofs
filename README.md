# NIP Experiments

## Requirements

- The library requires Python 3.11 or later. 
- You need [git](https://git-scm.com) to clone the repository.
- To log experiment data, you will need a [Weights & Biases](https://wandb.ai/site)
  account.
- To run experiments with OpenAI models, you need an OpenAI API key. You can get one by
  signing up at [OpenAI](https://platform.openai.com). Note that in general the use of
  the OpenAI API is not free.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SamAdamDay/neural-interactive-proofs.git
   ```

2. Change to the repository directory: `cd neural-interactive-proofs`

3. Install the requirements. If you just want to run experiments do:

   ```bash
   pip install wheel
   pip install -r requirements.txt
   ```

   If you also want to make changes to the codebase, do:

   ```bash
   pip install wheel
   pip install -r requirements_dev.txt
   ```

4. Install the library locally in edit mod: `pip install -e .`

5. Log in to Weights & Biases: `wandb login`

6. Copy the template secrets file: `cp .env.template .env`.

   Edit the ``.env`` file and fill in the necessary information for your use case. The
   comments in the file should guide you on what to fill in.


## Running an experiment

The `HyperParameters` class contains all experiment parameters needed for a reproducible
experiment. Running an experiment looks like:

```python
from nip import HyperParameters, run_experiment
hyper_params = HyperParameters(
    "{scenario_name}", 
    "{trainer_name}", 
    "{dataset_name}", 
    **additional_parameters,
)
run_experiment(hyper_params)
```

- `additional_parameters` can include nested dictionaries. See the `parameters` module
  for details.
- `run_experiment` takes additional parameters, like the device and whether to log to
  Weights & Biases.
- The `nip.utils.experiment` contains utility classes for running hyperparameter
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
    * If you installed pre-commit this will be done automatically on each commit.
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


## Git and pull requests

- The main branch is protected so that changes can only come through pull requests
- It's usually a good idea to have a different branch for each feature/PR
- GitHub will run the following checks when you make a PR, which should pass before
  merging.
    * Is everything formatted correctly according to black?
    * Do all the tests pass?


## Using Docker

A docker file is available which allows for iterative development and running
experiments. To build a new image and use it, follow the proceeding steps.

1. Create GitHub personal access token. Ideally use a fine-grained one which has access
   only to the contents of this repository.

2. Create a [Weights & Biases](https://wandb.ai) account and generate an [API
   key](https://wandb.ai/settings#dangerzone)

3. Build the image using the following command:

```
docker build -t DOCKER_USER/DOCKER_REPO:DOCKER_TAG --target default --secret id=my_env,src=.env --build-arg CACHE_BUST=`git rev-parse main` .
```

replacing `DOCKER_USER` with your Docker Hub username, and `DOCKER_REPO` and
`DOCKER_TAG` suitable Docker repository and tag names (e.g. 'neural-interactive-proofs/default').

Alternatively, you can build an image with all of the datasets already downloaded. This
will result in a much larger image, but can make the process of spinning up and running
a new instance faster overall, if using a large dataset. To do this, use the 'datasets'
target as follows:

```
docker build -t DOCKER_USER/DOCKER_REPO:DOCKER_TAG --target datasets --secret id=my_env,src=.env --build-arg CACHE_BUST=`git rev-parse main` .
```

4. Push the image to the Docker Hub, ready for use:

```
docker push DOCKER_USER/DOCKER_REPO:DOCKER_TAG
```
