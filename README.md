# Neural Interactive Proofs Experiments

<p align="center">
    <a href="https://arxiv.org/abs/2412.08897">arXiv</a> |
    <a href="https://openreview.net/forum?id=R2834dhBlo">OpenReview</a> |
    <a href="https://samadamday.github.io/neural-interactive-proofs/splash.html">Website</a> |
    <a href="https://samadamday.github.io/neural-interactive-proofs">Documentation</a>
</p>

This repository houses the code used to run the experiments for the ICLR 2025 paper
'[Neural Interactive Proofs](https://arxiv.org/abs/2412.08897)' by Lewis Hammond and Sam
Adam-Day.

The codebase is designed to be easy to use and extend. Read the [documentation](https://samadamday.github.io/neural-interactive-proofs)
for guides and the API reference.


## Requirements

- The library requires Python 3.11 or later. 
- You need [git](https://git-scm.com) to clone the repository.
- To log experiment data, you will need a [Weights & Biases](https://wandb.ai/site)
  account.
- To run experiments with OpenAI models, you need an OpenAI API key. You can get one by
  signing up at [OpenAI](https://platform.openai.com). Note that in general the use of
  the OpenAI API is not free.


## Installation

See [the installation docs](https://samadamday.github.io/neural-interactive-proofs/guides/installation.html) 
for a more detailed installation guide.

1. Clone the repository:

   ```bash
   git clone https://github.com/SamAdamDay/neural-interactive-proofs.git
   ```

2. Change to the repository directory: `cd neural-interactive-proofs`

3. Install the requirements (ideally inside a virtual environment). If you just want to
   run experiments do:

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

See [the guide to running experiments](https://samadamday.github.io/neural-interactive-proofs/guides/running_experiments.html) 
for a more information.

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


## Contributing

We welcome issues and pull requests! See the [guide to contributing](https://samadamday.github.io/neural-interactive-proofs/guides/contributing.html)
for more information.


## Citation

```bibtex
@inproceedings{neural_interactive_proofs,
    author = {Lewis Hammond and Sam Adam-Day},
    title = {Neural Interactive Proofs},
    booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
    year = {2025},
    eprint={2412.08897},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
}
```
