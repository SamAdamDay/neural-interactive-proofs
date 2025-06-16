# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with
respect to the public API: the hyper-parameters, the experiment run function and the
language model server API. Version numbers take the form `MAJOR.MINOR.PATCH`, where
bumping each has general the following meanings:

- `MAJOR`: There are backwards-incompatible changes to one of (a) either the
  hyper-parameters themselves, how they are interpreted; (b) the run function; or (c)
  the language server API.
- `MINOR`: New hyper-parameters are added in a backwards-compatible way, the run
  function is changed in a backwards-compatible way, or new language server API
  endpoints are added. We may also bump the `MINOR` version on changes to the developer
  API.
- `PATCH`: A bug has been fixed.

Since the version number is stored with any runs tracked, this allows comparing the
compatibility of two runs and checking whether an old run can be resumed with the
current codebase. If the older run differs by a `MINOR` version, its hyper-parameters
are guaranteed to be compatible, but not if it differs by a `MAJOR` version.

Similarly, it ensures that a client and server which agree on `MAJOR` version are able
to communicate reliably.


## [2.0.0] - 2025-06-16

### Changed

- Renamed `ReinforcementLearningTrainer` to `TensorDictRlTrainer`.
- Refactored the agent-building part of the factory so that which parts to build are
  determined by class properties of the trainer classes, rather than by hard-coding the
  names of the trainers.
- Moved the `ScenarioInstance` dataclass into its own `scenario_instance` module.
- Refactored code validation `RolloutAnalyser` class hierarchy
- Implemented prompt template versions
- Switched to using Jinja for prompt templates
- Defaulting to not generating multiple responses for frozen agents in MALT (this is now
  configurable).
- The mid-point reward estimate for the verifier is now more comprehensive, taking into
  account the reward for not guessing.
- Renamed the `cv_benchmark.py` script to `benchmark_cv.py`.
- Renamed the `ei_cv.py` script to `cv_experiment.py`, and allowed it run MALT
  experiments too.
- The solo agent trainer now uses the train and test splits provided by the dataset,
  rather than using its own.
- For the code validation scenario, the test dataset is split into "train" and
  "validation" sets, and testing can now be done on the validation set rather than the
  test dataset.
- When doing MALT, for testing we sample rollouts in the regular way, rather than
  constructing a MALT tree. 
- When generating preference pairs for MALT, timesteps are now filtered based on whether
  an agent has taken an action, rather than on whether they are active according to the
  protocol handler. This is currently functionally equivalent, but this could change in
  the future.
- The way MALT forests are stored as arrays has changed a little. We now explicitly
  store the parent ID of a node. The "has_positive_and_negative", "is_positive",
  "sampled_positive_example" and "sampled_negative_example" fields have been removed.
  Instead the Boolean fields "is_pair_positive" and "is_pair_negative" record the nodes
  which form the positive-negative pairs for their parents. This is easier to work with
  and less brittle. As a consequence of this, the stats logged when doing MALT are now
  completely different.
- Logging is now done consistently on the module-level with
  `logging.getLogger(__name__)`, rather than on an ad hoc basis.
- Switched to using `async.io` for pure-text rollout generation, rather than
  `multiprocessing.Pool`.
- Renamed `InvalidResponseError` to `UnparsableResponseError`.
- The `cv_experiments.py` script now takes a config file as an argument, which specifies
  the hyper-parameters of the experiment.
- The "vLLM-OpenAI" model provider has been renamed to "SelfHosted", which corresponds
  to models hosted using the language model server, which uses vLLM for inference and
  HuggingFace for training.
- Switched to `uv` as the recommended package and project manager.
- The Dockerfile now uses CUDA version 12.0.


### Added

- A guide to creating a new trainer.
- An overview doc on how an experiment is built and run.
- Ability to use more models for code validation inference using either vLLM or
  OpenRouter.
- Implemented `max_train_size` and `max_test_size` for code validation datasets.
- Allowed setting `repetition_penalty` for code validation agents.
- Logging the proportion of rollouts where the verifier does not make a decision, for
  pure text trainers.
- Ability to specify a custom prompt template for the code validation task.
- Verifier format conformance rollout analyser: how well does the verifier conform to
  the required format?
- Utilities for downloading and loading checkpoints
- The script `download_cv_checkpoints.py` to download code validation checkpoint files
- A utility to compute the decision agreement between rollouts
- Option to have the verifier give a decision on a spectrum, rather than a binary accept
  or reject.
- Utilities to compute the histogram of verifier decisions and thresholded performance.
- Functionality for appending a 'supervisor' message to the chat history before sending
  it to the model, to help it better follow the instructions.
- Enabled analysing a batch of rollouts with `scripts/analyse_cv_rollouts.py`.
- A database of language model metadata. This helps with running a sweep of experiments
  across multiple models.
- The beta parameter can now be specified when doing DPO with the OpenAI API.
- Script and utility for visualising MALT rollouts as a forest of trees.
- The option to select response pairs for MALT by thresholding the difference in
  expected reward between the two.
- The option to run some rounds of expert iteration before MALT.
- The `test_dataset_split`, which controls which dataset split is used for testing.
- Implemented `max_test_size` for graph isomorphism and image classification datasets.
- The `_PartialRolloutNode` instances generated when building a MALT tree can now be
  visualised.
- Enabled terminating an episode when a prover generates an invalid response, giving
  them a penalty.
- Ability to force a code validation experiment to run for more iterations that its
  original hyper-parameters.
- A self-hosting language model server and client. The server controls a vLLM process
  and HuggingFace trainer, allowing for convenient self-hosting of open-weight models.


### Removed

- The `test_size` hyper-parameter, which is now unused.
- The `PureTextAgentParameters.vllm_openai_base_url` parameter, which is now specified
  by `PureTextAgentParameters.language_model_server_scheme_host` and
  `PureTextAgentParameters.vllm_server_port`.
- The `datasets` Dockerfile target, which pre-downloaded all datasets


### Fixed

- Bug where `mean_decision` and `std_decision` were incorrectly logged for pure text
  trainers.
- Bug where one of the provers in the MNIP protocol for code validation got the
  incorrect rewards, due to a mistake with inheritance.
- Design flaw where MALT preference pairs were not generated for the root node.


## [1.0.0] - 2025-03-10

First public release
