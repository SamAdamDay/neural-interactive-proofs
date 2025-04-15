# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with
respect to the public API: the hyper-parameters and the experiment run function. Version
numbers take the form `MAJOR.MINOR.PATCH`, where bumping each has general the following
meanings:

- `MAJOR`: There are backwards-incompatible changes either to the hyper-parameters
  themselves or how they are interpreted, or there is a backwards-incompatible change to
  the run function.
- `MINOR`: New hyper-parameters are added in a backwards-compatible way, or the run
  function is changed in a backwards-compatible way. We may also bump the `MINOR`
  version on changes to the developer API.
- `PATCH`: A bug has been fixed.

Since the version number is stored with any runs tracked, this allows comparing the
compatibility of two runs and checking whether an old run can be resumed with the
current codebase. If the older run differs by a `MINOR` version, its hyper-parameters
are guaranteed to be compatible, but not if it differs by a `MAJOR` version.

## Unreleased

### Changed

- Renamed `ReinforcementLearningTrainer` to `TensorDictRlTrainer`.
- Refactored the agent-building part of the factory so that which parts to build are
  determined by class properties of the trainer classes, rather than by hard-coding the
  names of the trainers.
- Moved the `ScenarioInstance` dataclass into its own `scenario_instance` module.
- Refactored code validation `RolloutAnalyser` class hierarchy


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
- Option to have the verifier give a decision on a scale, rather than a binary accept or
  reject.
- Utilities to compute the histogram of verifier decisions and thresholded performance.


### Fixed

- Bug where `mean_decision` and `std_decision` were incorrectly logged for pure text
  trainers.


## [1.0.0] - 2025-03-10

First public release
