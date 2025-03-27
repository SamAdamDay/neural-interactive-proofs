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

- Renamed `ReinforcementLearningTrainer` to `TensorDictRlTrainer`
- Refactored the agent-building part of the factory so that which parts to build are
  determined by class properties of the trainer classes, rather than by hard-coding the
  names of the trainers.
- Moved the `ScenarioInstance` dataclass into its own `scenario_instance` module.


## Added

- A guide to creating a new trainer.
- An overview doc on how an experiment is built and run.


## [1.0.0] - 2025-03-10

First public release
