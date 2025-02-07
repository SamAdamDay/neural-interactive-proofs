Trainers (``pvg.trainers``)
===========================

.. currentmodule:: pvg.trainers


Overview
--------

Trainers are responsible for optimising the agents in an experiment. A trainer
(:class:`Trainer <pvg.trainers.trainer_base.Trainer>`) takes as input the following:

1. The hyper-parameter of the experiment (a :class:`HyperParameters
   <pvg.parameters.HyperParameters>` object).
2. A `ScenarioInstance <pvg.factory.ScenarioInstance>` object, which contains all the
   components of the experiment. The most important components are:

    - The datasets.
    - The interaction protocol handler (:class:`ProtocolHandler
      <pvg.protocols.protocol_base.ProtocolHandler>`).
    - The agents.
    - The environment.

3. An `ExperimentSettings <pvg.experiment_settings.ExperimentSettings>` object, which
   contains various settings for the experiment not relevant to reproducibility (e.g.
   the GPU device number and whether to use :term:`Weights & Biases`).

When called, the trainer performs some number of optimisation steps on the agents, using
the environment to generate the training data.


.. _tensordict-or-pure-text-trainer:

TensorDict or Pure Text Trainer?
--------------------------------

There are two types of trainers: those that deal directly with neural networks and those
that interact with text-based models through an API. The former kind use data structures
based on PyTorch's :external+tensordict:class:`TensorDict <tensordict.TensorDict>`
objects, while the latter use a similar, custom data structure containing nested
dictionaries of Numpy string arrays (:class:`NestedArrayDict
<pvg.utils.nested_array_dict.NestedArrayDict>`). TensorDict-based trainers use the
:torchrl:doc:`TorchRL Library <torchrl:index>`.

Which of these two types is appropriate depends on the type of agents in the experiment.
The environment, datasets and agents must be the appropriate type for the trainer. See
:ref:`tensordict-or-pure-text-scenario` for more information.


Base classes
------------

Base classes for trainers are found in the following modules:

.. autosummary::
   :toctree: generated
   :recursive:

   trainer_base
   rl_tensordict_base
   rl_pure_text_base


Available Trainers
------------------

.. autosummary::
   :toctree: generated
   :recursive:

   solo_agent
   vanilla_ppo
   spg
   reinforce
   ei_pure_text
   malt_pure_text


Trainer Registry
----------------

Trainers are registered by using the following function as a decorator:

.. autofunction:: pvg.trainers.registry.register_trainer