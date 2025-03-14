######################
Creating a New Trainer
######################

.. currentmodule:: nip.trainers

In this guide, we will walk through the process of creating a new RL :term:`trainer`.

A trainer takes a :term:`scenario instance`, consisting of the environment, agents,
:term:`interaction protocol` handler and other components, and is responsible for the
whole training process, including things like logging, checkpointing, and evaluation.

A number of base classes are provided, which are designed to allow you to specify just
the parts of the training process that are specific to your trainer.

The first decision is whether your trainer will work with :term:`TensorDict`-based
scenarios or pure text scenarios. 

1. :term:`TensorDict`-based trainers work with scenarios where the agents are locally
   run neural networks, so we need to pass around PyTorch tensors, which are stored
   together in :term:`TensorDict` data structured. RL training is done using the
   :external+torchrl:doc:`TorchRL <index>` library.
2. Pure text trainers work with scenarios where the agents are text-based models
   accessed by an API. In this case we need to pass around strings, rather than tensors.
   These trainers typically call an agent method which performs some training using an
   API. 

See :ref:`tensordict-or-pure-text-trainer` and :ref:`tensordict-or-pure-text-scenario`
for more information.

It is recommended that you read :doc:`experiment-build-process` to understand how the
trainer fits into the overall experiment.


Which Parts of this Guide to Read
=================================

1. Read the :ref:`new-trainer-main-steps` section to get an overview of the process of
   creating a new trainer.
2. Look at the flow chart in the :ref:`trainer-base-classes` section to decide which
   base class to subclass.
3. Read the description of the base class you've chosen in the
   :ref:`trainer-base-classes` section.


.. _new-trainer-main-steps:

Main Steps
==========

Here are the main steps to create a new trainer:

1. Add the name of the trainer to :const:`TrainerType <nip.parameters.types.TrainerType>`.
2. (Optional) Create a :class:`SubParameters
   <nip.parameters.parameters_base.SubParameters>` subclass in
   ``nip/parameters/trainers.py`` to hold the trainer-specific parameters
   (see :ref:`creating-new-parameters`).
3. Implement the trainer by subclassing one of the base classes. See
   :ref:`trainer-base-classes` below. Register the trainer with the
   :func:`register_trainer <nip.trainers.registry.register_trainer>` decorator.



.. _trainer-base-classes:

Trainer Base Classes
====================

To choose which base class to subclass, either follow the following flowchart or
directly read the descriptions under each heading below.

.. mermaid::

    flowchart TD
        data_structure_type{{"`What type of data structure 
                               will you use?`"}}
        data_structure_type -->|"`TensorDict`"| tensordict_novelty{{"`Is it enough to specify a
                                                                      new loss function and use
                                                                      the default train loop?`"}}
        data_structure_type -->|pure text| pure_text_class[PureTextRlTrainer]
        data_structure_type -->|other| trainer_class[Trainer]

        tensordict_novelty --->|yes| rl_trainer[TensorDictRlTrainer]
        tensordict_novelty --->|no| tensordict_trainer[TensorDictTrainer]


:class:`TensorDictRlTrainer <nip.trainers.rl_tensordict_base.TensorDictRlTrainer>`
----------------------------------------------------------------------------------

Use this class if your trainer works with :term:`TensorDict` data structures and you're
happy to use a standard TorchRL training loop, but you need to specify a new loss
function. 

To implement a new trainer, subclass this class and define the
:func:`_get_loss_module_and_gae
<nip.trainers.rl_tensordict_base.TensorDictRlTrainer._get_loss_module_and_gae>` method.
This method should return a loss module (an instance of a subclass of :class:`Objective
<nip.rl_objectives.Objective>`) and, optionally, a Generalised Advantage Estimation
(GAE) module (if you're using GAE). The GAE is typically constructed from the loss
module as follows.

.. code-block:: python

    from torchrl.objectives import ValueEstimators

    ...

    loss_module.make_value_estimator(ValueEstimators.GAE, **additional_parameters)
    gae = loss_module.value_estimator

The train and test loops are implemented in the :func:`_run_train_loop
<nip.trainers.rl_tensordict_base.TensorDictRlTrainer._run_train_loop>` and
:func:`_run_test_loop
<nip.trainers.rl_tensordict_base.TensorDictRlTrainer._run_test_loop>` methods,
respectively. You can customise these methods as needed. Look at the source code for
each method to see how to do this.


.. _TensorDictTrainer-base-class:

:class:`TensorDictTrainer <nip.trainers.trainer_base.TensorDictTrainer>`
------------------------------------------------------------------------------

Use this class if your trainer works with :term:`TensorDict` data structures and you
need to implement a custom training loop.

You'll need to implement the :func:`train
<nip.trainers.trainer_base.TensorDictTrainer.train>` method, which should perform
the following steps, as appropriate:

1. Set the seed
2. Run the training loop, logging the results
3. Run the test loop, logging the results
4. Save the models

It is recommended that you define separate ``_run_train_loop`` and ``_run_test_loop``
methods, decorating with :func:`attach_progress_bar
<nip.trainers.trainer_base.attach_progress_bar>` as follows. This will allow the run
script to compute the total number of training steps in the whole process, and also
allow you to customise the progress bar.

.. code-block:: python

    from nip.trainers.trainer_base import attach_progress_bar

    ...

    class MyTrainer(TensorDictTrainer):

        ...

        # The ``attach_progress_bar`` takes a function that returns the total number of
        # iterations for this phase of training.
        @attach_progress_bar(lambda self: self.hyper_params.rl.num_iterations)
        def _run_train_loop(self, iteration_context: IterationContext):

            # Add a description to the progress bar
            iteration_context.set_description("Training")

            ...

        @attach_progress_bar(lambda self: self.hyper_params.rl.num_test_iterations)
        def _run_test_loop(self, iteration_context: IterationContext):

            # Add a description to the progress bar
            iteration_context.set_description("Testing")
            
            ...

It is also recommended that you call these methods in an
:external+python:class:`ExitStack <contextlib.ExitStack>` context manager build using
:func:`_build_test_context
<nip.trainers.trainer_base.TensorDictTrainer._build_test_context>` and
:func:`_build_train_context
<nip.trainers.trainer_base.TensorDictTrainer._build_train_context>`, as follows. This
will ensure we make the appropriate PyTorch configuration.

.. code-block:: python

    from contextlib import ExitStack

    ...

    class MyTrainer(TensorDictTrainer):

        ...

        def train(self):

            ...   
            
            # Run the training loop with the appropriate context managers
            with ExitStack() as stack:
                self._build_train_context(stack)
                self._run_train_loop()

            # Run the test loop with the appropriate context managers
            with ExitStack() as stack:
                self._build_test_context(stack)
                self._run_test_loop()

            ...

Most trainers will be reinforcement learning trainers, but if you're using this class it
may be because you're doing something other than reinforcement learning. So that the
:mod:`factory <nip.factory>` knows which parts of the agents it should build for the
trainer, you should define the ``trainer_type`` class attribute of your
:class:`TensorDictTrainer <nip.trainers.trainer_base.TensorDictTrainer>` subclass.
Currently, this can take the following values.

- ``"rl"`` (default): A reinforcement learning trainer. This means that the factory will
  build policy and value heads for the agents.
- ``"solo_agent"``: A trainer that trains a single agent to solve the task using
  supervised learning. The factory will build only the solo agent head for the agents.

If you want to do something different, you can define a new value for this attribute,
and you many need to modify the :mod:`factory <nip.factory>` to handle this. See
:doc:`experiment-build-process`.


:class:`PureTextRlTrainer <nip.trainers.rl_pure_text_base.PureTextRlTrainer>`
-----------------------------------------------------------------------------

Use this class if your trainer works with pure text data structures.

All subclasses must define at least the :func:`_stage_create_fine_tune_jobs
<nip.trainers.rl_pure_text_base.PureTextRlTrainer._stage_create_fine_tune_jobs>` method
(see below).

Rather than using :term:`TensorDict` objects, these trainers use the custom
:class:`NestedArrayDict <nip.utils.nested_array_dict.NestedArrayDict>` data structure.
This is similar to :term:`TensorDict`, in that it is a nested dictionary, but it
contains Numpy string arrays rather than PyTorch tensors.

The :class:`PureTextRlTrainer <nip.trainers.rl_pure_text_base.PureTextRlTrainer>` class
implements a training loop consisting of multiple stages. The experiment state is saved
after each stage, and the experiment can be resumed from any stage. The stages are as
follows.

1. **Sample rollouts from the environment**. You can customise this stage by overriding
   the :func:`_sample_rollouts
   <nip.trainers.rl_pure_text_base.PureTextRlTrainer._sample_rollouts>` method.
2. **Log statistics for the sampled rollouts**. Logging can be customised by overriding
   the :func:`_get_log_stats
   <nip.trainers.rl_pure_text_base.PureTextRlTrainer._get_log_stats>` method. This
   method returns a dictionary of statistics to log, and when overriding, it is
   recommended to call the superclass method and update the dictionary.
3. **Test the agents**. This stage runs the test loop if specified by the
   :class:`hyper_params.text_rl.test_scheme <nip.parameters.trainers.TextRlParameters>`
   hyper-parameter. Any customisation to the :func:`_sample_rollouts
   <nip.trainers.rl_pure_text_base.PureTextRlTrainer._sample_rollouts>` method will also
   affect the test loop.
4. **Create fine-tune jobs for each agent**. This stage creates API jobs for each group
   of agents which share a model (see :ref:`sharing-weights-between-agents`). This stage
   must be implemented by the subclass, which is done by defining the
   :func:`_stage_create_fine_tune_jobs
   <nip.trainers.rl_pure_text_base.PureTextRlTrainer._stage_create_fine_tune_jobs>`
   method. This method takes as input the rollouts sampled in the first stage and calls
   an appropriate method of each :class:`PureTextSharedModelGroup
   <nip.scenario_base.agents.PureTextSharedModelGroup>` agent group (e.g.
   :func:`create_supervised_fine_tune_job
   <nip.scenario_base.agents.PureTextSharedModelGroup.create_supervised_fine_tune_job>`).
   This is the only method that must be implemented by the subclass.
5. **Wait for all fine-tune jobs to complete**.


:class:`Trainer <nip.trainers.trainer_base.Trainer>`
----------------------------------------------------

This is the base class for all trainers, and can be subclassed if you want to do
something more specialised, which doesn't fit into the other categories. This probably
means you're using a custom data structure. You'll need to implement the :func:`train
<nip.trainers.trainer_base.Trainer.train>` method, which should perform the following
steps, as appropriate:

1. Set the seed
2. Run the training loop, logging the results
3. Run the test loop, logging the results
4. Save the models


.. _trainer-experiment-components:

Available Experiment Components
===============================

This section details the components which are available to trainers. All trainers are
initialised with the following objects:

- A :class:`HyperParameters <nip.parameters.HyperParameters>` instance, which contains
  the :term:`hyper-parameters` specifying the experiment.
- A :class:`ScenarioInstance <nip.scenario_instance.ScenarioInstance>` instance, which
  a dataclass holding all the components of the experiment.
- A :class:`ExperimentSettings <nip.experiment_settings.ExperimentSettings>` instance,
  which contains various :term:`experiment settings` not relevant to reproducibility
  (e.g. the GPU device number and whether to use Weights & Biases).

The following components are derived from these objects. For more information on these
components and how they are built, see :doc:`experiment-build-process`.

Some components are only available to specific base classes, which is indicated in the
description.

The following assumes we are working in a method of a trainer class, so ``self``
refers to the trainer instance.