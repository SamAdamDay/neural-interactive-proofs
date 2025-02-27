Running Experiments
===================

Overview
--------

Running an experiment involves the following two steps:

1. Create a :class:`HyperParameters <nip.parameters.HyperParameters>` object to specify
   all the parameters for the experiment.
2. Call the :func:`run_experiment <nip.run.run_experiment>` function with the
   :class:`HyperParameters <nip.parameters.HyperParameters>` object.

For example, here's how to run a basic code validation experiment with the NIP protocol,
expert iteration (EI) trainer, and default hyper-parameters:

.. code-block:: python

    from nip import HyperParameters, run_experiment

    hyper_params = HyperParameters(
        scenario="code_validation",
        trainer="pure_text_ei",
        dataset="lrhammond/buggy-apps",
        interaction_protocol="nip",
    )

    run_experiment(hyper_params)

See :doc:`../reference/running_experiments` for the API reference.


Specifying Hyper-Parameters
---------------------------

Available hyper-parameters and possible values are listed in the :class:`HyperParameters
<nip.parameters.HyperParameters>` class. Some hyper-parameters are nested. For example,
the ``nip_protocol`` hyper-parameter holds a :class:`NipProtocolParameters
<nip.parameters.protocol.NipProtocolParameters>` object, which in turn holds the
hyper-parameters which are specific to the NIP protocol.

To specify the NIP parameters, you can either pass a :class:`NipProtocolParameters
<nip.parameters.protocol.NipProtocolParameters>` object to the ``nip_protocol``
parameter, or a dictionary, which will be converted to a :class:`NipProtocolParameters
<nip.parameters.protocol.NipProtocolParameters>` object.

.. tabs::
    
    .. code-tab:: python Using NipProtocolParameters

        hyper_params = HyperParameters(
            scenario="code_validation",
            trainer="pure_text_ei",
            dataset="lrhammond/buggy-apps",
            interaction_protocol="nip",
            nip_protocol=NipProtocolParameters(
                max_message_rounds=11,
                min_message_rounds=5,
            ),
        )
    
    .. code-tab:: python Using a dict

        hyper_params = HyperParameters(
            scenario="code_validation",
            trainer="pure_text_ei",
            dataset="lrhammond/buggy-apps",
            interaction_protocol="nip",
            nip_protocol={
                "max_message_rounds": 11,
                "min_message_rounds": 5,
            },
        )

See :doc:`/reference/parameters` for more information about hyper-parameters.


Additional Experiment Settings
------------------------------

The :func:`run_experiment <nip.run.run_experiment>` function has several optional
arguments that allow you to customize the experiment. These are settings that should
not (in theory) affect the results of the experiment. The most important ones are:

.. list-table::
   :header-rows: 1

   * - Argument
     - Description
   * - ``device``
     - The device to run the experiment on, if run locally.
   * - ``use_wandb``
     - Whether to log the experiment to :term:`Weights & Biases`.
   * - ``wandb_project``
     - The :term:`Weights & Biases` project to log the experiment to, if different from the
       default.
   * - ``run_id``
     - The ID of the run to log the experiment to.
   * - ``num_rollout_workers``
     - The number of workers to use for collecting rollout samples in text-based tasks.

See the documentation for :func:`run_experiment <nip.run.run_experiment>` for the full
list of arguments.


Experiment Scripts
------------------

The library comes with a suite of scripts to facilitate running experiments. In these
scripts, the hyper-parameters are specified in a grid, which allows you to run multiple
experiments with different hyper-parameters either in parallel or sequentially. The
scripts also allow configuring logging to :term:`Weights & Biases`.

The following are the available scripts for running hyper-parameter sweeps. See also the
API reference :doc:`/reference/scripts` for a complete list of scripts.

.. list-table::
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`ppo_gi.py </reference/generated/scripts/ppo_gi>`
     - Run a PPO experiment with :term:`graph isomorphism <graph isomorphism scenario>`
       task.
   * - :doc:`solo_agents_gi.py </reference/generated/scripts/solo_agents_gi>`
     - Do supervised training of a single agent on the :term:`graph isomorphism <graph
       isomorphism scenario>` task.
   * - :doc:`ppo_ic.py </reference/generated/scripts/ppo_ic>`
     - Run a PPO experiment with :term:`image classification <image classification
       scenario>` task.
   * - :doc:`solo_agents_ic.py </reference/generated/scripts/solo_agents_ic>`
     - Do supervised training of a single agent on the :term:`image classification
       <image classification scenario>` task.
   * - :doc:`ei_cv.py </reference/generated/scripts/ei_cv>`
     - Run an expert iteration (EI) experiment with the :term:`code validation <code
       validation scenario>` task.

Let's consider the :doc:`ei_cv.py </reference/generated/scripts/ei_cv>` script. This
script contains the variable ``param_grid``, which is a dictionary, where the keys are
hyper-parameters and the values are lists of values for those hyper-parameters. The
script will run an experiment for each combination of hyper-parameters in the grid.

For example, the following grid will run 4 experiments, running the NIP and Debate
protocols with the "introductory" and "interview" level code validation datasets:

.. code-block:: python

    param_grid = dict(
      interaction_protocol=["nip", "debate"],
      dataset_name=["lrhammond/buggy-apps"],
      apps_difficulty=["introductory", "interview"],
      num_iterations=[8],
      rollouts_per_iteration=[200],
      ...
    )

The experiment (which we'll call ``test_difficulty_levels``) can now be run by calling the script with the following command:

.. code-block:: bash

    python scripts/ei_cv.py --use_wandb test_difficulty_levels

This will run the experiments sequentially, logging data to :term:`Weights & Biases`
with run IDs ``test_difficulty_levels_0``, ``test_difficulty_levels_1``, etc.

See the :doc:`documentation for the script </reference/generated/scripts/ei_cv>` for more information on how to run it, or run:

.. code-block:: bash

    python scripts/ei_cv.py --help
