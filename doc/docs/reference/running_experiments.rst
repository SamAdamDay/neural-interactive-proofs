Running Experiments (``nip.run``)
=================================

.. currentmodule:: nip.run

Basic Workflow
--------------

An experiment is built and run using the :func:`run_experiment()
<nip.run.run_experiment>` function. This takes as input a :class:`HyperParameters
<nip.parameters.HyperParameters>` object, as well as various configuration options. The
basic workflow is as follows:

1. Create a :class:`HyperParameters <nip.parameters.HyperParameters>` object. This
   specifies all the parameters for the experiment. In theory an experiment should be
   completely reproducible from its hyper-parameters (in practice, things like hardware
   quirks prevent this).
2. Call :func:`run_experiment() <nip.run.run_experiment>` with the hyper-parameters
   object and other configuration options. These options specify things like the device
   to run on, and whether to save the results to Weights & Biases. These
   additional options should not affect the experiment's outcome (in theory).

The :func:`run_experiment() <nip.run.run_experiment>` function takes care of setting up
all the experiment components, running the experiment, and saving the results. It is
designed to be as simple as possible, while still allowing for a wide range of
experiments.

See :doc:`../guides/running_experiments` for a more detailed guide on running
experiments and :doc:`../guides/experiment-build-process` for an overview of how the
experiment components are built by the :func:`run_experiment() <nip.run.run_experiment>`
function.


Example
-------

Run a graph isomorphism experiment using PPO, with a few custom parameters:

.. code-block:: python

    from nip import run_experiment
    from nip.parameters import HyperParameters, AgentsParams, GraphIsomorphismAgentParameters

    hyper_params = HyperParameters(
        scenario="graph_isomorphism",
        trainer="ppo",
        dataset="eru10000",
        agents=AgentsParams(
            prover=GraphIsomorphismAgentParameters(d_gnn=128),
            verifier=GraphIsomorphismAgentParameters(num_gnn_layers=2),
        ),
    )

    run_experiment(hyper_params, device="cuda", use_wandb=True, run_id="my_run")


Preparing Experiments
---------------------

If you are running multiple experiments (e.g. with a hyper-parameter sweep), it can be
convenient to do some preparation in advance, such as downloading datasets. This is
especially important if experiments are run in parallel, as downloading the same dataset
multiple times can be slow, wasteful, and potentially lead to errors.

The :func:`prepare_experiment() <nip.run.prepare_experiment>` function is designed to
help with this. It takes a :class:`HyperParameters <nip.parameters.HyperParameters>`
object and simulates building all experiment components, without actually running the
experiment. It also returns some information about the experiment, such as the total
number of steps taken by the trainer (useful for progress bars).


Module Contents
---------------

.. autofunction:: run_experiment
.. autofunction:: prepare_experiment
.. autoclass:: PreparedExperimentInfo
