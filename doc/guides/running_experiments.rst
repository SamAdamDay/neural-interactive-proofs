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
expert iteration trainer and default hyper-parameters:

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

See :doc:`../reference/parameters` for more information about hyper-parameters.


Additional Experiment Settings
------------------------------

The :func:`run_experiment <nip.run.run_experiment>` function has several optional
arguments that allow you to customize the experiment. These are setting which should
not (in theory) affect the results of the experiment. The most important ones are:

.. list-table::
   :header-rows: 1

   * - Argument
     - Description
   * - ``device``
     - The device to run the experiment on, if run locally.
   * - ``use_wandb``
     - Whether to log the experiment to Weights & Biases.
   * - ``wandb_project``
     - The Weights & Biases project to log the experiment to, if different from the
       default.
   * - ``run_id``
     - The ID of the run to log the experiment to.
   * - ``num_rollout_workers``
     - The number of workers to use for collecting rollout samples in text-based tasks.

See the documentation for :func:`run_experiment <nip.run.run_experiment>` for the full
list of arguments.