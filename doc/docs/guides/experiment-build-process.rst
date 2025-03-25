How an Experiment is Built
==========================

The is an overview of the main steps taken when the :func:`run_experiment
<nip.run.run_experiment>` function is called, which is the main entry point for running
an experiment.

We assume that code of the following form has been executed:


.. code-block:: python

    from nip import HyperParameters, run_experiment

    hyper_params = HyperParameters(...)
    run_experiment(hyper_params)

1. We see if the hyper-parameters need to be modified. There are two reasons we might
   want to do this, both controlled by the :class:`BaseRunParameters
   <nip.parameters.base_run.BaseRunParameters>` sub-parameters, located at
   ``hyper_params.base_run``.

   - If ``hyper_params.base_run`` is ``"parameters"``, we copy the parameters from a
     previous run stored in Weights & Biases. This is useful if we want resume a run.
   - If ``hyper_params.base_run`` is ``"rerun_tests"``, we copy the parameters from a
     previous run stored in Weights & Biases, except for parameters which control how
     tests are run. This is useful if we have a previous run without tests, and we want
     to rerun it, just doing the testing loop.

#. We set up Weights & Biases, if the ``use_wandb`` argument of :func:`run_experiment
   <nip.run.run_experiment>` is set to ``True``.

#. An :class:`ExperimentSettings <nip.experiment_settings.ExperimentSettings>` object is
   created, which contains various settings for the experiment not relevant to
   reproducibility (e.g. the GPU device number, and the Weights & Biases run).

#. The :class:`ScenarioInstance <nip.scenario_instance.ScenarioInstance>` object is
   created, which contains all the components of the experiment. This is done by calling
   the :func:`build_scenario_instance <nip.factory.build_scenario_instance>` function,
   which executes the following steps.

   i. We build the :class:`ProtocolHandler
      <nip.protocols.protocol_base.ProtocolHandler>`, which will handle the interaction
      protocol. This is done by calling the :func:`build_protocol_handler
      <nip.protocols.registry.build_protocol_handler>` function, which looks for the
      appropriate protocol handler for the parameters in the registry.

   #. The train and test datasets are loaded, by initialising the appropriate
      :class:`Dataset <nip.scenario_base.data.Dataset>` class.

   #. We build the agents. Agents typically consist of multiple parts, and which parts
      get built depends on the hyper-parameters. Each agent specified in the
      :class:`AgentsParameters <nip.parameters.agents.AgentsParameters>` object located
      at ``hyper_params.agents`` is built in the following steps. Here ``agent_params =
      hyper_params.agents[agent_name]`` is an instance of :class:`AgentParameters
      <nip.parameters.agents.AgentParameters>`.

      a. We set the seed based on ``hyper_params.seed`` and the agent name.

      #. Agents are either composed of parts, like bodies and heads, or are a single
         entity (a :class:`WholeAgent <nip.scenario_base.agents.WholeAgent>`). Which of
         these options pertains, and which parts are built, is determined by the
         hyper-parameters. For example, :term:`TensorDict`-based RL trainers require
         agents consisting of parts, with a policy head and a value head (see
         :doc:`new-trainer` for more information). These parts are built by initialising
         the appropriate :class:`AgentPart <nip.scenario_base.agents.AgentPart>`
         classes.

      #. An instance of an :class:`Agent <nip.scenario_base.agents.Agent>` dataclass is
         created, which holds all the parts of the agent.

      #. If we're loading a checkpoint (i.e.
         ``agent_params.load_checkpoint_and_parameters`` is ``True``), we load the
         checkpoint and parameters from the Weights & Biases run specified by
         ``agent_params.checkpoint_run_id``. Otherwise, we let the agent's weights be
         initialised randomly.

   #. If set in the hyper-parameters, pretrained embeddings for each agent are loaded
      into the datasets. This is done by initialising the appropriate
      :class:`PretrainedModel <nip.scenario_base.pretrained_models.PretrainedModel>`
      class, and generating embeddings.

   #. If the trainer and scenario are pure-text based (see
      :ref:`tensordict-or-pure-text-trainer` and
      :ref:`tensordict-or-pure-text-scenario`), we also build shared model groups
      (instances of :class:`PureTextSharedModelGroup
      <nip.scenario_base.agents.PureTextSharedModelGroup>`). These provide an interface
      for dealing with agents which share an underlying model, allowing for running
      fine-tuning jobs on a group level rather than on an agent level.

   #. For RL trainers, the following additional components are built.

      a. The train and test environments are built, by initialising the appropriate
         :class:`Environment <nip.scenario_base.environment.Environment>` class.

      #. The agent parts are combined into combined agent parts (instances of
         :class:`CombinedAgentPart <nip.scenario_base.agents.CombinedAgentPart>`). Each
         combined agent part contains the corresponding parts of all agents, so can be
         treated as a single actor in reinforcement learning (with observations and
         actions indexed by a new agent dimension). This allows working easily with the
         :external+torchrl:doc:`TorchRL <index>` library.

   #. The trainer is built, by initialising the appropriate :class:`Trainer
      <nip.trainers.trainer_base.Trainer>` class.

   #. Finally, the trainer is run by calling the :func:`train
      <nip.trainers.trainer_base.Trainer.train>` method of the trainer.