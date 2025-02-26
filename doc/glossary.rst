Glossary
========

.. glossary::
    :sorted:

    scenario
        A scenario contains the specification and implementation of all of the elements
        of a classification task in which we want to train agents. The main components
        of a scenario are the dataset, the environment and the agents.
        
        An example of such a task is the code validation problem, where a problem
        statement together with a purported solution is given, and the task is to
        determine whether the solution is correct.

        Scenarios are implemented by subclassing the base classes in the
        :ref:`nip.scenario_base <scenario-base-classes>` module.

    scenario instance
        A :class:`ScenarioInstance <nip.factory.ScenarioInstance>` object, which
        contains all the components of the experiment. These components are instances of
        the classes defined in the :term:`scenario`.

    graph isomorphism scenario
        A :term:`scenario` where the task is to determine whether two graphs are
        isomorphic.

    image classification scenario
        A :term:`scenario` where the task is to a binary classification of images. This
        is available with a number of image datasets.

    code validation scenario
        A :term:`scenario` where the task is to determine whether a given piece of code
        is a correct solution to a problem statement. Each problem instance consists of a problem statement and a candidate solution.

    interaction protocol
        The set of rules that govern how agents interact in the environment. This
        includes the names of the agents involved, the communication channels between
        agents, the order of play and the reward signal for each agent in each turn.
        Protocols are implemented by subclassing :class:`ProtocolHandler
        <nip.protocols.protocol_base.ProtocolHandler>`. See the guide to 
        :doc:`Creating a new protocol <guides/new-protocol>`.

    deterministic interaction protocol
        An :term:`interaction protocol` where the order of play is fixed and does not vary
        between trajectories. Deterministic protocols with a single verifier are
        implemented by subclassing :class:`DeterministicSingleVerifierProtocolHandler
        <nip.protocols.protocol_base.DeterministicSingleVerifierProtocolHandler>`.

    hyper-parameter
        One of the :term:`hyper-parameters`.
    
    hyper-parameters
        The parameters that define the configuration of an experiment. These are
        implemented by the :class:`HyperParameters <nip.parameters.HyperParameters>`
        class and its sub-parameters classes.

        An experiment should be completely reproducible from its hyper-parameters, up to
        hardware quirks and model API non-reproducibility.

    experiment settings
        An :class:`ExperimentSettings <nip.experiment_settings.ExperimentSettings>`
        object, which contains various settings for the experiment not relevant to
        reproducibility (e.g. the GPU device number and whether to use Weights &
        Biases).

    trainer
        The class that performs the optimisation steps on the agents, using the
        environment to generate the training data. Trainers are implemented by
        subclassing :class:`Trainer <nip.trainers.trainer_base.Trainer>`.

    TensorDict
        A nested dictionary of PyTorch tensors. :tensordict:doc:`Read the documentation
        <tensordict:index>`.

    Weights & Biases
        A tool for tracking and visualising machine learning experiments. See `the
        Weights & Biases website <https://wandb.ai/site>`_.