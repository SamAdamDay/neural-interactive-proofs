Scenarios
=========

What is a Scenario?
-------------------

A scenario contains the specification and implementation of all of the elements of a
classification task in which we want to train agents. An example of such a task is the
code validation problem, where a problem statement together with a purported solution is
given, and the task is to determine whether the solution is correct.

The main components of a scenario are:

- **Datasets**. The train and test datasets, which contain all instances of the task.
- **Environment**. The environment in which the agents interact. Crucially, this
  specifies the message space in which agents exchange messages.
- **Agents**. The agents that interact in the environment. These should be trainable
  models which take as input a task instance and the sequence of messages exchanged so
  far, and output a message to send to the other agent, and potentially a decision.


.. _tensordict-or-pure-text-scenario:

TensorDict or Pure Text Scenario?
---------------------------------

There are two types of scenario depending on how the agent models are implemented.

1. **TensorDict-based scenarios**. These scenarios are those where the agents are
   locally run neural networks, so we need to pass around PyTorch tensors. The data
   structures used are based on PyTorch's :external+tensordict:class:`TensorDict
   <tensordict.TensorDict>` objects. The environment and agents are based on
   :torchrl:doc:`TorchRL <torchrl:index>` components.
2. **Pure text scenarios**. These scenarios are those where the agents are text-based
   models accessed by an API. In this case we need to pass around strings, rather than
   tensors. The data structures used are similar to TensorDicts, but contain nested
   dictionaries of Numpy string arrays, as implemented in the :class:`NestedArrayDict
   <pvg.utils.nested_array_dict.NestedArrayDict>` class.

The :term:`trainer` used must be compatible with the scenario type. See
:ref:`tensordict-or-pure-text-trainer` for more information.


Base Classes
------------

Base classes for all elements that make up a scenario are found in the
``pvg.scenario_base`` module. This contains the following sub-modules:

.. autosummary::
   :toctree: generated
   :recursive:

   pvg.scenario_base.data
   pvg.scenario_base.environment
   pvg.scenario_base.agents
   pvg.scenario_base.pretrained_models
   pvg.scenario_base.rollout_analysis
   pvg.scenario_base.rollout_samples


Available Scenarios
-------------------

Implementations of scenarios are placed in their own modules. The following scenarios
are available:

.. autosummary::
   :toctree: generated
   :recursive:

   pvg.graph_isomorphism
   pvg.image_classification
   pvg.code_validation


Scenario Hyper-Parameters
-------------------------

Each scenario may have its own :term:`hyper-parameters`, which are sub-parameter objects living
in the main :class:`HyperParameters <pvg.parameters.HyperParameters>` object. See
:doc:`generated/pvg.parameters.scenario` for more information.


How scenarios get instantiated (``pvg.factory``)
------------------------------------------------

Every scenario implementation registers its derived classes with
:func:`pvg.factory.register_scenario_class`. When the experiment gets run the
:func:`pvg.factory.build_scenario_instance` function is called, which creates instances
of the classes defined in the scenario, according to some initialisation logic. These
instances are stored in a :class:`pvg.factory.ScenarioInstance`
object, which is passed to the :term:`trainer`.

.. autosummary::
   :toctree: generated
   :recursive:

   pvg.factory