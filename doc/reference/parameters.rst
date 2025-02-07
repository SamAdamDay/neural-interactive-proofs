Experiment Hyper-Parameters (``pvg.parameters``)
================================================

.. currentmodule:: pvg.parameters


Overview
--------

The hyper-parameters object is the main API for the package. Hyper-parameters determine
everything about how the experiment is run: the task, trainer, dataset and all
parameters. An experiment should be completely reproducible from its hyper-parameters,
up to hardware quirks and model API non-reproducibility.

A :class:`HyperParameters` object is a nested :func:`dataclass <dataclasses.dataclass>`.
All configuration options (parameters) are listed in its documentation. Some parameters
are themselves collections of sub-parameters. A :class:`HyperParameters` object thus has
a hierarchical (or nested) structure. All hyper-parameter and sub-parameter classes have
:class:`BaseHyperParameters <pvg.parameters.parameters_base.BaseHyperParameters>` as
base.

When creating sub-parameters, you can either pass then as an object of the appropriate
:class:`SubParameters <pvg.parameters.parameters_base.SubParameters>` class, or as a
dictionary. The advantage of the former is that you can use symbol inspection (e.g. in
VS Code) to have easy access to the parameter names and descriptions. If you pass a
dictionary, it will be converted to the appropriate :class:`SubParameters
<pvg.parameters.parameters_base.SubParameters>` class.


Examples
--------

1. Create a parameters object, using default values for ppo parameters, and others

.. code-block:: python

   from pvg.parameters import HyperParameters, AgentsParams, GraphIsomorphismAgentParameters

   hyper_params = HyperParameters(
        scenario="graph_isomorphism",
        trainer="ppo",
        dataset="eru10000",
        agents=AgentsParams(
            prover=GraphIsomorphismAgentParameters(d_gnn=128),
            verifier=GraphIsomorphismAgentParameters(num_gnn_layers=2),
        ),
    )

2. Convert the parameters object to a dictionary

>>> hyper_params.to_dict()
{'scenario': 'graph_isomorphism', 'trainer': 'ppo', 'dataset': 'eru10000', ...}

3. Create a parameters object using a dictionary for the ppo parameters


.. code-block:: python
   
   from pvg.parameters import HyperParameters

   hyper_params = HyperParameters(
        scenario="graph_isomorphism",
        trainer="ppo",
        dataset="eru10000",
        ppo={
            "num_epochs": 100,
            "batch_size": 256,
        },
    )

Specifying agent parameters
---------------------------

The number and names of the agents in the experiment vary depending on the protocol.
Therefore, the sub-object which specifies the parameters for each agent is a special
kind. The :class:`AgentsParameters <pvg.parameters.agents.AgentsParameters>` class is a
subclass of :class:`dict`. The keys are the agent names, and the values are the
parameters for that agent. Note that the names of the agents must match those specified
by the protocol. The :func:`get_protocol_agent_names
<pvg.parameters.get_protocol_agent_names>` function can be used to get the names of the
agents in a protocol.


Converting to and from nested dicts
-----------------------------------

A :class:`HyperParameters` object can be converted to a nested dictionary. Example uses
of this are attaching the hyper-parameters to a :term:`Weights & Biases` run, and
serialising the parameters to store them in a JSON file. To convert
:class:`HyperParameters` object to a dict, use the :func:`to_dict
<pvg.parameters.HyperParameters.to_dict>` method (which is also available for
sub-parameters). This performs the following special operations:

- Some parameter values are not serialisable to JSON (e.g. :class:`AgentUpdateSchedule
  <pvg.parameters.update_schedule.AgentUpdateSchedule>`). These are converted to special
  dictionaries that can be converted back to the original object later.
- Agents get the special key ``is_random`` added to their dictionary. This is a convenient
  way to see just from the dict if the agent selects their actions uniformly at random.

To convert a nested dictionary to a :class:`HyperParameters` object, use the
:func:`from_dict <pvg.parameters.HyperParameters.from_dict>` class method. This will
reconstruct the original object, including all sub-parameters.


.. _creating-new-parameters:

Creating new parameters
-----------------------

New parameters can be added by adding elements to the :class:`HyperParameters` class or
any of the sub-parameters classes. To a new sub-parameters class, just subclass
:class:`SubParameters <pvg.parameters.parameters_base.SubParameters>` and decorate it
with :func:`register_parameter_class
<pvg.parameters.parameters_base.register_parameter_class>` to register it, and
:func:`dataclass <dataclasses.dataclass>` to make it a dataclass, as shown in the
following example:

.. code-block:: python

   from pvg.parameters import SubParameters, register_parameter_class
   from dataclasses import dataclass

   @register_parameter_class
   @dataclass
   class MyNewParameters(SubParameters):
       my_param: int = 10


Main hyper-parameters class
---------------------------

.. autoclass:: HyperParameters

   .. rubric:: Methods

   .. autosummary::
   
      ~HyperParameters.__init__
      ~HyperParameters.construct_test_params
      ~HyperParameters.from_dict
      ~HyperParameters.get
      ~HyperParameters.to_dict


Modules for sub-parameters
--------------------------

Sub-parameter classes are grouped into sub-modules of. Each module contains classes for a
specific part of the hyper-parameters.

.. autosummary::
   :toctree: generated
   :recursive:

   protocol
   trainers
   agents
   dataset
   scenario
   base_run
   message_regression
   update_schedule


Bases classes and enum types
----------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   parameters_base
   types


Module-level functions
----------------------

.. autofunction:: get_protocol_agent_names
.. autofunction:: register_parameter_class
