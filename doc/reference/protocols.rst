Interaction Protocols (``nip.protocols``)
=========================================

.. currentmodule:: nip.protocols


Overview
--------

Interaction protocols define the way agents interact with each other. In particular, a
protocol specifies the following:

- **Agents**. The names of the agents involved.
- **Channels**. The communication channels between agents.
- **Turn order**. Which agents are active in each turn.
- **Rewards**. The reward signal for each agent in each turn.


Creating a New Protocol
-----------------------

See :doc:`../guides/new-protocol` for a guide on how to create a new protocol.


Base classes
------------

.. autosummary::
   :toctree: generated/classes
   :recursive:

   protocol_base.ProtocolHandler
   protocol_base.SingleVerifierProtocolHandler
   protocol_base.DeterministicSingleVerifierProtocolHandler


Built-in Protocols
------------------

.. autosummary::
   :toctree: generated/classes
   :recursive:

   main_protocols.NipProtocol
   main_protocols.AdpProtocol
   main_protocols.DebateProtocol
   main_protocols.MerlinArthurProtocol
   main_protocols.MnipProtocol
   main_protocols.SoloVerifierProtocol
   main_protocols.MultiChannelTestProtocol


.. _zero-knowledge-protocols-reference:

Zero-Knowledge Protocols
------------------------

All protocols can be converted to zero-knowledge protocols by settings the
``protocol_common.zero_knowledge`` :term:`hyper-parameter <hyper-parameters>` to
``True``. The way this is implemented is that a :class:`ZeroKnowledgeProtocol
<zero_knowledge.ZeroKnowledgeProtocol>` meta-handler is used as the protocol handler for
the experiment. This handler creates a child handler for the actual protocol, and runs
the zero-knowledge protocol on top of it.

.. autosummary::
   :toctree: generated/classes
   :recursive:

   zero_knowledge.ZeroKnowledgeProtocol


Code Validation Protocols
-------------------------

In order for protocols to be used in code validation scenarios, some additional
configuration is required:

- Various configuration options should be specified, such as the human-readable names of
  the agents.
- System prompt templates should be defined for each agent.

The first item is done by creating and registering a
:class:`nip.code_validation.protocols.CodeValidationProtocolHandler` class, which
subclasses the desired protocol handler, and provides an
:class:`nip.code_validation.protocols.CodeValidationAgentSpec` specification for each
agent. The second is done by creating files of the form ``nip/code_validation/prompt_templates/system_prompts/{protocol_name}/{agent_name}.txt``.

See :doc:`../guides/new-protocol` for more information.


Protocol Registry
-----------------

The following methods handle registering and building protocol handlers:

.. autofunction:: nip.protocols.registry.register_protocol_handler
.. autofunction:: nip.protocols.registry.build_protocol_handler
