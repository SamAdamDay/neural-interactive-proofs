Interaction Protocols (``pvg.protocols``)
=========================================

.. currentmodule:: pvg.protocols


Overview
--------

Interaction protocols define the way agents interact with each other. In particular, a
protocol specifies the following:

- **Agents**. The names of the agents involved.
- **Channels**. The communication channels between agents.
- **Turn order**. Which agents are active in each turn.
- **Rewards**. The reward signal for each agent in each turn.


.. _creating-new-protocol:

Creating a New Protocol
-----------------------

To create a new protocol, follow these steps:

1. Add the name of the protocol to :const:`pvg.parameters.types.InteractionProtocolType`.
2. If necessary, create a :class:`pvg.parameters.parameters_base.SubParameters` subclass
   in :doc:`generated/pvg.parameters.protocol` to hold the protocol-specific
   parameters (see :ref:`creating-new-parameters`)
3. Define the implementation of the protocol by subclassing one of
   :class:`ProtocolHandler <protocol_base.ProtocolHandler>`,
   :class:`SingleVerifierProtocolHandler <protocol_base.SingleVerifierProtocolHandler>`
   or :class:`DeterministicSingleVerifierProtocolHandler
   <protocol_base.DeterministicSingleVerifierProtocolHandler>`. This class may use the
   protocol-specific parameters to configure the protocol. Register the class with
   :func:`register_protocol_handler`.


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

   main_protocols.PvgProtocol
   main_protocols.AdpProtocol
   main_protocols.DebateProtocol
   main_protocols.MerlinArthurProtocol
   main_protocols.MnipProtocol
   main_protocols.SoloVerifierProtocol
   main_protocols.MultiChannelTestProtocol


Protocol Registry
-----------------

The following methods handle registering and building protocol handlers:

.. autofunction:: pvg.protocols.registry.register_protocol_handler
.. autofunction:: pvg.protocols.registry.build_protocol_handler
