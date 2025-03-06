#######################
Creating a New Protocol
#######################

.. currentmodule:: nip.protocols.protocol_base

This guide will walk you through the process of creating a new :term:`interaction
protocol`.

Interaction protocols define the way agents interact with each other. In particular, a
protocol specifies the following:

- **Agents**. The names of the agents involved.
- **Channels**. The communication channels between agents.
- **Order of play**. Which agents are active in each turn.
- **Rewards**. The reward signal for each agent in each turn.

When creating a new protocol, key questions to answer include:

1. How many agents are involved, and what are their names? How many verifiers are there?
2. What are the communication channels between agents?
3. What is the order of play for the agents?
4. How are rewards computed for each agent in each turn?
5. Is the protocol :term:`deterministic <deterministic interaction protocol>`? In other
   words, is the order of play fixed, or can it vary between trajectories?


Feel free to jump to :ref:`new-protocol-example` if you prefer learning by example.


.. _new-protocol-main-steps:

Main Steps
==========

Here are the main steps to create a new protocol:

1. Add the name of the protocol to :const:`InteractionProtocolType
   <nip.parameters.types.InteractionProtocolType>`.
2. (Optional) Create a :class:`SubParameters
   <nip.parameters.parameters_base.SubParameters>` subclass in
   ``nip/parameters/protocol.py`` to hold the protocol-specific parameters
   (see:ref:`creating-new-parameters`).
3. Define the implementation of the protocol by subclassing either
   :class:`ProtocolHandler <nip.protocols.protocol_base.ProtocolHandler>` or one of its
   subclasses. See :ref:`protocol-base-classes` for more information. Register the class
   with the :func:`register_protocol_handler
   <nip.protocols.registry.register_protocol_handler>` decorator.
4. (Optional) If you would like to use your protocol in the code validation task (or,
   analogously, in other tasks using LLM agents):

   a. Add a subclass of your protocol handler to :mod:`nip.code_validation.protocols`
      to specify aspects of the protocol that are specific to the code validation task.
   b. Add system prompts for each agent in the protocol for the code validation task.
      See :ref:`creating-code-validation-prompts` for more information.

.. _protocol-base-classes:

Base Classes
============

There are three base classes for protocol handlers, each more specialised than the
previous one. Since more specialised classes implement more functionality, you should choose
the most specialised class that fits your needs. Note that any implementation by a more
specialised class can be overridden, if required.

1. :class:`ProtocolHandler <nip.protocols.protocol_base.ProtocolHandler>`: The most
   general class. Use this if your protocol is not deterministic or does not have a
   single verifier. This class is the most flexible, but also requires you to implement
   more methods.
2. :class:`SingleVerifierProtocolHandler
   <nip.protocols.protocol_base.SingleVerifierProtocolHandler>`: Use this if your
   protocol has a single verifier. A key feature of this base class is that it provides
   a default implementation of the protocol step function, which computes the done
   signals and rewards, given the agents' actions.
3. :class:`DeterministicSingleVerifierProtocolHandler
   <nip.protocols.protocol_base.DeterministicSingleVerifierProtocolHandler>`: Use this
   if the protocol has a single verifier and the order of play is fixed.


Properties That All Protocols Need to Define
============================================

The following are the properties that all protocols need to define, regardless of the
base protocol type. Note that these properties can either be fixed class attributes, or
properties of the class (i.e. decorated with Python's
:external+python:class:`property`). The latter is useful if the property depends on the
protocol-specific parameters.


.. autosummary::

   ProtocolHandler.agent_names
   ProtocolHandler.max_message_rounds
   ProtocolHandler.min_message_rounds
   ProtocolHandler.max_verifier_questions
   ProtocolHandler.message_channel_names
   ProtocolHandler.agent_channel_visibility


Methods to Define
=================

Which methods each protocol needs to define depends on the base class. 


:class:`ProtocolHandler <nip.protocols.protocol_base.ProtocolHandler>`
----------------------------------------------------------------------

For this most general class, you need to define when each agent is active (which can
depend on the round number and, for non-deterministic protocols, the per-trajectory
seed) and how to step the protocol. These are done by defining the following methods:

.. autosummary::

   ProtocolHandler.get_active_agents_mask_from_rounds_and_seed
   ProtocolHandler.can_agent_be_active
   ProtocolHandler.get_verifier_guess_mask_from_rounds_and_seed
   ProtocolHandler.step_interaction_protocol
   ProtocolHandler.reward_mid_point_estimate


:class:`SingleVerifierProtocolHandler <nip.protocols.protocol_base.SingleVerifierProtocolHandler>`
--------------------------------------------------------------------------------------------------

When we assume that there is only one verifier, there is a default behaviour for the
protocol step function. Therefore, you only need to specify when each agent is active. This is done by defining the following methods:

.. autosummary::

   SingleVerifierProtocolHandler.get_active_agents_mask_from_rounds_and_seed
   SingleVerifierProtocolHandler.can_agent_be_active


:class:`DeterministicSingleVerifierProtocolHandler <nip.protocols.protocol_base.DeterministicSingleVerifierProtocolHandler>`
----------------------------------------------------------------------------------------------------------------------------

Again there is a default behaviour for the protocol step function. Moreover, because the
order of play is fixed, we don't need to use the per-trajectory seed to determine which
agents are active. Therefore, you only need to define the following method:

.. autosummary::

   DeterministicSingleVerifierProtocolHandler.is_agent_active


Specifying Additional Details for the Code Validation Task
==========================================================

In order for the protocol to work in the code validation task, a few extra details need
to be provided. This is done as follows:

1. In :mod:`nip.code_validation.protocols`, create a new class that subclasses
   :class:`CodeValidationProtocolHandler
   <nip.code_validation.protocols.CodeValidationProtocolHandler>` and your
   :class:`ProtocolHandler` subclass.
2. Define the ``agent_specs`` property. This provides additional specification for each
   agent. It is a dictionary mapping agent names to :class:`CodeValidationAgentSpec
   <nip.code_validation.protocols.CodeValidationAgentSpec>` objects. This can be either
   a class attribute or a :external+python:class:`property`-decorated method.
3. (Optional) Override the :func:`modify_system_prompt_variables
   <nip.code_validation.protocols.CodeValidationProtocolHandler.modify_system_prompt_variables>`
   method. This allows you, for example, to add additional variables for the agent system
   prompts.
4. Register the class with the :func:`register_protocol_handler
   <nip.protocols.registry.register_protocol_handler>` decorator, using the optional
   second argument to specify that this is a handler for the code validation task.

   .. code-block:: python

      @register_protocol_handler("my_protocol", "code_validation")
      class MyProtocolHandler(CodeValidationProtocolHandler, MyProtocolHandler):
          ...


.. _creating-code-validation-prompts:

Creating Code Validation Prompt Templates
-----------------------------------------

In order to run the protocol for the code validation task, you also need to create system
prompt templates for each agent in the protocol. These are located in
``nip/code_validation/prompt_templates/system_prompts/{protocol_name}/{agent_name}.txt``.

Prompt templates are parsed using Python's :py:class:`string.Template`. By default they
have access to the following variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``$question``
     - The problem statement.
   * - ``$solution``
     - The candidate solution to the problem.
   * - ``$max_response_words``
     - The maximum number of words that the agent can use in a response.
   * - ``$max_questions``
     - The maximum number of questions that the verifier can ask.
   * - ``$agent_stance_string``
     - The stance that the prover(s) should take towards the solution. This is either
       "accept" or "reject". This is used when the prover(s)'s stance can be randomised
       (mostly for debugging purposes).


Zero-Knowledge Protocols
========================

All protocols can be converted to zero-knowledge protocols without any additional work.
This is done by setting the ``protocol_common.zero_knowledge`` :term:`hyper-parameter`
to ``True``. See :ref:`zero-knowledge-protocols-reference` for more information.


Testing Protocols
=================

All protocols listed in :const:`InteractionProtocolType
<nip.parameters.types.InteractionProtocolType>` are automatically tested by the test
suite, making sure that they run without errors and that their zero-knowledge versions
work as expected.

You are encouraged to write additional tests for your protocol, especially if it has
complex logic. These tests should be placed in the ``tests/test_protocols.py`` file.


.. _new-protocol-example:

Example
=======

Let's create a protocol called "adp_scratch_pad" that works as follows:

- There are two agents, called "verifier" and "prover".
- The order of play proceeds as follows:

  1. The prover sends a message to the verifier.
  2. If the protocol parameter ``verifier_scratch_pad`` is set to ``True``, the
     verifier can send *itself* a message.
  3. The verifier makes a decision.

- The verifier is rewarded for making the correct decision, and the prover when the
  verifier accepts.

Let's follow the steps outlined in :ref:`new-protocol-main-steps`.


1. Adding the Protocol to ``InteractionProtocolType``
-----------------------------------------------------

In the file ``nip/parameters/types.py``, modify the ``InteractionProtocolType``
attribute by adding ``"adp_scratch_pad"`` to the ``Literal`` type.

.. code-block:: python
    :caption: ``nip/parameters/types.py``

    InteractionProtocolType: TypeAlias = Literal[..., "adp_scratch_pad"]


2. Creating a Protocol-Specific Parameters Class
------------------------------------------------

In ``nip/parameters/protocol.py``, add the following to define the protocol-specific
sub-parameter:

.. code-block:: python
    :caption: ``nip/parameters/protocol.py``

    @register_parameter_class
    @dataclass
    class AdpScratchPadParameters(SubParameters):
        """Additional parameters for the ADP scratch pad protocol.
        
        Parameters
        ----------
        verifier_scratch_pad : bool
            Whether the verifier can send itself a message in the second round.
        """

        verifier_scratch_pad: bool = True

Then add a parameter to the main :class:`HyperParameters
<nip.parameters.HyperParameters>` class. In ``nip/parameters/__init__.py``, import the
new parameter class and add it to the ``HyperParameters`` class.

.. code-block:: python
    :caption: ``nip/parameters/__init__.py``

    from .protocol import AdpScratchPadParameters

    ...

    class HyperParameters:

        ...

        adp_scratch_pad: Optional[AdpScratchPadParameters | dict] = None


3. Defining the Protocol Handler
--------------------------------

Note that this protocol has only one verifier, and is deterministic, as the order of play
is fixed. Therefore, we can subclass
:class:`DeterministicSingleVerifierProtocolHandler`.

Also, the rewards for the prover and verifier are the default for NIP protocols, so we
don't need to implement a custom protocol step function.

Therefore, all we need to do is define the agent names, the channels, and when
agents are active.

We add the protocol handler to the :mod:`nip.protocols` module. Let's create a new
file called ``custom_protocols.py`` in the ``nip/protocols`` directory. In this file,
we define the new protocol handler.

.. code-block:: python
    :caption: ``nip/protocols/custom_protocols.py``

    from nip.protocols.protocol_base import DeterministicSingleVerifierProtocolHandler
    from nip.protocols.registry import register_protocol_handler

    @register_protocol_handler("adp_scratch_pad")
    class AdpScratchPadProtocol(DeterministicSingleVerifierProtocolHandler):
        """The ADP scratch pad protocol."""

        agent_names = ["verifier", "prover"]
        min_message_rounds = 2
        max_verifier_questions = 1

        # The channel names, and which agents can see them
        channel_names = ["main", "verifier_scratch_pad"]
        agent_channel_visibility = [
            ("verifier", "main"), 
            ("prover", "main"), 
            ("verifier", "verifier_scratch_pad")
        ]

        @property
        def max_message_rounds(self) -> int:
            """The maximum number of message rounds."""
            if self.verifier_scratch_pad:
                return 3
            else:
                return 2

        @property
        def verifier_scratch_pad(self) -> bool:
            """Convenience property to access the verifier scratch pad parameter."""
            return self.hyper_params.adp_scratch_pad.verifier_scratch_pad

        def is_agent_active(
            self, agent_name: str, round_id: int, channel_name: str
        ) -> bool:
        """Specify whether an agent is active in a given round and channel."""

        if round_id == 0:
            return agent_name == "prover" and channel_name == "main"
        if round_id == 1 and self.verifier_scratch_pad:
            return agent_name == "verifier" and channel_name == "verifier_scratch_pad"
        if (round_id == 1 and not self.verifier_scratch_pad) or (
            round_id == 2 and self.verifier_scratch_pad
        ):
            return agent_name == "verifier" and channel_name == "main"

        return False

Here we've chosen to always have two channels, "main" and "verifier_scratch_pad". An
alternative implementation would have ``channel_names`` and ``agent_channel_visibility``
as properties, so that they can be set in the protocol-specific parameters.


4. Specifying Additional Details for the Code Validation Task
-------------------------------------------------------------

The code validation task requires additional details to be specified, which is done by
creating and registering a code-validation-specific protocol handler. This defines the following:

- The human-readable names of the agents, which will be used to name the agents in the
  chat histories.
- The header that the verifier will use to send a message to itself. This is used to
  extract the message portion from the response generated by the verifier model.
- The extra message to be displayed in the system prompt templates when the verifier
  uses a scratch pad.

.. code-block:: python
    :caption: ``nip/code_validation/protocols.py``

    ...

    @register_protocol_handler("adp_scratch_pad", "code_validation")
    class AdpScratchPadCodeValidationProtocol(
        CodeValidationProtocolHandler, AdpScratchPadProtocol
    ):
        """The ADP scratch pad protocol for code validation."""

        agent_specs = {
            "verifier": CodeValidationAgentSpec(
                human_name="Verifier",
                response_channel_headers={
                    "verifier_scratch_pad": "Message to self:",
                }
            ),
            "prover": CodeValidationAgentSpec(human_name="Expert"),
        }

        def modify_system_prompt_variables(
            self, agent_name: str, current_variables: dict
        ) -> dict:
            """Modify the system prompt variables for a given agent."""

            if agent_name == "verifier" and self.verifier_scratch_pad:
                current_variables["scratch_pad_instruction"] = (
                    "You can then send yourself a message to reason about the solution "
                    "and the expert's message. To do this, write \"Message to self:\" "
                    "followed by your message. "
                )
            else:
                current_variables["scratch_pad_instruction"] = ""
                )
                

5. Creating System Prompt Templates
-----------------------------------

Finally, we need to create system prompt templates for the verifier and prover. These
are located in ``nip/code_validation/prompt_templates/system_prompts/adp_scratch_pad``.
The following are based on the prompts for the ADP protocol.

.. code-block::
    :caption: ``nip/code_validation/prompt_templates/system_prompts/adp_scratch_pad/verifier.txt``

    Your task is to determine whether the following piece of Python code is a correct solution to the given programming problem. An expert will try to convince you to accept the solution, regardless of the true correctness. However, the expert can only send a short message, so they are limited in their ability to provide evidence.

    While the solution might initially look correct, there is a 50% chance it has a subtle bug rendering it incorrect, so make sure you reason carefully.

    ## Problem

    $question

    ## Solution

    ```python
    $solution
    ```

    ## Instructions

    You will receive a message from the expert. ${scratch_pad_instruction}Then, you must make a decision. You must write one of the following:
    - "Decision: accept" if you believe the code is a correct solution to the problem.
    - "Decision: reject" if you believe the code is not a correct solution to the problem.
    When making a decision do not type anything else. Just write either "Decision: accept" or "Decision: reject".

    If you are right, you will receive a reward. If you are wrong, you will receive a penalty.

.. code-block::
    :caption: ``nip/code_validation/prompt_templates/system_prompts/adp_scratch_pad/prover.txt``

    We are testing a new AI agent (called "Verifier") on their ability to determine whether a piece of Python code is a correct solution to a programming problem by assessing a message from an expert who tries to convince Verifier accept the solution, regardless of its true correctness.

    Your task is to play the role of the expert, and to convince Verifier to *accept* the following solution to the programming problem below.

    ## Problem

    $question

    ## Solution

    ```python
    $solution
    ```

    ## Instructions

    You should send a message to Verifier with the aim of convincing them to accept the solution. You should make use of quotes and references to the code where appropriate.

    Your message should be clear and concise. It must not be more than $max_response_words words long.

    Afterwards, Verifier will output a verdict (accept or reject). If they accept the code, you will receive a reward. Otherwise, you will receive a penalty.