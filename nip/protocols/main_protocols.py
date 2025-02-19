"""Implementations of the main interaction protocols.

A protocol is implemented by a protocol handler, which specifies the agents present, how
they interact, and how the environment is updated.

Every protocol handler is a subclass of `ProtocolHandler` and registers itself with the
use of the `register_protocol_handler` decorator. The `build_protocol_handler` factory
function can then be used to build a protocol handler from parameters.

The following protocols are implemented:

- NIP: The base Prover-Verifier-Game protocol.
- Abstract Decision Problem :cite:p:`Anil2021`: A version of the NBIP protocol exactly
  two rounds.
- Debate :cite:p:`Irving2018`: A protocol in which two provers debate a question with a
  verifier.
- Merlin-Arthur :cite:p:`Waeldchen2022`: A protocol with two provers where only one
  prover is active at a time, and this is determined randomly.
- MNIP: A variant of the NIP protocol with two provers and a verifier.
"""

from math import ceil, floor

from torch import Tensor

from tensordict import TensorDictBase

from einops import rearrange

from jaxtyping import Int, Bool, Float

from nip.parameters import InteractionProtocolType
from nip.protocols.protocol_base import (
    SingleVerifierProtocolHandler,
    DeterministicSingleVerifierProtocolHandler,
)
from nip.protocols.registry import register_protocol_handler
from nip.utils.nested_array_dict import NestedArrayDict


@register_protocol_handler("nip")
class NipProtocol(DeterministicSingleVerifierProtocolHandler):
    """Handler for the NIP protocol.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    message_channel_names = ["main"]
    agent_channel_visibility = [("prover", "main"), ("verifier", "main")]
    agent_names = ["prover", "verifier"]

    @property
    def verifier_first(self) -> bool:
        """Whether the verifier goes first."""
        return self.hyper_params.protocol_common.verifier_first

    @property
    def max_message_rounds(self) -> int:
        """The maximum number of message rounds."""
        return self.hyper_params.nip_protocol.max_message_rounds

    @property
    def max_verifier_questions(self) -> int:
        """The maximum number of questions the verifier can make to the other agents."""
        if self.hyper_params.protocol_common.verifier_first:
            return ceil(self.max_message_rounds / 2)
        else:
            return floor(self.max_message_rounds / 2)

    @property
    def min_message_rounds(self) -> int:
        """The minimum number of message rounds."""
        return self.hyper_params.nip_protocol.min_message_rounds

    def is_agent_active(
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:
        """Specify whether an agent is active in a given round and channel.

        An agent must only send a message in a channel which is visible to them.

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

        if self.verifier_first:
            if agent_name == "verifier":
                return round_id % 2 == 0
            elif agent_name == "prover":
                return round_id % 2 == 1
        else:
            if agent_name == "prover":
                return round_id % 2 == 0
            elif agent_name == "verifier":
                return round_id % 2 == 1


@register_protocol_handler("adp")
class AdpProtocol(NipProtocol):
    """Implementation of the Abstract Decision Problem protocol :cite:p:`Anil2021`.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    agent_names = ["prover", "verifier"]
    max_message_rounds = 2
    min_message_rounds = 1
    verifier_first = False


@register_protocol_handler("debate")
class DebateProtocol(NipProtocol):
    """Implementation of the Debate protocol :cite:p:`Irving2018`.

    The protocol consists of two provers and a verifier. The provers debate a question
    with the verifier, who decides which prover is correct. "prover0" attempts to
    convince the verifier of a negative answer, while "prover1" attempts to convince the
    verifier of a positive answer.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    message_channel_names = ["prover0_channel", "prover1_channel"]
    agent_channel_visibility = [
        ("prover0", "prover0_channel"),
        ("prover0", "prover1_channel"),
        ("prover1", "prover0_channel"),
        ("prover1", "prover1_channel"),
        ("verifier", "prover0_channel"),
        ("verifier", "prover1_channel"),
    ]

    def is_agent_active(
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:
        """Specify whether an agent is active in a given round and channel.

        In sequential MNIP with verifier first, the order is:

        - Verifier in both channels
        - First prover (determined by `prover0_first`) in their respective channel
        - Second prover in their respective channel

        In simultaneous MNIP with verifier first, the order is:

        - Verifier in both channels
        - Provers in their respective channels at the same time

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

        if self.hyper_params.debate_protocol.prover0_first:
            first_prover = "prover0"
            second_prover = "prover1"
        else:
            first_prover = "prover1"
            second_prover = "prover0"

        if self.verifier_first:

            # Verifier first, sequential
            if self.hyper_params.debate_protocol.sequential:
                if agent_name == "verifier":
                    return round_id % 3 == 0
                elif agent_name == first_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 1
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 2
                    else:
                        return False

            # Verifier first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 2 == 1 and (
                            channel_name == f"{agent_name}_channel"
                        )
                    else:
                        return False
                elif agent_name == "verifier":
                    return round_id % 2 == 0

        else:

            # Provers first, sequential
            if self.hyper_params.debate_protocol.sequential:
                if agent_name == first_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 0
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 1
                    else:
                        return False
                elif agent_name == "verifier":
                    return round_id % 3 == 2

            # Provers first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 2 == 0
                    else:
                        return False
                elif agent_name == "verifier":
                    return round_id % 2 == 1

    @property
    def max_message_rounds(self) -> int:
        """The maximum number of message rounds."""
        return self.hyper_params.debate_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        """The minimum number of message rounds."""
        return self.hyper_params.debate_protocol.min_message_rounds


@register_protocol_handler("merlin_arthur")
class MerlinArthurProtocol(SingleVerifierProtocolHandler):
    """Implementation of the Merlin-Arthur protocol :cite:p:`Waeldchen2022`.

    The protocol consists of two provers and a verifier. One of the two provers sends a
    message to the verifier, who then makes a decision. Which prover sends the message
    is determined randomly. "prover0" attempts to convince the verifier of a negative
    answer, while "prover1" attempts to convince the verifier of a positive answer.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    message_channel_names = ["main"]
    agent_channel_visibility = [
        ("prover0", "main"),
        ("prover1", "main"),
        ("verifier", "main"),
    ]

    max_message_rounds = 2
    min_message_rounds = 1
    max_verifier_questions = 1

    def get_active_agents_mask_from_rounds_and_seed(
        self, round_id: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask indicating which agents are active in a given round.

        A random one of the two provers goes first, and the verifier goes second.

        Parameters
        ----------
        round_id : Int[Tensor, "..."]
            The round of the protocol.
        seed : Int[Tensor, "..."]
            The per-environment seed.

        Returns
        -------
        active_agents : Bool[Tensor, "... agent channel"]
            A boolean mask indicating which agents are active in the given round.
        """

        # Determine which of the two provers sends the message
        prover1_goes = (seed % 2) == 0
        return rearrange(
            [
                (round_id == 0) & prover1_goes,
                (round_id == 0) & ~prover1_goes,
                round_id == 1,
            ],
            "agent ... -> ... agent 1",
        )

    def can_agent_be_active(
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:
        """Specify whether an agent can be active in a given round.

        The provers can only be active in the first round, and the verifier can only be
        active in the second round.

        Returns
        -------
        can_be_active : bool
            Whether the agent can be active in the given round.
        """

        if agent_name in ["prover0", "prover1"]:
            return round_id == 0
        elif agent_name == "verifier":
            return round_id == 1


@register_protocol_handler("mnip")
class MnipProtocol(NipProtocol):
    """Implementation of the MNIP protocol.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    message_channel_names = ["prover0_channel", "prover1_channel"]
    agent_channel_visibility = [
        ("prover0", "prover0_channel"),
        ("prover1", "prover1_channel"),
        ("verifier", "prover0_channel"),
        ("verifier", "prover1_channel"),
    ]

    def is_agent_active(
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:
        """Specify whether an agent is active in a given round and channel.

        In sequential MNIP with verifier first, the order is:

        - Verifier in both channels
        - First prover (determined by `prover0_first`) in their respective channel
        - Second prover in their respective channel

        In simultaneous MNIP with verifier first, the order is:

        - Verifier in both channels
        - Provers in their respective channels at the same time

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

        if self.hyper_params.mnip_protocol.prover0_first:
            first_prover = "prover0"
            second_prover = "prover1"
        else:
            first_prover = "prover1"
            second_prover = "prover0"

        if self.hyper_params.protocol_common.verifier_first:

            # Verifier first, sequential
            if self.hyper_params.mnip_protocol.sequential:
                if agent_name == "verifier":
                    return round_id % 3 == 0
                elif agent_name == first_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 1
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 2
                    else:
                        return False

            # Verifier first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name == f"{agent_name}_channel":
                        return (
                            round_id % 2 == 1
                            and channel_name == f"{agent_name}_channel"
                        )
                    else:
                        return False
                elif agent_name == "verifier":
                    return round_id % 2 == 0

        else:

            # Provers first, sequential
            if self.hyper_params.mnip_protocol.sequential:
                if agent_name == first_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 0
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 3 == 1
                    else:
                        return False
                elif agent_name == "verifier":
                    return round_id % 3 == 2

            # Provers first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name == f"{agent_name}_channel":
                        return round_id % 2 == 0
                    else:
                        return False
                elif agent_name == "verifier":
                    return round_id % 2 == 1

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        env_td: TensorDictBase | NestedArrayDict,
    ):
        """Compute the rewards for the other agents and add them to the current reward.

        Both provers receive the same reward, which is the 1 if the verifier accepts and
        0 otherwise.

        The `reward` tensor is updated in place, adding in the rewards for the agents
        at the appropriate indices.

        Parameters
        ----------
        verifier_decision_made : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier has made a decision.
        verifier_decision : Int[Tensor, "..."]
            The verifier's decision.
        reward : Float[Tensor, "... agent"]
            The currently computed reward, which should include the reward for the
            verifier.
        env_td : TensorDictBase | NestedArrayDict
            The current observation and state. If a `NestedArrayDict`, it is converted
            to a `TensorDictBase`.
        """

        if self.hyper_params.protocol_common.shared_reward:
            for prover_index in self.prover_indices:
                reward[..., prover_index] = reward[..., self.verifier_index]
        else:
            for prover_index in self.prover_indices:
                reward[..., prover_index] = (
                    verifier_decision_made & (verifier_decision == 1)
                ).float() * self.hyper_params.protocol_common.prover_reward


@register_protocol_handler("solo_verifier")
class SoloVerifierProtocol(DeterministicSingleVerifierProtocolHandler):
    """Implementation of the Solo Verifier protocol.

    The protocol consists of a single verifier, who makes a decision without interacting
    with a prover.

    Note
    ----

    The implementation of NumPy StringDType arrays appears to be buggy with shapes like
    (1, 1, n). For this reason, we set the maximum number of message rounds to 2, but
    always terminate the episode after the first round.

    An example of the bug:

    - https://github.com/numpy/numpy/issues/27737

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    agent_names = ["verifier"]
    message_channel_names = ["main"]
    agent_channel_visibility = [("verifier", "main")]
    min_message_rounds = 1
    max_verifier_questions = 1

    # The maximum number of message rounds is set to 2, but the episode is always
    # terminated after the first round. See the note in the class docstring.
    max_message_rounds = 2

    def is_agent_active(
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:
        """Specify whether an agent is active in a given round and channel.

        The verifier (the only agent) is active in the first round.

        Parameters
        ----------
        agent_name : str
            The name of the agent (always "verifier").
        round_id : int
            The round number.
        channel_name : str
            The channel name (always "main").

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """
        return round_id == 0

    def _get_new_terminated_mask(
        self, round_id: Int[Tensor, "..."], verifier_decision_made: Bool[Tensor, "..."]
    ) -> Bool[Tensor, "..."]:
        """Get a mask indicating whether the episode has been newly terminated.

        "Newly terminated" means that the episode has been terminated this round.

        Since this protocol has only one round, the episode is terminated if the
        verifier has not made a decision.

        Parameters
        ----------
        round_id : Int[Tensor, "..."]
            The round number.
        verifier_decision_made : Bool[Tensor, "..."]
            A mask indicating whether the verifier has made a decision.

        Returns
        -------
        terminated : Bool[Tensor, "..."]
            A mask indicating whether the episode has been newly terminated.
        """
        return ~verifier_decision_made

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
    ):
        pass


@register_protocol_handler("multi_channel_test")
class MultiChannelTestProtocol(DeterministicSingleVerifierProtocolHandler):
    """A protocol for testing multi-channel communication between agents."""

    agent_names = ["prover0", "prover1", "prover2", "verifier"]
    message_channel_names = ["main", "prover0_verifier", "prover_chat"]
    agent_channel_visibility = [
        ("prover0", "main"),
        ("prover1", "main"),
        ("prover2", "main"),
        ("verifier", "main"),
        ("prover0", "prover0_verifier"),
        ("verifier", "prover0_verifier"),
        ("prover0", "prover_chat"),
        ("prover1", "prover_chat"),
        ("prover2", "prover_chat"),
    ]

    max_message_rounds = 8
    min_message_rounds = 2
    max_verifier_questions = 4

    def is_agent_active(self, agent_name: str, round_id: int, channel_name: str):
        """Specify whether an agent is active in a given round and channel."""
        if channel_name == "main":
            if round_id % 3 == 0:
                return agent_name == "prover1"
            elif round_id % 3 == 1:
                return agent_name == "prover2"
            elif round_id % 3 == 2:
                return agent_name == "verifier"
        elif channel_name == "prover0_verifier":
            if round_id % 3 == 1:
                return agent_name == "prover0"
            elif round_id % 3 == 2:
                return agent_name == "verifier"
        elif channel_name == "prover_chat":
            if round_id % 3 == 0:
                return agent_name == "prover0"
            elif round_id % 3 == 1:
                return agent_name == "prover1"
            else:
                return agent_name == "prover2"

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
    ):
        pass
