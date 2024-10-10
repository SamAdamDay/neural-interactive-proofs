"""Zero-knowledge protocol handler.

The zero-knowledge protocol handler extends a base protocol handler to be
zero-knowledge. It does this by creating a child protocol handler instance.
"""

from functools import cached_property

import torch
from torch import Tensor
import torch.nn.functional as F

from tensordict.tensordict import TensorDictBase

from einops import rearrange, repeat, reduce

from jaxtyping import Int, Bool, Float

from pvg.parameters import Parameters, ScenarioType
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.nested_array_dict import NestedArrayDict

from pvg.protocols.base import ProtocolHandler, SingleVerifierProtocolHandler


class ZeroKnowledgeProtocol(ProtocolHandler):
    """Meta-handler for zero-knowledge protocols.

    Takes a base protocol as argument and extends it to be zero-knowledge. It does this
    by creating a child protocol handler instance.

    Introduces a second verifier and a simulator. The simulator tries to mimic the
    interaction between the second verifier and the prover(s), and the second verifier
    tries to prevent this. The prover(s) tries to make sure the simulator can succeed
    (which implies that it is not `leaking` knowledge).

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    base_protocol_cls : type[SingleVerifierProtocolHandler]
        The base protocol to extend to be zero-knowledge. A child instance of this
        protocol will be created.
    """

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        base_protocol_cls: type[SingleVerifierProtocolHandler],
    ):

        super().__init__(params, settings)

        if not base_protocol_cls.can_be_zero_knowledge:
            raise ValueError(
                f"{base_protocol_cls.__name__} cannot be used as a zero-knowledge "
                f"protocol (`can_be_zero_knowledge=False`)."
            )

        self.base_protocol = base_protocol_cls(params, settings)

        if self.base_protocol.verifier_names != ["verifier"]:
            raise ValueError(
                f"ZeroKnowledgeProtocol can only be used with protocols that have a "
                f"single verifier called 'verifier'. Got "
                f"{self.base_protocol.verifier_names}."
            )

        # We rely on the message logits, so we need to make sure they are available
        if params.scenario not in [
            ScenarioType.GRAPH_ISOMORPHISM,
            ScenarioType.IMAGE_CLASSIFICATION,
        ]:
            raise NotImplementedError(
                "Zero-knowledge protocols are only supported for graph isomorphism and "
                "image classification scenarios."
            )

    verifier_names = ["verifier", "adversarial_verifier"]

    @property
    def max_message_rounds(self) -> int:
        return self.base_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.base_protocol.min_message_rounds

    @property
    def max_verifier_turns(self) -> int:
        return self.base_protocol.max_verifier_turns

    @property
    def agent_names(self) -> list[str]:
        """The names of the agents in the protocol.

        The agents are the same as in the base protocol, with the addition of the
        adversarial verifier and the simulator.
        """
        return self.base_protocol.agent_names + ["adversarial_verifier", "simulator"]

    @property
    def verifier_index(self) -> int:
        """The index of the verifier in the agent names."""
        return self.agent_names.index(self.base_protocol.verifier_name)

    @property
    def adversarial_verifier_index(self) -> int:
        """The index of the adversarial verifier in the agent names."""
        return self.agent_names.index("adversarial_verifier")

    @property
    def simulator_index(self) -> int:
        """The index of the simulator in the agent names."""
        return self.agent_names.index("simulator")

    @property
    def standard_channel_names(self) -> list[str]:
        """The names of the standard message channels in the protocol.

        These are the same as the base protocol, with the suffix "_standard" added.
        """
        return [
            f"{channel_name}_standard"
            for channel_name in self.base_protocol.message_channel_names
        ]

    @property
    def adversarial_channel_names(self) -> list[str]:
        """The names of the adversarial message channels in the protocol.

        These are the same as the base protocol, with the suffix "_adversarial" added.
        """
        return [
            f"{channel_name}_adversarial"
            for channel_name in self.base_protocol.message_channel_names
        ]

    @property
    def simulator_channel_names(self) -> list[str]:
        """The names of the simulator message channels in the protocol.

        These are the same as the base protocol, with the suffix "_simulator" added.
        """
        return [
            f"{channel_name}_simulator"
            for channel_name in self.base_protocol.message_channel_names
        ]

    @cached_property
    def message_channel_names(self) -> list[str]:
        """The names of the message channels in the protocol.

        These are (in order):

        - The standard channels for the base protocol with the suffix "_standard"
        - The standard channels for the base protocol with the suffix "_adversarial"
        - The standard channels for the base protocol with the suffix "_simulator"
        """

        return (
            self.standard_channel_names
            + self.adversarial_channel_names
            + self.simulator_channel_names
        )

    @property
    def agent_channel_visibility(self) -> list[tuple[str, str]]:

        agent_channel_visibility = []

        # Clone the existing visibility settings and add a separate channel for the
        # simulator
        for agent_name, channel_name in self.base_protocol.agent_channel_visibility:
            if agent_name == "verifier":
                agent_channel_visibility.append(
                    ("verifier", channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    ("adversarial_verifier", channel_name + "_adversarial")
                )
                agent_channel_visibility.append(
                    ("simulator", channel_name + "_simulator")
                )
            else:
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_adversarial")
                )

        return agent_channel_visibility

    def can_agent_be_active(
        self, agent_name: str, round: int, channel_name: str
    ) -> bool:

        base_channel_name, _, channel_suffix = channel_name.rpartition("_")

        # Simulators are the only agents active in the simulator channels, and are
        # always active
        if agent_name == "simulator":
            return channel_suffix == "simulator"
        if channel_suffix == "simulator":
            return False

        # Verifiers are active in their corresponding channels
        if channel_suffix == "adversarial":
            if agent_name == "verifier":
                return False
            elif agent_name == "adversarial_verifier":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round, base_channel_name
                )
        elif channel_suffix == "standard":
            if agent_name == "adversarial_verifier":
                return False
            elif agent_name == "verifier":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round, base_channel_name
                )

        # Whether the provers are active is determined by the base protocol
        return self.base_protocol.can_agent_be_active(
            agent_name, round, base_channel_name
        )

    def get_active_agents_mask_from_rounds(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:

        num_base_channels = self.base_protocol.num_message_channels

        # Start with the active agents from the base protocol
        active_mask: Bool[Tensor, "... base_agent base_channel"] = (
            self.base_protocol.get_active_agents_mask_from_rounds(round)
        )

        # Copy the mask for the verifier to the adversarial verifier
        active_mask: Bool[Tensor, "... base_agent+1 base_channel"] = torch.cat(
            [active_mask, active_mask[..., [self.verifier_index], :]], dim=-2
        )

        # Duplicate channels for the verifier and adversarial verifier
        active_mask: Bool[Tensor, "... base_agent+1 base_channel*2"] = repeat(
            active_mask, "... agent channel -> ... agent (2 channel)"
        ).clone()

        # Set the verifier to inactive in the adversarial channels and vice versa
        active_mask[..., self.verifier_index, num_base_channels:] = False
        active_mask[..., self.adversarial_verifier_index, :num_base_channels] = False

        # Add the simulator channels, and set all agents to inactive there
        active_mask: Bool[Tensor, "... base_agent+1 base_channel*3"] = torch.cat(
            [active_mask, torch.zeros_like(active_mask[..., :num_base_channels])],
            dim=-1,
        )

        # Add the simulator agent and set it to active in the simulator channels
        active_mask: Bool[Tensor, "... base_agent+2 base_channel*3"] = torch.cat(
            [active_mask, torch.zeros_like(active_mask[..., [0], :])], dim=-2
        )
        active_mask[..., -1, -num_base_channels:] = True

        return active_mask

    def get_verifier_guess_mask_from_rounds(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating whether the verifier can make a decision.

        This is the case only when the base verifier can make a decision. TODO

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        guess_mask : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier can make a decision.
        """
        return self.base_protocol.get_verifier_guess_mask_from_rounds(round)

    def step_interaction_protocol(
        self,
        env_td: TensorDictBase | NestedArrayDict,
    ) -> tuple[
        Bool[Tensor, "..."],
        Bool[Tensor, "... agent"],
        Bool[Tensor, "..."],
        Float[Tensor, "... agent"],
    ]:
        """Take a step in the interaction protocol.

        Computes the done signals and reward. Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase | NestedArrayDict
            The current observation and state. If a `NestedArrayDict`, it is converted
            to a `TensorDictBase`. Has keys:

            - "y" (... 1): The target value.
            - "round" (...): The current round.
            - "done" (...): A boolean mask indicating whether the episode is done.
            - ("agents", "done") (... agent): A boolean mask indicating whether each
                agent is done.
            - "terminated" (...): A boolean mask indicating whether the episode has been
                terminated.
            - ("agents", "decision") (... agent): The decision of each agent.
            - ("agents", "main_message_logits") (... agent channel position logit): The
                main message logits for each agent.
            - ("agents", "decision_logits") (... agent 3): The decision logits for each
                agent.

        Returns
        -------
        shared_done : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode is done because all relevant
            agents have made a decision.
        agent_done : Bool[Tensor, "... agent"]
            A boolean mask indicating whether each agent is done, because they have made
            a decision. This can only be `True` for agents that can make decisions.
        terminated : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode has been terminated because
            the max number of rounds has been reached and the verifier has not guessed.
        reward : Float[Tensor, "... agent"]
            The reward for the agents.
        """

        y: Int[Tensor, "... 1"] = env_td["y"]
        round: Int[Tensor, "..."] = env_td["round"]
        decision: Int[Tensor, "... agent"] = env_td["agents", "decision"]

        if isinstance(env_td, NestedArrayDict):
            y = torch.from_numpy(y)
            round = torch.from_numpy(round)
            decision = torch.from_numpy(decision)

        # Create a copy of the environment tensor dict with the agents restricted to the
        # base protocol agents
        env_td_restricted = env_td.clone(recurse=False)
        env_td_restricted["agents"] = env_td["agents"][
            ..., : self.base_protocol.num_agents
        ]

        # Get the base protocol rewards
        shared_done, agent_done, terminated, reward = (
            self.base_protocol.step_interaction_protocol(env_td_restricted)
        )

        # Get the mask of the batch items where the adversarial verifier can make a
        # guess
        adversarial_verifier_guess_mask = self.get_verifier_guess_mask_from_rounds(
            round
        )

        # Determine if the adversarial verifier has made a decision
        adversarial_verifier_decision_made = self._get_agent_decision_made_mask(
            round=round,
            y=y,
            guess_mask=adversarial_verifier_guess_mask,
            decision=decision[..., self.adversarial_verifier_index],
        )

        # The episode is only done if the adversarial verifier has made a decision
        shared_done = shared_done & adversarial_verifier_decision_made

        # Set the verifier and adversarial verifier dones to their respective values,
        # and set the dones for the other agents to the shared done signal
        verifier_done = agent_done[..., self.verifier_index]
        agent_done = repeat(
            shared_done, "... -> ... agent", agent=len(self.agent_names)
        )
        agent_done[..., self.verifier_index] = verifier_done
        agent_done[..., self.adversarial_verifier_index] = (
            adversarial_verifier_decision_made
        )

        # Expand the reward tensor to include the new agents
        reward: Float[Tensor, "... agent"] = torch.cat(
            [
                reward,
                torch.zeros_like(reward[..., [0]]),
                torch.zeros_like(reward[..., [0]]),
            ],
            dim=-1,
        )

        # Get the main message logits and decision logits. Possibly we should also
        # include the additional linear message logits? #TODO-LH
        main_message_logits: Float[Tensor, "... agent channel position logit"] = env_td[
            "agents", "main_message_logits"
        ]
        decision_logits: Float[Tensor, "... agent 3"] = env_td[
            "agents", "decision_logits"
        ]
        message_size = main_message_logits.shape[-2]
        d_main_message_logits = main_message_logits.shape[-1]

        # Get necessary masks and indices
        active_agents_mask: Float[Tensor, "... agent channel"] = (
            self.get_active_agents_mask_from_rounds(round)
        )
        adversarial_verifier_guess_mask: Float[Tensor, "..."] = (
            self.get_verifier_guess_mask_from_rounds(round)
        )
        prover_indices = [self.agent_names.index(name) for name in self.prover_names]
        adversarial_channel_indices = [
            self.message_channel_names.index(name)
            for name in self.adversarial_channel_names
        ]
        simulator_channel_indices = [
            self.message_channel_names.index(name)
            for name in self.simulator_channel_names
        ]

        # Get the message logits for the adversarial channels. Note that this currently
        # assumes that in any given channel, only one agent is active. This assumption
        # can be removed by using utils.math.logit_or_n instead of simply summing the
        # logits across the agent dimension.
        adversarial_channel_logits: Float[
            Tensor, "... agent adversarial_channel position logit"
        ] = main_message_logits[..., adversarial_channel_indices, :, :]
        adversarial_channel_mask = repeat(
            active_agents_mask[..., adversarial_channel_indices],
            "... agent adversarial_channel -> ... agent adversarial_channel position logit",
            position=message_size,
            logit=d_main_message_logits,
        )
        logits_per_adversarial_channel = reduce(
            adversarial_channel_logits * adversarial_channel_mask,
            "... agent adversarial_channel position logit "
            "-> ... (adversarial_channel position logit)",
            reduction="sum",
        )

        # Get the message logits for the simulator channels
        logits_per_simulator_channel = main_message_logits[
            ..., self.simulator_index, simulator_channel_indices, :, :
        ]
        logits_per_simulator_channel = rearrange(
            logits_per_simulator_channel,
            "... simulator_channel position logit "
            "-> ... (simulator_channel position logit)",
        )

        # Simulator reward, which is the cosine similarity between the
        # adversarial_verifier and simulator logits, multiplied by a coefficient. At the
        # moment we sum the similarities for the message logits and decision logits
        # (because decisions are not necessarily allowed in every round, unlike
        # messages, but we may want to do something else here). Moreover, this does not
        # currently take into account the possibility that the adversarial verifier has
        # guessed (and therefore ended the trajectory) before the non-adversarial
        # verifier, which means that for some timesteps we will be comparing logits that
        # don't correspond to any action that makes any difference. #TODO-LH
        simulator_reward: Float[Tensor, "..."] = (
            F.cosine_similarity(
                logits_per_adversarial_channel, logits_per_simulator_channel, dim=-1
            )
            * self.params.zk_protocol.simulator_reward_coefficient
        )

        # Add the cosine similarity of the adversarial verifier and simulator decision
        # logits where the adversarial verifier can make a decision
        adversarial_verifier_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.adversarial_verifier_index, :
        ]
        simulator_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.simulator_index, :
        ]
        decision_simulator_reward: Float[Tensor, "..."] = (
            F.cosine_similarity(
                adversarial_verifier_decision_logits, simulator_decision_logits, dim=-1
            )
            * self.params.zk_protocol.simulator_reward_coefficient
        )
        simulator_reward += decision_simulator_reward * adversarial_verifier_guess_mask

        # Add the simulator reward to the reward tensor
        reward[..., self.simulator_index] = simulator_reward

        # The adversarial verifier reward is the negative of the simulator reward
        reward[..., self.adversarial_verifier_index] = -simulator_reward

        # Prover rewards
        expanded_simulator_reward = repeat(
            simulator_reward,
            "... -> ... agent",
            agent=len(self.base_protocol.prover_names),
        )
        reward[..., prover_indices] += (
            self.prover_zk_loss_coefficient * expanded_simulator_reward
        )

        return shared_done, agent_done, terminated, reward

    # Eventually this should dynamically update the coefficient to allow for
    # lexicographic optimisation, as in https://www.ijcai.org/proceedings/2022/0476.pdf
    @property
    def prover_zk_loss_coefficient(self) -> float:
        """The coefficient of the simulator reward in the prover reward.

        The prover rewards get a bonus for making the simulator succeed, which is
        controlled by this coefficient.
        """

        return self.params.zk_protocol.aux_prover_reward_coefficient
