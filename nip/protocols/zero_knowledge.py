"""Zero-knowledge protocol handler.

The zero-knowledge protocol handler extends a base protocol handler to be
zero-knowledge. It does this by creating a child protocol handler instance.
"""

from functools import cached_property

import torch
from torch import Tensor, as_tensor

from tensordict.tensordict import TensorDictBase

from einops import rearrange, repeat

from jaxtyping import Int, Bool, Float

from nip.parameters import HyperParameters
from nip.experiment_settings import ExperimentSettings
from nip.utils.nested_array_dict import NestedArrayDict

from nip.protocols.protocol_base import ProtocolHandler, SingleVerifierProtocolHandler


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
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    base_protocol_cls : type[SingleVerifierProtocolHandler]
        The base protocol to extend to be zero-knowledge. A child instance of this
        protocol will be created.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        base_protocol_cls: type[SingleVerifierProtocolHandler],
    ):

        super().__init__(hyper_params, settings)

        if not base_protocol_cls.can_be_zero_knowledge:
            raise ValueError(
                f"{base_protocol_cls.__name__} cannot be used as a zero-knowledge "
                f"protocol (`can_be_zero_knowledge=False`)."
            )

        self.base_protocol = base_protocol_cls(hyper_params, settings)

        if self.base_protocol.verifier_names != ["verifier"]:
            raise ValueError(
                f"ZeroKnowledgeProtocol can only be used with protocols that have a "
                f"single verifier called 'verifier'. Got "
                f"{self.base_protocol.verifier_names}."
            )

        # We rely on the message logits, so we need to make sure they are available
        if hyper_params.scenario not in [
            "graph_isomorphism",
            "image_classification",
        ]:
            raise NotImplementedError(
                "Zero-knowledge protocols are only supported for graph isomorphism and "
                "image classification scenarios."
            )

    verifier_names = ["verifier", "adversarial_verifier"]

    @property
    def max_message_rounds(self) -> int:
        """The maximum number of message rounds in the protocol."""
        return self.base_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        """The minimum number of message rounds in the protocol."""
        return self.base_protocol.min_message_rounds

    @property
    def max_verifier_questions(self) -> int:
        """The maximum number of questions the verifier can make to each prover."""
        return self.base_protocol.max_verifier_questions

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

        Note
        ----
        The order of the channels is important, as the other methods in the protocol
        handler rely on this order. If the order is changed, the other methods should be
        updated accordingly.
        """

        return (
            self.standard_channel_names
            + self.adversarial_channel_names
            + self.simulator_channel_names
        )

    @property
    def agent_channel_visibility(self) -> list[tuple[str, str]]:
        """The visibility settings for the agents in the protocol."""

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
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:
        """Check if an agent can be active in a given round and channel.

        1. The verifier is active in the standard channels when the base protocol
        verifier is active. The adversarial verifier and simulator are not active in the
        standard channels, but the provers are.
        2. Adversarial verifiers are active in the adversarial channels when the base
        protocol verifier is active. The verifier and simulator are not active in the
        adversarial channels, but the provers are.
        3. The simulator is active in the simulator channels when the base protocol
        verifier is active. No other agents are active in the simulator channels.
        4. Whether the provers are active is determined by the base protocol.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        round_id : int
            The round of the protocol.
        channel_name : str
            The name of the channel.

        Returns
        -------
        can_be_active : bool
            Whether the agent can be active in the given round and channel.
        """

        base_channel_name, _, channel_suffix = channel_name.rpartition("_")

        # The verifier is active in the standard channels when the base protocol
        # verifier is active. The adversarial verifier and simulator are not active in
        # the standard channels, but the provers are
        if channel_suffix == "standard":
            if agent_name in ["adversarial_verifier", "simulator"]:
                return False
            elif agent_name == "verifier":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round_id, base_channel_name
                )

        # Adversarial verifiers are active in the adversarial channels when the base
        # protocol verifier is active. The verifier and simulator are not active in the
        # adversarial channels, but the provers are
        elif channel_suffix == "adversarial":
            if agent_name in ["verifier", "simulator"]:
                return False
            elif agent_name == "adversarial_verifier":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round_id, base_channel_name
                )

        # The simulator is active in the simulator channels when the base protocol
        # verifier is active. No other agents are active in the simulator channels
        elif channel_suffix == "simulator":
            if agent_name == "simulator":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round_id, base_channel_name
                )
            else:
                return False

        # Whether the provers are active is determined by the base protocol
        return self.base_protocol.can_agent_be_active(
            agent_name, round_id, base_channel_name
        )

    def get_active_agents_mask_from_rounds_and_seed(
        self, round_id: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask indicating which agents are active in which channels.

        The adversarial verifier is active in the adversarial channels, the simulator is
        active in the simulator channels, and the provers are active in all channels. The
        verifier is active in the standard channels.

        Parameters
        ----------
        round_id : Int[Tensor, "..."]
            The round of the protocol.
        seed : Int[Tensor, "..."]
            The per-environment seed.

        Returns
        -------
        active_mask : Bool[Tensor, "... agent channel"]
            A boolean mask indicating which agents are active in which channels.
        """

        num_base_channels = self.base_protocol.num_message_channels

        # Start with the active agents from the base protocol
        active_mask: Bool[Tensor, "... base_agent base_channel"] = (
            self.base_protocol.get_active_agents_mask_from_rounds_and_seed(
                round_id, seed
            )
        )

        # Copy the mask for the verifier to the adversarial verifier and simulator
        active_mask: Bool[Tensor, "... base_agent+2 base_channel"] = torch.cat(
            [
                active_mask,
                active_mask[..., [self.verifier_index, self.verifier_index], :],
            ],
            dim=-2,
        ).clone()

        # Duplicate channels for the verifier, adversarial verifier and simulator
        active_mask: Bool[Tensor, "... base_agent+2 base_channel*3"] = repeat(
            active_mask, "... agent channel -> ... agent (3 channel)"
        ).clone()

        # Set the verifier, adversarial verifier and simulator to inactive in the
        # channels that are not theirs
        active_mask[..., self.verifier_index, num_base_channels:] = False
        active_mask[..., self.adversarial_verifier_index, :num_base_channels] = False
        active_mask[..., self.adversarial_verifier_index, 2 * num_base_channels :] = (
            False
        )
        active_mask[..., self.simulator_index, : 2 * num_base_channels] = False

        # Set the base agents to inactive in the simulator channels
        active_mask[..., : self.base_protocol.num_agents, 2 * num_base_channels :] = (
            False
        )

        # # Add the simulator channels, and set all agents to inactive there
        # active_mask: Bool[Tensor, "... base_agent+1 base_channel*3"] = torch.cat(
        #     [active_mask, torch.zeros_like(active_mask[..., :num_base_channels])],
        #     dim=-1,
        # )

        # # Add the simulator agent and set it to active in the simulator channels
        # active_mask: Bool[Tensor, "... base_agent+2 base_channel*3"] = torch.cat(
        #     [active_mask, torch.zeros_like(active_mask[..., [0], :])], dim=-2
        # )
        # active_mask[..., -1, -num_base_channels:] = True

        return active_mask

    def get_verifier_guess_mask_from_rounds_and_seed(
        self, round_id: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating whether the verifier can make a decision.

        This is the case only when the base verifier can make a decision. TODO

        Parameters
        ----------
        round_id : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        guess_mask : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier can make a decision.
        """
        return self.base_protocol.get_verifier_guess_mask_from_rounds_and_seed(
            round_id, seed
        )

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

        y: Int[Tensor, "... 1"] = as_tensor(env_td["y"])
        round_id: Int[Tensor, "..."] = as_tensor(env_td["round"])
        seed: Int[Tensor, "..."] = as_tensor(env_td["seed"])
        decision: Int[Tensor, "... agent"] = as_tensor(env_td["agents", "decision"])
        main_message_logits: Float[Tensor, "... agent channel position logit"] = (
            as_tensor(env_td["agents", "main_message_logits"])
        )
        decision_logits: Float[Tensor, "... agent 3"] = as_tensor(
            env_td["agents", "decision_logits"]
        )

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
        adversarial_verifier_guess_mask = (
            self.get_verifier_guess_mask_from_rounds_and_seed(round_id, seed)
        )

        # Determine if the adversarial verifier has made a decision and is done
        adversarial_verifier_decision_made = self._get_agent_decision_made_mask(
            round_id=round_id,
            y=y,
            guess_mask=adversarial_verifier_guess_mask,
            decision=decision[..., self.adversarial_verifier_index],
        )
        adversarial_verifier_done = (
            adversarial_verifier_decision_made
            | env_td["agents", "done"][..., self.adversarial_verifier_index]
        )

        # The episode done if both the verifier and  adversarial verifier have made a
        # decision
        shared_done = agent_done[..., self.verifier_index] & adversarial_verifier_done
        shared_done = shared_done | env_td["done"]

        # Set the verifier and adversarial verifier dones to their respective values,
        # and set the dones for the other agents to the shared done signal
        verifier_done = agent_done[..., self.verifier_index]
        agent_done = repeat(
            shared_done, "... -> ... agent", agent=len(self.agent_names)
        ).clone()
        agent_done[..., self.verifier_index] = verifier_done
        agent_done[..., self.adversarial_verifier_index] = adversarial_verifier_done

        # Expand the reward tensor to include the new agents
        reward: Float[Tensor, "... agent"] = torch.cat(
            [
                reward,
                torch.zeros_like(reward[..., [0]]),
                torch.zeros_like(reward[..., [0]]),
            ],
            dim=-1,
        )

        simulator_reward: Float[Tensor, "..."] = self._get_simulator_reward(
            round_id=round_id,
            seed=seed,
            main_message_logits=main_message_logits,
            decision_logits=decision_logits,
        )

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
        prover_indices = [self.agent_names.index(name) for name in self.prover_names]
        reward[..., prover_indices] += (
            self.prover_zk_loss_coefficient * expanded_simulator_reward
        )

        return shared_done, agent_done, terminated, reward

    def reward_mid_point_estimate(self, agent_name: str) -> float:
        """Get an estimate of the expected reward if all agents play randomly.

        This is used to compute the mid-point of the reward range for the agent.

        For example, if the agent gets reward -1 for a wrong guess and 1 for a correct
        guess, the mid-point estimate could be 0.

        For the zero-knowledge protocol, for the base agents we use the estimate from
        the base protocol. We set the mid-point estimate for the simulator to 0, because
        its reward is a cosine similarity. We set the mid-point estimate for the
        adversarial verifier to 0, because its reward is the negative of the simulator
        reward.

        Parameters
        ----------
        agent_name : str
            The name of the agent to get the reward mid-point for.

        Returns
        -------
        reward_mid_point : float
            The expected reward for the agent if all agents play randomly.
        """

        if agent_name in self.base_protocol.agent_names:
            return self.base_protocol.reward_mid_point_estimate(agent_name)
        elif agent_name in ("adversarial_verifier", "simulator"):
            return 0.0
        else:
            raise ValueError(f"Unknown agent {agent_name}.")

    def _get_simulator_reward(
        self,
        round_id: Int[Tensor, "..."],
        seed: Int[Tensor, "..."],
        main_message_logits: Float[Tensor, "... agent channel position logit"],
        decision_logits: Float[Tensor, "... agent 3"],
    ) -> Float[Tensor, "..."]:
        """Get the simulator reward.

        The simulator reward is the cosine similarity between the adversarial verifier
        and simulator logits, multiplied by a coefficient. At the moment we sum the
        similarities for the message logits and decision logits (because decisions are
        not necessarily allowed in every round, unlike messages, but we may want to do
        something else here). Moreover, this does not currently take into account the
        possibility that the adversarial verifier has guessed (and therefore ended the
        trajectory) before the non-adversarial verifier, which means that for some
        timesteps we will be comparing logits that don't correspond to any action that
        makes any difference. #TODO

        Note
        ----
        It is assumed that the adversarial verifier is active in the channels
        corresponding to those where the simulator is active, and vice versa. If this is
        changed, the method should be updated accordingly.

        Parameters
        ----------
        round_id : Int[Tensor, "..."]
            The round of the protocol.
        seed : Int[Tensor, "..."]
            The per-environment seed.
        main_message_logits : Float[Tensor, "... agent channel position logit"]
            The logits for the messages sent by the agents.
        decision_logits : Float[Tensor, "... agent 3"]
            The logits for the decisions made by the agents.

        Returns
        -------
        simulator_reward : Float[Tensor, "..."]
            The simulator reward.
        """

        num_base_channels = self.base_protocol.num_message_channels

        # Get the main message logits. Group the channels by kind (standard,
        # adversarial, simulator) and flatten the position dimension
        grouped_message_logits = rearrange(
            main_message_logits,
            "... agent (group base_channel) position logit "
            "-> ... agent group base_channel (position logit)",
            group=3,
        )

        # Get the message logits for the adversarial verifier and simulator
        adversarial_verifier_logits: Float[Tensor, "... base_channel flat_logit"] = (
            grouped_message_logits[..., self.adversarial_verifier_index, 1, :, :]
        )
        simulator_logits: Float[Tensor, "... base_channel flat_logit"] = (
            grouped_message_logits[..., self.simulator_index, 2, :, :]
        )

        # Get the mask for which batch items and base channels the adversarial verifier
        # and simulator are active in
        channel_active_mask: Bool[
            Tensor, "... base_channel"
        ] = self.get_active_agents_mask_from_rounds_and_seed(round_id, seed)[
            ...,
            self.adversarial_verifier_index,
            num_base_channels : 2 * num_base_channels,
        ]

        # Set the logits to zero where the adversarial verifier and simulator are not
        # active
        adversarial_verifier_logits[~channel_active_mask] = 0.0
        simulator_logits[~channel_active_mask] = 0.0

        # Flatten the base channel dimension
        adversarial_verifier_logits = rearrange(
            adversarial_verifier_logits,
            "... base_channel flat_logit -> ... (base_channel flat_logit)",
        )
        simulator_logits = rearrange(
            simulator_logits,
            "... base_channel flat_logit -> ... (base_channel flat_logit)",
        )

        # Compute the cosine similarity between the adversarial verifier and simulator
        # active message logits
        simulator_reward: Float[Tensor, "..."] = torch.cosine_similarity(
            adversarial_verifier_logits, simulator_logits, dim=-1
        )

        # Get the mask for which batch items the adversarial verifier can make a guess
        adversarial_verifier_guess_mask: Float[Tensor, "..."] = (
            self.get_verifier_guess_mask_from_rounds_and_seed(round_id, seed)
        )

        # Get the decision logits for the adversarial verifier and simulator
        adversarial_verifier_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.adversarial_verifier_index, :
        ]
        simulator_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.simulator_index, :
        ]

        # Set the decision logits to zero where the adversarial verifier cannot make a
        # decision
        adversarial_verifier_decision_logits[~adversarial_verifier_guess_mask] = 0.0
        simulator_decision_logits[~adversarial_verifier_guess_mask] = 0.0

        # Add the cosine similarity of the adversarial verifier and simulator decision
        # logits where the adversarial verifier can make a decision
        simulator_reward += torch.cosine_similarity(
            adversarial_verifier_decision_logits, simulator_decision_logits, dim=-1
        )

        # Scale the reward by the coefficient
        simulator_reward *= self.hyper_params.zk_protocol.simulator_reward_coefficient

        return simulator_reward

    @property
    def prover_zk_loss_coefficient(self) -> float:
        """The coefficient of the simulator reward in the prover reward.

        The prover rewards get a bonus for making the simulator succeed, which is
        controlled by this coefficient.

        Note
        ----
        Eventually this should dynamically update the coefficient to allow for
        lexicographic optimisation, as in
        https://www.ijcai.org/proceedings/2022/0476.pdf
        """

        return self.hyper_params.zk_protocol.aux_prover_reward_coefficient
