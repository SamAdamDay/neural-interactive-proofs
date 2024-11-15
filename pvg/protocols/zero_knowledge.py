"""Zero-knowledge protocol handler.

The zero-knowledge protocol handler extends a base protocol handler to be
zero-knowledge. It does this by creating a child protocol handler instance.
"""

from functools import cached_property

import torch
from torch import Tensor, as_tensor
import torch.nn.functional as F

from tensordict.tensordict import TensorDictBase

from einops import rearrange, repeat, reduce

from jaxtyping import Int, Bool, Float

from pvg.parameters import HyperParameters, ScenarioType
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.nested_array_dict import NestedArrayDict

from pvg.protocols.protocol_base import ProtocolHandler, SingleVerifierProtocolHandler


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

        self.is_zero_knowledge = True

        self.base_protocol = base_protocol_cls(hyper_params, settings)

        if self.base_protocol.verifier_names != ["verifier"]:
            raise ValueError(
                f"ZeroKnowledgeProtocol can only be used with protocols that have a "
                f"single verifier called 'verifier'. Got "
                f"{self.base_protocol.verifier_names}."
            )

        # We rely on the message logits, so we need to make sure they are available
        if hyper_params.scenario not in [
            ScenarioType.GRAPH_ISOMORPHISM,
            ScenarioType.IMAGE_CLASSIFICATION,
        ]:
            raise NotImplementedError(
                "Zero-knowledge protocols are only supported for graph isomorphism and "
                "image classification scenarios."
            )
        
        self.use_multiple_simulators = hyper_params.zk_protocol.use_multiple_simulators

        # The cached property doesn't get inherited properly for some reason (and similarly if we use ), so we need to redefine it
        self.agent_first_active_round = {}
        for round_id in range(100):
            for agent_name in set(self.agent_names) - set(
                self.agent_first_active_round.keys() - set(self.simulator_names)
            ):
                if self.can_agent_be_active_any_channel(agent_name, round_id):
                    self.agent_first_active_round[agent_name] = round_id
            if len(self.agent_first_active_round) == len(self.agent_names) - len(self.simulator_names):
                break
        else:
            raise ValueError(
                "Could not determine the first active round for all agents."
            )

    # verifier_names = ["verifier", "adversarial_verifier"]

    # # The cached property doesn't get inherited, so we need to redefine it
    # agents_first_active_rounds = {}
    # for round_id in range(100):
    #     for agent_name in set(self.agent_names) - set(
    #         self.agents_first_active_rounds.keys()
    #     ):
    #         if self.can_agent_be_active_any_channel(agent_name, round_id):
    #             self.agents_first_active_rounds[agent_name] = round_id
    #     if len(self.agents_first_active_rounds) == len(self.agent_names):
    #         break
    # else:
    #     raise ValueError(
    #         "Could not determine the first active round for all agents."
    #     )

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
        if self.use_multiple_simulators:
            simulator_names = [f"simulator_{a}" for a in self.base_protocol.agent_names]
        else:
            simulator_names = ["simulator"]
        return self.base_protocol.agent_names + ["adversarial_verifier"] + simulator_names
    
    @property
    def base_agent_indices(self) -> list[int]:
        """The indices of the agents in the base protocol.
        """
        return [self.agent_names.index(name) for name in self.base_protocol.agent_names]

    @property
    def simulator_names(self) -> list[str]:
        """The names of the simulator(s) in the protocol."""
        return [agent_name for agent_name in self.agent_names if "simulator" in agent_name]

    @property
    def verifier_index(self) -> int:
        """The index of the verifier in the agent names."""
        return self.agent_names.index(self.base_protocol.verifier_name)
    
    @property
    def prover_indices(self) -> int:
        """The indices of the provers in the agent names."""
        return [self.agent_names.index(name) for name in self.base_protocol.prover_names]

    @property
    def adversarial_verifier_index(self) -> int:
        """The index of the adversarial verifier in the agent names."""
        return self.agent_names.index("adversarial_verifier")

    @property
    def simulated_verifier_index(self) -> int:
        """The index of the simulated verifier in the agent names."""
        return self.agent_names.index("simulator_verifier") if self.use_multiple_simulators else self.agent_names.index("simulator")

    @property
    def adversarial_indices(self) -> list[int]:
        """The indices of the agents in the adversarial channels."""
        adversarial_names = ["adversarial_verifier"] + [name for name in self.base_protocol.agent_names if name != "verifier"] 
        return [self.agent_names.index(name) for name in adversarial_names]

    @property
    def simulator_indices(self) -> list[int]:
        """The indices of the simulator(s) in the agent names."""
        if not self.use_multiple_simulators:
            return [self.agent_names.index("simulator")]
        else:
            simulator_indices = [self.agent_names.index(f"simulator_{name}") for name in self.base_protocol.agent_names if name != "verifier"]
            return [self.agent_names.index("simulator_verifier")] + simulator_indices

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

    # @property
    # def simulator_channel_names(self) -> list[str]:
    #     """The names of the simulator message channels in the protocol.

    #     These are the same as the base protocol, with the suffix "_simulator" added.
    #     """
    #     return [
    #         f"{channel_name}_simulator"
    #         for channel_name in self.base_protocol.message_channel_names
    #     ]

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
            # + self.simulator_channel_names
        )

    @property
    def agent_channel_visibility(self) -> list[tuple[str, str]]:

        agent_channel_visibility = []

        # Clone the existing visibility settings and add separate channels for the
        # simulator
        for agent_name, channel_name in self.base_protocol.agent_channel_visibility:
            if agent_name == "verifier":
                agent_channel_visibility.append(
                    ("verifier", channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    ("adversarial_verifier", channel_name + "_adversarial")
                )
            else:
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_adversarial")
                )
            if self.use_multiple_simulators:
                agent_channel_visibility.append(
                    # (f"simulator_{agent_name}", channel_name + "_simulator")
                    (f"simulator_{agent_name}", channel_name + "_adversarial")
                )
        if not self.use_multiple_simulators:
            for channel_name in self.base_protocol.message_channel_names:
                # agent_channel_visibility.append("simulator", channel_name + "_simulator")
                agent_channel_visibility.append("simulator", channel_name + "_adversarial")

        return agent_channel_visibility

    # # For some reason this is equal to None if we try to inherit directly from ProtocolHandler, so we need to redefine it
    # @cached_property
    # def agent_first_active_round(self) -> dict[str, int]:
    #     """The first round in which each agent is or can be active.

    #     For non-deterministic protocols, this is the first round in which the agent has
    #     some probability of being active.

    #     Returns
    #     -------
    #     agents_first_active_rounds : dict[str, int]
    #         The first round in which each agent is active
    #     """

    #     agents_first_active_rounds = {}
    #     for round_id in range(100):
    #         for agent_name in set(self.agent_names) - set(
    #             agents_first_active_rounds.keys()
    #         ):
    #             if self.can_agent_be_active_any_channel(agent_name, round_id):
    #                 agents_first_active_rounds[agent_name] = round_id
    #         if len(agents_first_active_rounds) == len(self.agent_names):
    #             break
    #     else:
    #         raise ValueError(
    #             "Could not determine the first active round for all agents."
    #         )

    def can_agent_be_active(
        self, agent_name: str, round_id: int, channel_name: str
    ) -> bool:

        base_channel_name, _, channel_suffix = channel_name.rpartition("_")

        # The verifier is active in the standard channels when the base protocol
        # verifier is active. The adversarial verifier and simulator are not active in
        # the standard channels, but the provers are
        if channel_suffix == "standard":
            if "adversarial" in agent_name or "simulator" in agent_name:
                return False
            elif agent_name == "verifier":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round_id, base_channel_name
                )

        # Adversarial verifiers are active in the adversarial channels when the base
        # protocol verifier is active. The verifier and simulator are not active in the
        # adversarial channels, but the provers are
        elif channel_suffix == "adversarial":
            if agent_name == "verifier" or "simulator" in agent_name:
                return False
            elif agent_name == "adversarial_verifier":
                return self.base_protocol.can_agent_be_active(
                    "verifier", round_id, base_channel_name
                )

        # When there are multiple simulators, they copy the visibility of the corresponding base agent. Otherwise the (single) simulator is always active. No other agents are active in the simulator channel(s).
        # elif channel_suffix == "simulator":
        #     if self.use_multiple_simulators:
        #         if "simulator_" in agent_name:
        #             return self.base_protocol.can_agent_be_active(
        #                 agent_name[10:], round_id, base_channel_name
        #             )
        #         else:
        #             return False
        #     else:
        #         return agent_name == "simulator"

        # Whether the provers are active is determined by the base protocol
        return self.base_protocol.can_agent_be_active(
            agent_name, round_id, base_channel_name
        )

    def get_active_agents_mask_from_rounds_and_seed(
        self, round_id: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:

        num_base_channels = self.base_protocol.num_message_channels

        # Start with the active agents from the base protocol
        active_mask: Bool[Tensor, "... base_agent base_channel"] = (
            self.base_protocol.get_active_agents_mask_from_rounds_and_seed(
                round_id, seed
            )
        )

        # Copy the mask for the verifier to the adversarial verifier
        active_mask: Bool[Tensor, "... base_agent+1 base_channel"] = torch.cat(
            [
                active_mask,
                active_mask[..., [self.verifier_index], :],
            ],
            dim=-2,
        ).clone()

        # Copy the mask for relevant agent(s) to the the simulator
        if self.use_multiple_simulators:
            new_num_agents = "(base_agent-1)*2"
            agent_range = range(self.base_protocol.num_agents)
        else:
            new_num_agents = "base_agent+2"
            agent_range = range(1)
        active_mask: Bool[Tensor, f"... {new_num_agents} base_channel"] = torch.cat(
                [
                    active_mask,
                    # active_mask[..., agent_range, :],
                    torch.zeros_like(active_mask[..., agent_range, :]),
                ],
                dim=-2,
            ).clone()
        
        # Duplicate channels for the verifier, adversarial verifier and simulator(s)
        # active_mask: Bool[Tensor, f"... {new_num_agents} base_channel*3"] = repeat(
        #     active_mask, "... agent channel -> ... agent (3 channel)"
        # ).clone()
        active_mask: Bool[Tensor, f"... {new_num_agents} base_channel*2"] = repeat(
            active_mask, "... agent channel -> ... agent (2 channel)"
        ).clone()
        
        # Set the verifier, adversarial verifier and simulator(s) to inactive in the
        # channels that are not theirs
        active_mask[..., self.verifier_index, num_base_channels:] = False
        active_mask[..., self.adversarial_verifier_index, :num_base_channels] = False
        # active_mask[..., self.adversarial_verifier_index, 2 * num_base_channels :] = (
        #     False
        # )
        # active_mask[..., self.simulator_indices, : 2 * num_base_channels] = False
        active_mask[..., self.simulator_indices, :] = False

        # Set the base agents to inactive in the simulator channels
        # active_mask[..., : self.base_protocol.num_agents, 2 * num_base_channels :] = (
        #     False
        # )

        # If there is only one simulator, make sure they are always active
        # if not self.use_multiple_simulators:
        #     active_mask[..., self.simulator_indices, 2 * num_base_channels :] = True

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

    def get_trajectory_log_probs(self, tensordict: TensorDictBase) -> Float[Tensor, "..."]:

        round_id = tensordict.get("round")
        seed = tensordict.get("seed")
        main_message_logits = tensordict.get(("agents","main_message_logits"))
        decision_logits = tensordict.get(("agents","decision_logits"))
        indices = {"decision": tensordict.get(("agents","decision")).to(torch.int64),
                    "main_message": tensordict.get(("agents","node_selected")).to(torch.int64),
        }
        l_p = lambda x : F.log_softmax(x, dim=-1)
        num_base_channels = self.base_protocol.num_message_channels

        # Get the main message logits. Group the channels by kind (standard,
        # adversarial, simulator) and flatten the position dimension
        grouped_message_logits = rearrange(
            main_message_logits,
            "... agent (group base_channel) position logit "
            "-> ... agent group base_channel (position logit)",
            group=2,
        )
        indices["main_message"] = rearrange(
            indices["main_message"],
            "... agent (group base_channel) node"
            "-> ... agent group base_channel node",
        group=2,
        )

        # Get the message logits for the adversarial channels
        adversarial_channel_logits: Float[Tensor, "... base_channel flat_logit"] = (
            grouped_message_logits[..., 1, :, :]
        )
        indices["main_message"] = indices["main_message"][...,1,:,:]

        # Get the active agent masks for the adversarial channel
        adversarial_channel_mask: Bool[
            Tensor, "... base_channel"
        ] = self.get_active_agents_mask_from_rounds_and_seed(round_id, seed)[
            ..., num_base_channels : 2 * num_base_channels,
        ]
        restricted_adversarial_channel_mask = adversarial_channel_mask[...,self.adversarial_indices,:].unsqueeze(-1)

        # Set the logits to zero where the adversarial verifier and simulator are not
        # active
        adversarial_log_probs = l_p(adversarial_channel_logits[...,self.adversarial_indices,:,:]) * restricted_adversarial_channel_mask
        simulator_log_probs = l_p(adversarial_channel_logits[...,self.simulator_indices,:,:]) * restricted_adversarial_channel_mask

        # Get the log probabilities of the main message logits
        dims_to_sum = tuple(i for i in range(adversarial_log_probs.dim()) if i != 0)
        adv_log_prob = torch.gather(adversarial_log_probs, -1, indices["main_message"][...,self.adversarial_indices,:,:]).sum(dim=dims_to_sum)
        sim_log_prob = torch.gather(simulator_log_probs, -1, indices["main_message"][...,self.adversarial_indices,:,:]).sum(dim=dims_to_sum)

        # Get the mask for which batch items the adversarial verifier can make a guess
        verifier_guess_mask: Float[Tensor, "..."] = (
            self.get_verifier_guess_mask_from_rounds_and_seed(round_id, seed)
        ).unsqueeze(-1)

        # Get the decision logits for the adversarial verifier and the relevant simulator
        adversarial_verifier_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.adversarial_verifier_index, :
        ]
        simulator_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.simulated_verifier_index, :
        ]

        # Set the decision logits to zero where the adversarial verifier cannot make a
        # decision
        adversarial_verifier_decision_log_probs = l_p(adversarial_verifier_decision_logits) * verifier_guess_mask
        simulator_decision_log_probs = l_p(simulator_decision_logits) * verifier_guess_mask

        # Get the log probabilities of the decision logits
        dims_to_sum = tuple(i for i in range(adversarial_verifier_decision_log_probs.dim()) if i != 0)
        adv_log_prob += torch.gather(adversarial_verifier_decision_log_probs, -1, indices["decision"][...,self.adversarial_verifier_index].unsqueeze(dim=-1)).sum(dim=dims_to_sum)
        sim_log_prob += torch.gather(simulator_decision_log_probs, -1, indices["decision"][...,self.adversarial_verifier_index].unsqueeze(dim=-1)).sum(dim=dims_to_sum)

        return torch.stack((adv_log_prob, sim_log_prob))

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
        trajectory_log_probs: Float[Tensor, "..., 2"]
            The trajectory log_probs for the adversarial and simulator interactions, respectively. This is only calculated when the protocol is zero-knowledge.
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
        trajectory_log_probs: Float[Tensor, "..."] = as_tensor(env_td["trajectory_log_probs"])

        # Create a copy of the environment tensor dict with the agents restricted to the
        # base protocol agents
        env_td_restricted = env_td.clone(recurse=False)
        env_td_restricted["agents"] = env_td["agents"][
            ..., : self.base_protocol.num_agents
        ]

        # Get the base protocol rewards
        shared_done, agent_done, terminated, reward, _ = (
            self.base_protocol.step_interaction_protocol(env_td_restricted)
        )

        trajectory_log_probs += self.get_trajectory_log_probs(env_td)

        # Get the mask of the batch items where the adversarial verifier can make a
        # guess
        adversarial_verifier_guess_mask = (
            self.get_verifier_guess_mask_from_rounds_and_seed(round_id, seed)
        )

        # Determine if the adversarial verifier has made a decision and is done
        adversarial_verifier_decision_made = self._get_agent_decision_made_mask( # LH-TODO do we need to define this separately?
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
                *[torch.zeros_like(reward[..., [0]]) for _ in range(self.num_agents - self.base_protocol.num_agents)],
            ],
            dim=-1,
        )

        simulator_reward: Float[Tensor, "... agent"] = self.get_simulator_reward(
            round_id=round_id,
            seed=seed,
            main_message_logits=main_message_logits,
            decision_logits=decision_logits,
        )

        # Add the simulator rewards to the reward tensor
        reward[..., self.simulator_indices] = simulator_reward

        total_simulator_reward = simulator_reward.sum(dim=-1) #TODO

        if self.use_mixed_sl_and_rl:

            # Here, we don't actually add the full ZK rewards here because we can used supervised learning instead of RL for the other parts, and including them here effectively double counts them
            reward[..., self.adversarial_verifier_index] = -(total_simulator_reward - reward[..., self.simulated_verifier_index])
            for prover_name in self.base_protocol.prover_names:
                reward[..., self.agent_names.index(prover_name)] += (
                    self.prover_zk_loss_coefficient * (total_simulator_reward - reward[..., self.agent_names.index(f"simulator_{prover_name}")]) / self.simulator_reward_coefficient
                )

        else:

            # Add the simulator rewards to the adversarial verifier's loss
            reward[..., self.adversarial_verifier_index] = -total_simulator_reward
            # Add the simulator rewards to the provers' rewards
            for prover_name in self.base_protocol.prover_names:
                reward[..., self.agent_names.index(prover_name)] += (
                    self.prover_zk_loss_coefficient * total_simulator_reward / self.simulator_reward_coefficient
                )
            
        # expanded_simulator_reward = repeat(
        #     total_simulator_reward,
        #     "... -> ... agent",
        #     agent=len(self.base_protocol.prover_names),
        # )
        # prover_indices = [self.agent_names.index(name) for name in self.base_protocol.prover_names if "simulator" not in name]
        # reward[..., prover_indices] += (
        #     self.prover_zk_loss_coefficient * expanded_simulator_reward / self.hyper_params.zk_protocol.simulator_reward_coefficient
        # )

        return shared_done, agent_done, terminated, reward, trajectory_log_probs

    def get_simulator_reward(
        self,
        round_id: Int[Tensor, "..."],
        seed: Int[Tensor, "..."],
        main_message_logits: Float[Tensor, "... agent channel position logit"],
        decision_logits: Float[Tensor, "... agent 3"],
        distance_function = lambda adv, sim : torch.cosine_similarity(adv, sim, dim=-1),
    ) -> Float[Tensor, "..."]:
        """Get the simulator reward(s).

        The simulator reward is the cosine similarity between the simulator logits and the other agents' logits in the adversarial verifier channels, multiplied by a coefficient. At the moment we sum the
        similarities for the message logits and decision logits (because decisions are
        not necessarily allowed in every round, unlike messages, but we may want to do
        something else here). Moreover, this does not currently take into account the
        possibility that the adversarial verifier has guessed (and therefore ended the
        trajectory) before the non-adversarial verifier, which means that for some
        timesteps we will be comparing logits that don't correspond to any action that
        makes any difference. #TODO

        #TODO: Maybe add linear message logits (low priority since linear message logits
        are primarily for debugging)

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
            group=2,
        )

        # Get the message logits for the adversarial channels
        adversarial_channel_logits: Float[Tensor, "... base_channel flat_logit"] = (
            grouped_message_logits[..., 1, :, :]
        )
        # simulator_channel_logits: Float[Tensor, "... base_channel flat_logit"] = (
        #     grouped_message_logits[..., 2, :, :]
        # )

        # Get the active agent masks for the adversarial channel
        adversarial_channel_mask: Bool[
            Tensor, "... base_channel"
        ] = self.get_active_agents_mask_from_rounds_and_seed(round_id, seed)[
            ..., num_base_channels : 2 * num_base_channels,
        ]
        # simulator_channel_mask: Bool[
        #     Tensor, "... base_channel"
        # ] = self.get_active_agents_mask_from_rounds_and_seed(round_id, seed)[
        #     ..., - num_base_channels :,
        # ]
        restricted_adversarial_channel_mask = adversarial_channel_mask[...,self.adversarial_indices,:].unsqueeze(-1)

        # Set the logits to zero where the adversarial verifier and simulator are not
        # active
        adversarial_logits = adversarial_channel_logits[...,self.adversarial_indices,:,:] * restricted_adversarial_channel_mask
        simulator_logits = adversarial_channel_logits[...,self.simulator_indices,:,:] * restricted_adversarial_channel_mask

        # # Extract the logits for the corresponding agents
        # adversarial_channel_logits[..., self.adversarial_indices,:,:]
        # adversarial_channel_logits[..., self.simulator_indices,:,:]

        # Flatten the base channel dimension
        # adversarial_channel_logits = rearrange(
        #     adversarial_channel_logits,
        #     "... base_channel flat_logit -> ... (base_channel flat_logit)",
        # )
        # simulator_channel_logits = rearrange(
        #     simulator_channel_logits,
        #     "... base_channel flat_logit -> ... (base_channel flat_logit)",
        # )

        # Compute the cosine similarity between the adversarial verifier and simulator
        # active message logits

        simulator_reward = distance_function(adversarial_logits, simulator_logits).sum(dim=-1)

        # Get the mask for which batch items the adversarial verifier can make a guess
        verifier_guess_mask: Float[Tensor, "..."] = (
            self.get_verifier_guess_mask_from_rounds_and_seed(round_id, seed)
        ).unsqueeze(-1)

        # Get the decision logits for the adversarial verifier and the relevant simulator
        adversarial_verifier_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.adversarial_verifier_index, :
        ]
        simulator_decision_logits: Float[Tensor, "... 3"] = decision_logits[
            ..., self.simulated_verifier_index, :
        ]

        # Set the decision logits to zero where the adversarial verifier cannot make a
        # decision
        adversarial_verifier_decision_logits = adversarial_verifier_decision_logits * verifier_guess_mask
        simulator_decision_logits = simulator_decision_logits * verifier_guess_mask

        # Add the cosine similarity (or other difference function) of the adversarial verifier and simulator decision
        # logits where the adversarial verifier can make a decision to the relevant simulator's loss
        additional_reward = [torch.zeros_like(simulator_reward[:, 0]) for _ in range(simulator_reward.shape[-1])]

        additional_reward[0] = distance_function(adversarial_verifier_decision_logits, simulator_decision_logits)

        simulator_reward = simulator_reward + torch.stack(additional_reward,dim=-1)

        # Scale the reward by the coefficient
        simulator_reward = simulator_reward * self.simulator_reward_coefficient

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

    @property
    def simulator_reward_coefficient(self) -> float:
        """The coefficient of the simulator reward.

        Note
        ----
        Eventually this should dynamically update the coefficient to allow for
        lexicographic optimisation, as in
        https://www.ijcai.org/proceedings/2022/0476.pdf
        """

        return self.hyper_params.zk_protocol.simulator_reward_coefficient

    @property
    def use_mixed_sl_and_rl(self) -> bool:
        """Whether to use supervised learning for some terms in the adversarial verifier and provers' losses and RL for others. Defaults to False.
        """

        return self.hyper_params.zk_protocol.use_mixed_sl_and_rl