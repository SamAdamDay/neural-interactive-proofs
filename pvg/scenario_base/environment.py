"""Base class for the RL environment."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from jaxtyping import Float, Int, Bool

from pvg.scenario_base import DataLoader, Dataset
from pvg.parameters import Parameters, InteractionProtocolType
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import VariableDataCycler


class Environment(EnvBase, ABC):
    """The base class for all Prover-Verifier RL environments.

    To implement a new environment, subclass this class and implement the following
    attribute and methods:

    - `_get_observation_spec`: The specification of the agent observations.
    - `_get_action_spec`: The specification of the agent actions.
    - `_get_state_spec` (optional): The specification of the states space.
    - `_get_reward_spec` (optional): The specification of the agent rewards.
    - `_get_done_spec` (optional): The specification of the agent done signals.
    - `_step`: Perform a step in the environment.
    - `_masked_reset`: Reset the environment for a subset of the episodes.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : TorchDevice
        The device on which to run the environment.
    dataset : Dataset
        The dataset for the environment.
    """

    _int_dtype: torch.dtype = (torch.int,)

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        dataset: Dataset,
        train: bool = True,
    ):
        super().__init__(device=settings.device)
        self.params = params
        self.settings = settings
        self.train = train

        self.agent_names = list(params.agents.keys())

        # Create a random number generator
        self.rng = torch.Generator()

        # Load the dataset
        self.dataset = dataset
        self.data_cycler: Optional[VariableDataCycler] = None

        # The number of environments is the number of episodes we can fit in a batch
        self.num_envs = params.ppo.frames_per_batch // params.max_message_rounds
        self.batch_size = (self.num_envs,)

        # Create environment specs
        self.observation_spec = self._get_observation_spec()
        self.action_spec = self._get_action_spec()
        self.state_spec = self._get_state_spec()
        self.reward_spec = self._get_reward_spec()
        self.done_spec = self._get_done_spec()

    @abstractmethod
    def _get_observation_spec(self) -> TensorSpec:
        """Get the specification of the agent observations.

        The observation space has the following elements:

        - `round`: The current round of the interaction.
        - `decision_restriction`: The restriction on what the verifier can decide.
            * 0: The verifier can decide anything.
            * 1: The verifier can only decide to continue interacting.
            * 2: The verifier can only make a guess.
        - `x`: The message history.
        - `message`: The next message.


        Subclasses should call this method and add at least:

        - `x`: The message history.
        - `message`: The next message.

        Returns
        -------
        observation_spec : TensorSpec
            The observation specification.
        """
        return CompositeSpec(
            round=DiscreteTensorSpec(
                self.params.max_message_rounds,
                shape=(self.num_envs,),
                dtype=torch.long,
            ),
            decision_restriction=DiscreteTensorSpec(
                3,
                shape=(self.num_envs,),
                dtype=self._int_dtype,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    @abstractmethod
    def _get_action_spec(self) -> TensorSpec:
        """Get the specification of the agent actions.

        Subclasses should call this method and add any additional action spaces.

        Returns
        -------
        action_spec : TensorSpec
            The action specification.
        """
        return CompositeSpec(
            agents=CompositeSpec(
                decision=DiscreteTensorSpec(
                    3,
                    shape=(self.num_envs, 2),
                    dtype=self._int_dtype,
                ),
                shape=(self.num_envs,),
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _get_state_spec(self) -> TensorSpec:
        """Get the specification of the states space.

        Defaults to the true label.

        Returns
        -------
        state_spec : TensorSpec
            The state specification.
        """
        return CompositeSpec(
            y=BinaryDiscreteTensorSpec(
                1,
                shape=(self.num_envs, 1),
                dtype=torch.long,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _get_reward_spec(self) -> TensorSpec:
        """Get the specification of the agent rewards.

        Returns
        -------
        reward_spec : TensorSpec
            The reward specification.
        """
        return CompositeSpec(
            agents=CompositeSpec(
                reward=UnboundedContinuousTensorSpec(shape=(self.num_envs, 2)),
                shape=(self.num_envs,),
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _get_done_spec(self) -> TensorSpec:
        """Get the specification of the agent done signals.

        Returns
        -------
        done_spec : TensorSpec
            The done specification.
        """
        return CompositeSpec(
            done=BinaryDiscreteTensorSpec(
                self.num_envs, shape=(self.num_envs,), dtype=torch.bool
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _step(self, env_td: TensorDictBase) -> TensorDictBase:
        """Perform a step in the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.

        Returns
        -------
        next_td : TensorDictBase
            The next observation, state, reward, and done signal.
        """

        # Create an initial next tensordict, which will be updated with the message
        # history, next message, done signal and reward
        next_td = TensorDict(
            dict(
                round=env_td["round"] + 1,
            ),
            batch_size=self.batch_size,
        )

        # The observations are passed through unchanged
        for key in self.observation_spec.keys():
            if key not in ["x", "message", "round", "decision_restriction"]:
                next_td[key] = env_td[key]

        # Compute the message history and next message
        next_td = self._compute_x_and_message(env_td, next_td)

        # Compute the done signal and reward
        next_td = self._step_interaction_protocol(env_td, next_td)

        return next_td

    @abstractmethod
    def _compute_x_and_message(
        self,
        env_td: TensorDictBase,
        next_td: TensorDictBase,
    ) -> TensorDictBase:
        """Compute the message history and next message.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.
        next_td : TensorDictBase
            The 'next' tensordict, to be updated with the message history and next
            message.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """

    def _step_interaction_protocol(
        self,
        env_td: TensorDictBase,
        next_td: TensorDictBase,
    ) -> TensorDictBase:
        """Take a step in the interaction protocol.

        Computes the done signal, reward and next decision restriction.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.
        next_td : TensorDictBase
            The 'next' tensordict, to be updated with the done signal and reward.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """
        if self.params.interaction_protocol == InteractionProtocolType.PVG:
            return self._step_pvg_protocol(env_td, next_td)
        else:
            raise NotImplementedError(
                f"Interaction protocol {self.params.interaction_protocol} not "
                "implemented."
            )

    def _step_pvg_protocol(
        self,
        env_td: TensorDictBase,
        next_td: TensorDictBase,
    ) -> TensorDictBase:
        """Take a step in the PVG interaction protocol.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.
        next_td : TensorDictBase
            The 'next' tensordict, to be updated with the done signal and reward.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """

        protocol_params = self.params.pvg_protocol

        y: Int[Tensor, "batch 1"] = env_td["y"]
        round: Int[Tensor, "batch"] = env_td["round"]
        decision: Int[Tensor, "batch agent"] = env_td["agents", "decision"]
        done: Bool[Tensor, "batch"] = env_td["done"]

        # Compute index of the agent whose turn it is.
        agent_index: Int[Tensor, "batch"] = round % len(self.agent_names)

        # If the verifier has made a guess, compute the reward and terminate the episode
        verifier_agent_num = self.agent_names.index("verifier")
        verifier_decision_made = (agent_index == verifier_agent_num) & (
            decision[:, verifier_agent_num] != 2
        )
        done = done | verifier_decision_made
        reward: dict[str, int] = dict()
        reward["verifier"] = torch.zeros_like(done, dtype=torch.float)
        reward["verifier"][
            verifier_decision_made & (decision[:, verifier_agent_num] == y.squeeze())
        ] = protocol_params.verifier_reward
        reward["verifier"][
            verifier_decision_made & (decision[:, verifier_agent_num] != y.squeeze())
        ] = protocol_params.verifier_incorrect_penalty
        reward["prover"] = (
            verifier_decision_made & (decision[:, verifier_agent_num] == 1)
        ).float()
        reward["prover"] = reward["prover"] * protocol_params.prover_reward

        # If we reach the end of the episode and the verifier has not made a guess,
        # terminate it with a negative reward for the verifier
        done = done | (round >= self.params.max_message_rounds - 1)
        reward["verifier"][
            (round >= self.params.max_message_rounds - 1) & ~verifier_decision_made
        ] = protocol_params.verifier_terminated_penalty

        # If the verifier has not made a guess and it's their turn, given them a small
        # reward for continuing
        reward["verifier"][
            (agent_index == verifier_agent_num) & ~done
        ] = protocol_params.verifier_no_guess_reward

        # Stack the rewards for the two agents
        reward = torch.stack([reward[name] for name in self.agent_names], dim=-1)

        # Put the done signal and reward into the next tensordict
        next_td["done"] = done
        next_td["agents"] = TensorDict(
            dict(
                reward=reward,
            ),
            batch_size=self.batch_size,
        )
        next_td["decision_restriction"] = torch.zeros_like(done, dtype=self._int_dtype)

        return next_td

    def _reset(self, env_td: Optional[TensorDictBase] = None) -> TensorDictBase:
        """(Partially) reset the environment.

        For each episode which is done, takes a new sample from the dataset and resets
        the episode.

        Parameters
        ----------
        env_td : Optional[TensorDictBase]
            The current observation, state and done signal.

        Returns
        -------
        env_td : TensorDictBase
            The reset environment tensordict.
        """

        # If no tensordict is given, we're starting afresh
        if env_td is None:
            observation_zeros = self.observation_spec.zero()
            state_zeros = self.state_spec.zero()
            done_zeros = self.done_spec.zero()
            env_td = observation_zeros.update(state_zeros).update(done_zeros)
            new_mask = torch.ones(
                *self.batch_size, dtype=torch.bool, device=self.device
            )

        else:
            new_mask = env_td["done"]
            env_td = env_td.clone()

        # If we don't have a data cycler yet, create one
        if self.data_cycler is None:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.num_envs,
                shuffle=True,
                generator=self.rng,
            )
            self.data_cycler = VariableDataCycler(dataloader)

        # Sample a new batch of data for the episodes that are done
        batch = self.data_cycler.get_batch(new_mask.sum().item()).to(self.device)

        # Reset the episodes that are done
        env_td = self._masked_reset(env_td, new_mask, batch)

        return env_td

    @abstractmethod
    def _masked_reset(
        self, env_td: TensorDictBase, mask: torch.Tensor, data_batch: TensorDict
    ) -> TensorDictBase:
        """Reset the environment for a subset of the episodes.

        Takes a new sample from the dataset and inserts it into the given episodes. Also
        resets the other elements of the episodes.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation, state and done signal.
        mask : torch.Tensor
            A boolean mask of the episodes to reset.
        data_batch : TensorDict
            The data batch to insert into the episodes.

        Returns
        -------
        env_td : TensorDictBase
            The reset environment tensordict.
        """
        pass

    def _set_seed(self, seed: int | None):
        self.rng = torch.manual_seed(seed)
