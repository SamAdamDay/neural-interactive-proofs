"""Implementations of RL objectives, extending those of TorchRL."""

from dataclasses import dataclass, make_dataclass, field, fields
from typing import Iterable, Optional, Iterator
import contextlib
from abc import ABC, abstractmethod
import warnings

import torch
from torch import Tensor
from torch.nn import Parameter

from torchrl.objectives import (
    PPOLoss,
    ClipPPOLoss,
    KLPENPPOLoss,
    ReinforceLoss,
    LossModule,
)

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.utils import NestedKey

from nip.parameters import SpgVariantType, LrFactors
from nip.scenario_base.agents import Agent
from nip.utils.maths import (
    dict_dot_product,
    inverse_hessian_vector_product,
    compute_sos_update,
)
from nip.utils.torch import flatten_batch_dims
from nip.utils.distributions import CompositeCategoricalDistribution
from nip.utils.tensordict import get_key_batch_size
from nip.utils.data import dict_update_add


class Objective(LossModule, ABC):
    """Base class for all RL objectives.

    Extends the LossModule class from TorchRL to allow multiple actions keys and
    normalise advantages.

    The implementation is a bit of a hack. We change the _AcceptedKeys class dynamically
    to allow for multiple action keys.

    See `torchrl.objectives.LossModule` for more details
    """

    action_keys: Iterable[NestedKey] = ("action",)

    @dataclass
    class _BaseAcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using
        '.set_keys(key_name=key_value)' and their default values

        Attributes
        ----------
            advantage (NestedKey): The input tensordict key where the advantage is
            expected.
                Will be used for the underlying value estimator. Defaults to
                ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state
            value is expected.
                Will be used for the underlying value estimator Defaults to
                ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is
            expected.
                Will be used for the underlying value estimator. Defaults to
                ``"state_value"``.
            sample_log_prob (NestedKey): The input tensordict key where the
               sample log probability is expected.  Defaults to ``"sample_log_prob"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to
                ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value
                estimator. Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying
                value estimator. Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey = "sample_log_prob"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    @dataclass
    class _AcceptedKeys(_BaseAcceptedKeys):
        pass

    def set_keys(self, **kwargs):
        """Set the keys of the input TensorDict that are used by this loss.

        The keyword argument 'action' is treated specially. This should be an iterable
        of action keys. These are not validated against the set of accepted keys for
        this class. Instead, each is added to the set of accepted keys.

        All other keyword arguments should match `self._AcceptedKeys`.

        Parameters
        ----------
        **kwargs
            The keyword arguments to set.
        """
        if "action" in kwargs:
            action_keys = kwargs.pop("action")
            if not isinstance(action_keys, Iterable):
                raise TypeError(f"Action must be an iterable, but got {action_keys}.")

            # Make a dict whose keys are the attribute names for the keys (to use as
            # attributes for the _AcceptedKeys class)
            named_action_keys = {}
            for key in action_keys:
                if isinstance(key, str):
                    named_action_keys[key] = key
                elif isinstance(key, tuple):
                    named_action_keys["_".join(key)] = key
                else:
                    raise TypeError(
                        f"Action keys must be strings or tuples of strings, but got "
                        f"key {key}"
                    )

            # Add the new accepted keys class to self. We can't use the `bases` keyword
            # argument because of the way torchrl checks membership of the class (using
            # the `__dict__` attribute).
            dataclass_fields = []
            for base_field in fields(self._BaseAcceptedKeys):
                dataclass_fields.append(
                    (
                        base_field.name,
                        base_field.type,
                        field(default=base_field.default),
                    )
                )
            for name, key in named_action_keys.items():
                dataclass_fields.append((name, NestedKey, field(default=key)))
            _AcceptedKeys = make_dataclass(
                "_AcceptedKeys", dataclass_fields, bases=(self._BaseAcceptedKeys,)
            )

            self._AcceptedKeys = _AcceptedKeys

            self.action_keys = tuple(named_action_keys.values())

            # Add the new accepted keys to the kwargs
            kwargs.update(named_action_keys)

        super().set_keys(**kwargs)

    def _set_in_keys(self):
        keys = [
            *self.action_keys,
            self.tensor_keys.sample_log_prob,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
        ]
        if self.critic_network is not None:
            keys.extend(self.critic_network.in_keys)
        self._in_keys = list(set(keys))

    # Modified to output both log probabilities and log_weights
    def _log_weight(
        self, sample: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """Compute the log weight for the given TensorDict sample.

        Parameters
        ----------
        sample : TensorDictBase
            The sample TensorDict.

        Returns
        -------
        log_prob : torch.Tensor
            The log probabilities of the sample
        log_weight : torch.Tensor
            The log weight of the sample
        dist : torch.distributions.Distribution
            The distribution used to compute the log weight.

        """
        # current log_prob of actions
        actions = {key: sample.get(key) for key in self.action_keys}
        actions_batch_size = None
        for key, action in actions.items():
            if action.requires_grad:
                raise RuntimeError(f"tensordict stored {key} requires grad.")
            batch_size = get_key_batch_size(sample, key)
            if actions_batch_size is None:
                actions_batch_size = batch_size
            elif actions_batch_size != batch_size:
                raise RuntimeError(
                    f"Not all action keys have the same batch size. Key {key!r} has "
                    f"batch size {batch_size} but expected {actions_batch_size}."
                )

        action_tensordict = TensorDict(actions, batch_size=actions_batch_size)

        with (
            self.actor_network_params.to_module(self.actor_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            dist = self.actor_network.get_dist(sample)

        if not isinstance(dist, CompositeCategoricalDistribution):
            raise RuntimeError(
                f"Actor must return a CompositeCategoricalDistribution to work with "
                f"{type(self).__name__}, but got {dist}."
            )

        log_prob = dist.log_prob(action_tensordict).get(
            self.tensor_keys.sample_log_prob
        )
        prev_log_prob = sample.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensordict prev_log_prob requires grad.")

        log_weight = log_prob - prev_log_prob  # .unsqueeze(-1)

        return log_prob, log_weight, dist

    def _get_advantage(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Get the advantage for a tensordict, normalising it if required.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        advantage : torch.Tensor
            The normalised advantage.
        """

        num_batch_dims = len(tensordict.batch_size)

        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)

        # Normalise advantages per agent
        advantage_flat = flatten_batch_dims(advantage, num_batch_dims)
        if self.normalize_advantage and advantage_flat.shape[0] > 1:
            loc = advantage_flat.mean(dim=0)
            scale = advantage_flat.std(dim=0).clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        return advantage

    @abstractmethod
    def backward(self, loss_vals: TensorDictBase):
        """Perform the backward pass for the loss.

        Parameters
        ----------
        loss_vals : TensorDictBase
            The loss values.
        """


class PPOLossImproved(Objective, PPOLoss, ABC):
    """Base PPO loss class which allows multiple actions keys and normalises advantages.

    See `torchrl.objectives.PPOLoss` for more details
    """

    def _set_entropy_and_critic_losses(
        self,
        tensordict: TensorDictBase,
        td_out: TensorDictBase,
        dist: CompositeCategoricalDistribution,
    ):
        """Set the entropy and critic losses in the output TensorDict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.
        td_out : TensorDictBase
            The output TensorDict, which will be modified in place.
        dist : CompositeCategoricalDistribution
            The distribution used to compute the log weight.
        """

        num_batch_dims = len(tensordict.batch_size)

        if self.entropy_bonus:
            entropy = dist.entropy(batch_size=tensordict.get("agents").batch_size)
            entropy_flat = flatten_batch_dims(entropy, num_batch_dims)
            td_out.set(
                "entropy",
                entropy_flat.mean(dim=0).sum().detach(),
            )
            loss_entropy_per_agent = (-self.entropy_coef * entropy_flat).mean(dim=0)
            td_out.set("loss_entropy", loss_entropy_per_agent.sum())
            td_out.set(("agents", "loss_entropy"), loss_entropy_per_agent)
        if self.critic_coef:
            loss_critic_per_agent = flatten_batch_dims(
                self._loss_critic(tensordict), num_batch_dims
            ).mean(dim=0)
            td_out.set("loss_critic", loss_critic_per_agent.sum())
            td_out.set(("agents", "loss_critic"), loss_critic_per_agent)

    def backward(self, loss_vals: TensorDictBase):
        """Perform the backward pass for the loss.

        Parameters
        ----------
        loss_vals : TensorDictBase
            The loss values.
        """
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        loss_value.backward()

    def _loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Get the critic loss without the clip fraction.

        TorchRL's `loss_critic` method returns a tuple with the critic loss and the
        clip fraction. This method returns only the critic loss.
        """
        return self.loss_critic(tensordict)[0]


class ClipPPOLossImproved(PPOLossImproved, ClipPPOLoss):
    """Clipped PPO loss which allows multiple actions keys and normalises advantages.

    See `torchrl.objectives.ClipPPOLoss` for more details.
    """

    def _set_ess(self, num_batch_dims: int, td_out: TensorDictBase, log_weight: Tensor):
        """Set the ESS in the output TensorDict, for logging.

        Parameters
        ----------
        num_batch_dims : int
            The number of batch dimensions.
        td_out : TensorDictBase
            The output TensorDict, which will be modified in place.
        log_weight : Tensor
            The log weights.
        """

        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same
            # source. Here we sample according to different, unrelated trajectories,
            # which is not standard. Still it can give a idea of the dispersion of the
            # weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        td_out.set(
            "ESS", flatten_batch_dims(ess, num_batch_dims).mean(dim=0).sum() / batch
        )

    # We modify the loss function to normalise per agent
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the PPO algorithm with clipping.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        td_out : TensorDictBase
            The output TensorDict containing the losses.
        """

        tensordict = tensordict.clone(False)

        num_batch_dims = len(tensordict.batch_size)

        # Compute the advantage
        advantage = self._get_advantage(tensordict)

        # Compute the log weights
        _, log_weight, dist = self._log_weight(tensordict)

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = log_weight_clip.exp() * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]

        # Compute the KL divergence for logging
        kl = -log_weight.mean(0)

        loss_objective_per_agent = -flatten_batch_dims(gain, num_batch_dims).mean(dim=0)
        loss_objective = loss_objective_per_agent.sum()

        td_out = TensorDict(
            {
                "loss_objective": loss_objective,
                "kl_divergence": kl.detach().mean(),
                "agents": {
                    "loss_objective": loss_objective_per_agent,
                },
            },
            [],
        )

        self._set_entropy_and_critic_losses(tensordict, td_out, dist)
        self._set_ess(num_batch_dims, td_out, log_weight)

        return td_out


class KLPENPPOLossImproved(PPOLossImproved, KLPENPPOLoss):
    """KL penalty PPO loss which allows multiple actions keys and normalises advantages.

    See `torchrl.objectives.KLPENPPOLoss` for more details
    """

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Compute the loss for the PPO algorithm with a KL penalty.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        td_out : TensorDict
            The output TensorDict containing the losses.
        """

        tensordict = tensordict.clone(False)

        # Compute the advantage
        advantage = self._get_advantage(tensordict)

        # Compute the log weights
        _, log_weight, dist = self._log_weight(tensordict)
        neg_loss = log_weight.exp() * advantage

        # Compute KL penalty
        kl = -log_weight.mean(0)
        if self.beta.shape == torch.Size([]):
            self.beta = self.beta * torch.ones_like(kl)
        kl_penalty = self.beta * kl

        # Update KL penalty terms
        for i in range(len(kl)):
            if kl[i] > self.dtarg * 1.5:
                self.beta[i].data *= self.increment
            elif kl[i] < self.dtarg / 1.5:
                self.beta[i].data *= self.decrement

        loss_objective_per_agent = -neg_loss.mean(dim=0) + kl_penalty
        loss_objective = loss_objective_per_agent.sum()

        td_out = TensorDict(
            {
                "loss_objective": loss_objective,
                "kl_divergence": kl.detach().mean(),
                "agents": {
                    "loss_objective": loss_objective_per_agent,
                },
            },
            [],
        )

        self._set_entropy_and_critic_losses(tensordict, td_out, dist)

        return td_out


class SpgLoss(ClipPPOLossImproved):
    """Loss for Stackelberg Policy Gradient :cite:p:`Huang2022` and several variants.

    In contrast to other objectives, the `forward` method returns the gains per agent
    and the sum of the log probabilities separately. These must be combined later to
    compute the true loss. This is because we need to compute the gradients of these
    separately.

    The following variants are supported:

    - SPG: Standard Stackelberg Policy Gradient.
    - PSPG: SPG with the clipped PPO loss.
    - LOLA: The Learning with Opponent-Learning Awareness algorithm
      :cite:p:`Foerster2018`.
    - POLA: LOLA with the clipped PPO loss.
    - SOS: The Stable Opponent Shaping algorithm :cite:p:`Letcher2019`.
    - PSOS: SOS with the clipped PPO loss.
    """

    @property
    def stackelberg_sequence_flat(self) -> Iterator[str]:
        """A flattened version of the Stackelberg sequence.

        Yields
        ------
        agent_name : str
            The name of the agent in the Stackelberg sequence.
        """

        for group in self.stackelberg_sequence:
            for agent_name in group:
                yield agent_name

    def __init__(
        self,
        actor: TensorDictModule,
        critic: TensorDictModule,
        variant: SpgVariantType,
        stackelberg_sequence: list[tuple[str, ...]],
        agent_names: list[str],
        agents: dict[str, Agent],
        ihvp_arguments: dict,
        additional_lola_term: bool,
        sos_scaling_factor: float,
        sos_threshold_factor: float,
        agent_lr_factors: dict[str, Optional[LrFactors | dict]],
        lr: float,
        clip_epsilon: float,
        entropy_coef: float,
        normalize_advantage: bool,
        loss_critic_type: str,
        clip_value: bool | float | None,
        device: torch.device,
        functional: bool = True,
    ):

        super().__init__(
            actor=actor,
            critic=critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            normalize_advantage=normalize_advantage,
            functional=functional,
            loss_critic_type=loss_critic_type,
            clip_value=clip_value,
        )
        self.variant = variant
        self.stackelberg_sequence = stackelberg_sequence
        self.ihvp_arguments = ihvp_arguments
        self.additional_lola_term = additional_lola_term
        self.sos_scaling_factor = sos_scaling_factor
        self.sos_threshold_factor = sos_threshold_factor
        self.agent_lr_factors = agent_lr_factors
        self.lr = lr
        self.agent_names = agent_names
        self.agents = agents
        self.device = device

        # Compute the groups immediately before and after each agent, and also the set
        # of all agents before or after each agent
        self.immediate_followers, self.all_followers = self._get_followers()
        self.immediate_leaders = {
            agent_name: tuple(
                follower
                for follower in self.agent_names
                if agent_name in self.immediate_followers[follower]
            )
            for agent_name in self.agent_names
        }
        self.all_leaders = {
            agent_name: tuple(
                follower
                for follower in self.agent_names
                if agent_name in self.all_followers[follower]
            )
            for agent_name in self.agent_names
        }

        # Get the actor parameters and names for each agent
        self.actor_params = self._get_actor_params()

        self.register_buffer(
            "clip_epsilon", torch.tensor(clip_epsilon, device=self.device)
        )

    # Define the (variants of the) SPG loss @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the Stackelberg Policy Gradient algorithm.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        td_out : TensorDictBase
            The output TensorDict containing the losses.
        """

        tensordict = tensordict.clone(False)
        num_batch_dims = len(tensordict.batch_size)
        advantage = self._get_advantage(tensordict)
        log_prob, log_weight, dist = self._log_weight(tensordict)

        # TODO we don't take a mean across samples (because the score multiplicands
        # have already been averaged across samples), which I believe is correct in
        # theory, but might need changed in practice
        log_prob_sum = log_prob.sum()

        gains = {}

        # Use vanilla A2C losses for SPG and LOLA and SOS
        if self.variant == "spg" or self.variant == "lola" or self.variant == "sos":
            probs = log_prob.exp()
            for agent_id, agent_name in enumerate(self.agent_names):
                gains[agent_name] = flatten_batch_dims(
                    (probs * advantage[..., agent_id].unsqueeze(dim=-1)), num_batch_dims
                ).mean(dim=0)

        # Otherwise use the clipped PPO loss for PSPG and POLA and PSOS
        elif self.variant == "pspg" or self.variant == "pola" or self.variant == "psos":
            for agent_id, agent_name in enumerate(self.agent_names):
                gain1 = log_weight.exp() * advantage[..., agent_id].unsqueeze(dim=-1)
                log_weight_clip = log_weight.clamp(*self._clip_bounds)
                gain2 = log_weight_clip.exp() * advantage[..., agent_id].unsqueeze(
                    dim=-1
                )
                gains[agent_name] = flatten_batch_dims(
                    torch.stack([gain1, gain2], -1).min(dim=-1)[0], num_batch_dims
                ).mean(dim=0)

            # log_prob_sum = log_weight.sum()
            # TODO maybe this should be log_weight.sum(), also note that we don't take a
            # mean across samples because the score multiplicands have already been
            # averaged across samples

        loss_objective_per_agent = torch.stack(
            [-gains[agent_name] for agent_name in self.agent_names], dim=-1
        )

        # Return losses per agent and set of followers
        td_out = TensorDict(
            {
                "sum_log_probs": log_prob_sum,
                "agents": {
                    "loss_objective": loss_objective_per_agent,
                },
            },
            [],
        )

        self._set_entropy_and_critic_losses(tensordict, td_out, dist)
        self._set_ess(num_batch_dims, td_out, log_weight)

        return td_out

    def backward(self, loss_vals: TensorDictBase):
        """Compute and assign the gradients of the loss for each agent.

        Parameters
        ----------
        loss_vals : TensorDictBase
            The loss values.
        """

        # Compute scores first
        loss_vals["sum_log_probs"].backward(retain_graph=True)
        scores = self._get_and_zero_all_grads()

        # Then compute objective gradients for all agents
        objective_loss_grads: dict[tuple[str, str], dict[str, Tensor]] = {}
        for leader_id, leader_name in enumerate(self.agent_names):
            loss_vals.get(("agents", "loss_objective"))[..., leader_id].sum().backward(
                retain_graph=True
            )
            grads = self._get_and_zero_all_grads()
            for follower_name, follower_grads in grads.items():
                objective_loss_grads[leader_name, follower_name] = follower_grads

        # TODO This is not very memory-efficient but it does prevent the gradients of
        # the parameters getting messed up between calculating JVPs and assigning
        # gradients
        if self.variant == "spg" or self.variant == "pspg":
            ihvps = self._compute_ihvps(loss_vals)

        # The gradient of each parameter with respect to the corresponding agent's loss.
        # Named $\xi$ in Letcher et al. A mapping from parameter names to tensors.
        simultaneous_grad: dict[str, Tensor] = {}

        # The opponent shaping term for each parameter. Named $\chi$ in Letcher et al.
        # A mapping from parameter names to tensors.
        opponent_shaping: dict[str, Tensor] = {}

        # TODO (Sam): Rename this to something more descriptive
        H_0_xi: dict[str, Tensor] = {}

        # TODO (Sam): Explain what this is
        total_derivatives: dict[str, dict[str, Tensor]] = {}

        # TODO (Sam): Explain a bit more what's happening here, adding some comments and
        # maybe putting this inside a method with a nice docstring
        for leader_name in reversed(list(self.stackelberg_sequence_flat)):

            simultaneous_grad.update(objective_loss_grads[leader_name, leader_name])
            total_derivatives[leader_name] = objective_loss_grads[
                leader_name, leader_name
            ]
            param_names = list(total_derivatives[leader_name].keys())

            for param_name in param_names:
                H_0_xi[param_name] = 0.0
                opponent_shaping[param_name] = 0.0

            for follower_name in self.all_followers[leader_name]:
                score_coefficient, pg_coefficient = self._compute_jacobian_terms(
                    leader_name, follower_name, objective_loss_grads, scores
                )

                # Compute (an approximation of) the true Stackelberg gradient using
                # an inverse Hessian vector product
                if self.variant == "spg" or self.variant == "pspg":
                    multiplier = ihvps[leader_name, follower_name]
                # If using other algorithms we effectively assume that the inverse
                # Hessian is the identity matrix
                else:
                    multiplier = objective_loss_grads[leader_name, follower_name]

                chi_score_term = dict_dot_product(multiplier, scores[follower_name])
                chi_pg_term = dict_dot_product(
                    multiplier, objective_loss_grads[follower_name, follower_name]
                )

                H_0_xi_pg_term = dict_dot_product(
                    scores[follower_name], total_derivatives[follower_name]
                )
                H_0_xi_score_term = dict_dot_product(
                    objective_loss_grads[leader_name, follower_name],
                    total_derivatives[follower_name],
                )

                for param_name in param_names:

                    if self.variant == "spg" or self.variant == "pspg":
                        lr_coefficient = 1.0
                        H_0_xi_term = 0.0

                    # For LOLA and POLA we need to multiply the gradients by the
                    # learning rate of the follower agent
                    else:
                        lr_coefficient = (
                            self.agent_lr_factors[follower_name].actor * self.lr
                        )
                        if (
                            self.additional_lola_term
                            or self.variant == "sos"
                            or self.variant == "psos"
                        ):
                            H_0_xi_term = (
                                H_0_xi_pg_term
                                * objective_loss_grads[leader_name, leader_name][
                                    param_name
                                ]
                            ) + (H_0_xi_score_term * scores[leader_name][param_name])

                    chi_term = (chi_score_term * score_coefficient[param_name]) + (
                        chi_pg_term * pg_coefficient[param_name]
                    )

                    opponent_shaping[param_name] += lr_coefficient * chi_term
                    H_0_xi[param_name] += lr_coefficient * H_0_xi_term

                    total_derivatives[leader_name][param_name] -= lr_coefficient * (
                        chi_term + H_0_xi_term
                    )

        if self.variant == "sos" or self.variant == "psos":
            update = compute_sos_update(
                simultaneous_grad,
                H_0_xi,
                opponent_shaping,
                self.sos_scaling_factor,
                self.sos_threshold_factor,
            )
        else:
            update = {}
            for total_derivative in total_derivatives:
                update.update(total_derivatives[total_derivative])

        for actor_params in self.actor_params.values():
            for param_name, param in actor_params.items():
                param.grad = update[param_name]

        additional_loss = loss_vals["loss_critic"] + loss_vals["loss_entropy"]
        additional_loss.backward()

    def _get_followers(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        """Get dictionaries of the followers of each agent.

        For each agent in the Stackelberg sequence, we get the agents in the group
        immediately following them, as well as all the agents in the groups following
        them. This is returned as two dictionaries.

        Returns
        -------
        immediate_followers : dict[str, tuple[str, ...]]
            A dictionary where the keys are agent names and the values are tuples of
            agent names for the immediate followers of each agent.
        descendent_followers : dict[str, tuple[str, ...]]
            A dictionary where the keys are agent names and the values are tuples of
            agent names for all the followers of each agent (i.e. the immediate
            followers, as well as all the followers of the immediate followers, and so
            on).
        """

        immediate_followers: dict[str, tuple[str, ...]] = {}
        descendent_followers: dict[str, tuple[str, ...]] = {}

        for group_id in range(len(self.stackelberg_sequence)):
            if group_id != len(self.stackelberg_sequence) - 1:

                # Flatten the remaining groups
                remaining_agent_names = []
                for subsequent_group_id in range(
                    group_id + 1, len(self.stackelberg_sequence)
                ):
                    remaining_agent_names += self.stackelberg_sequence[
                        subsequent_group_id
                    ]
                remaining_agent_names = tuple(remaining_agent_names)

                for follower in self.stackelberg_sequence[group_id]:
                    immediate_followers[follower] = self.stackelberg_sequence[
                        group_id + 1
                    ]
                    descendent_followers[follower] = remaining_agent_names

            else:
                for follower in self.stackelberg_sequence[group_id]:
                    immediate_followers[follower] = ()
                    descendent_followers[follower] = ()

        return immediate_followers, descendent_followers

    def _get_actor_params(self) -> dict[str, dict[str, Parameter]]:
        """Get the parameters for each agent as dictionaries with the parameter names.

        Returns
        -------
        actor_params : dict[str, dict[str, Parameter]]
            A dictionary whose keys are the agent names and whose values are
            dictionaries where the keys are the parameter names and the values are the
            parameters.
        """

        actor_params: dict[str, dict[str, Parameter]] = {}
        for agent_name in self.agent_names:
            filtered_params = self.agents[agent_name].filter_actor_named_parameters(
                self.named_parameters()
            )
            actor_params[agent_name] = {name: param for name, param in filtered_params}

        return actor_params

    @torch.no_grad()
    def _get_and_zero_all_grads(self) -> dict[str, dict[str, Tensor]]:
        """Get and zero the gradients for the parameters of all the agents.

        Returns
        -------
        grads : dict[str, dict[str, Tensor]]
            A dictionary where the keys are agent names and the values are dictionaries
            where the keys are the parameter names and the values are the gradients.
        """

        grads: dict[str, dict[str, Parameter]] = {}
        for agent_name, actor_params in self.actor_params.items():
            actor_grads = {}
            for param_name, param in actor_params.items():
                actor_grads[param_name] = param.grad
                param.grad = torch.zeros_like(param)
            grads[agent_name] = actor_grads

        return grads

    def _compute_jacobian_terms(
        self,
        leader_name: str,
        follower_name: str,
        objective_loss_grads: dict[tuple[str, str], dict[str, Tensor]],
        scores: dict[str, dict[str, Tensor]],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        r"""Compute the score and policy gradient coefficients for the Jacobian terms.

        Recursive function to compute elements of the Jacobian (of agent 2's loss with
        respect to agent 2's parameters then agent 1's parameters) using the chain rule
        -- we maintain separate coefficients for the score term and the policy gradient
        term to avoid computing full Jacobian matrices.

        Consider a leader agent $g$ and a follower agent $f$. Let $\nabla_j \Ell_i$ be
        the gradient loss of agent $i$ with respect to the parameters of agent $j$ and
        $\bar S_i$ be the score for agent $i$, which is the gradient of the log sum
        probability of the actions with respect to the parameters of agent $i$. This
        function computes $C_S(g, f)$ and $C_{PG}(g, f)$, the coefficients for the score
        and policy gradient terms in the Jacobian of agent $f$'s loss with respect to
        agent $g$'s parameters.

        When $f$ follows $g$ immediately in the Stackelberg sequence, the score
        coefficient is:
            $$
                C_S(g, f) = \nabla_g \Ell_f
            $$
        and the policy gradient coefficient is:
            $$
                C_{PG}(g, f) = \bar S_g
            $$

        Otherwise, we recursively compute the Jacobian terms for the leader and
        immediate leader $g'$ of $f$. Let $Q$ be the set of immediate leaders of $f$.
        Then the score coefficient is:
            $$
                C_S(g, f) = \sum_{g' \in Q} (
                    (\nabla_{g'} \Ell_f \cdot S_{g'}) C_S(g', f)
                  + (\nabla_{g'} \Ell_f \cdot \nabla_{g'} \Ell_{g'}) C_{PG}(g', f)
                )
            $$

        Parameters
        ----------
        leader_name : str
            The name of the leader agent, which comes before the follower agent in the
            stackelberg_sequence.
        follower_name : str
            The name of the follower agent.
        objective_loss_grads : dict[tuple[str, str], dict[str, Tensor]]
            The gradients of the objective loss for each agent with respect to each
            agent's parameters. The first index is the agent whose loss it is, and the
            second index is the agent whose parameters it is with respect to.
        scores : dict[str, dict[str, Tensor]]
            The scores for each agent.

        Returns
        -------
        score_coefficient : dict[str, Tensor]
            A dictionary of the coefficients for the score term for each of the leader's
            parameters.
        pg_coefficient : dict[str, Tensor]
            A dictionary of the coefficients for the policy gradient term for each of
            the leader's parameters.
        """

        # When the follower is an immediate follower of the leader, the score
        # coefficient is the gradient of the follower's loss with respect to the
        # leader's parameters, and the policy gradient coefficient is the score of the
        # leader.
        if follower_name in self.immediate_followers[leader_name]:
            score_coefficient = objective_loss_grads[follower_name, leader_name]
            pg_coefficient = scores[leader_name]

            return score_coefficient, pg_coefficient

        else:
            score_coefficient = {}
            pg_coefficient = {}

            for immediate_leader_name in self.immediate_leaders[follower_name]:

                # Recursively compute the Jacobian terms for leader and immediate leader
                # of the follower
                leader_score_coefficient, leader_pg_coefficient = (
                    self._compute_jacobian_terms(
                        leader_name,
                        immediate_leader_name,
                        objective_loss_grads,
                        scores,
                    )
                )

                param_names = list(leader_score_coefficient.keys())

                # Compute the contribution to the score function coefficient
                score_coefficient_score = dict_dot_product(
                    objective_loss_grads[follower_name, immediate_leader_name],
                    scores[immediate_leader_name],
                )
                score_coefficient_pg = dict_dot_product(
                    objective_loss_grads[follower_name, immediate_leader_name],
                    objective_loss_grads[immediate_leader_name, immediate_leader_name],
                )
                for param_name in param_names:
                    to_add = (
                        score_coefficient_score * leader_score_coefficient[param_name]
                    )
                    to_add += score_coefficient_pg * leader_pg_coefficient[param_name]
                    dict_update_add(score_coefficient, param_name, to_add)

                # Policy gradient coefficient
                pg_coefficient_score = dict_dot_product(
                    scores[immediate_leader_name], scores[immediate_leader_name]
                )
                pg_coefficient_pg = dict_dot_product(
                    scores[immediate_leader_name],
                    objective_loss_grads[immediate_leader_name, immediate_leader_name],
                )
                for param_name in param_names:
                    to_add = pg_coefficient_score * leader_score_coefficient[param_name]
                    to_add += pg_coefficient_pg * leader_pg_coefficient[param_name]
                    dict_update_add(pg_coefficient, param_name, to_add)

            return score_coefficient, pg_coefficient

    def _compute_ihvps(
        self, loss_vals: TensorDictBase
    ) -> dict[tuple[str, str], dict[str, Tensor]]:
        """Compute the inverse Hessian vector products for each agent and follower.

        This is the inverse Hessian of the follower's loss with respect to the
        follower's parameters multiplied by the gradient of the leader's loss with
        respect to the follower's parameters.

        Parameters
        ----------
        loss_vals : TensorDictBase
            The loss values.

        Returns
        -------
        ihvps : dict[tuple[str, str], dict[str, Tensor]]
            A dictionary where the keys are the agent and follower names, and the values
            are dictionaries of the inverse Hessian vector products for each of the
            follower's parameters.
        """

        loss_objective = loss_vals.get(("agents", "loss_objective"))

        ihvps = {}
        for agent_id, agent_name in enumerate(self.agent_names):
            if len(self.all_followers[agent_name]) == 0:
                continue
            for follower_name in self.all_followers[agent_name]:
                follower_id = self.agent_names.index(follower_name)
                ihvps[(agent_name, follower_name)] = inverse_hessian_vector_product(
                    follower_loss=loss_objective[..., follower_id, follower_id],
                    leader_loss=loss_objective[..., agent_id, agent_id],
                    follower_params=self.actor_params[follower_name],
                    leader_params=self.actor_params[agent_name],
                    variant=self.ihvp_arguments["variant"],
                    num_iterations=self.ihvp_arguments["num_iterations"],
                    rank=self.ihvp_arguments["rank"],
                    rho=self.ihvp_arguments["rho"],
                )

        return ihvps


class ReinforceLossImproved(Objective, ReinforceLoss):
    """Reinforce loss which allows multiple actions keys and normalises advantages.

    The implementation is also tweaked slightly to allow it to work without a critic. In
    this case reward-to-go is used instead of the advantage.

    The __init__ method is copied from the original ReinforceLoss class with some
    tweaks.

    See `torchrl.objectives.ReinforceLoss` for more details

    Parameters
    ----------
    actor_network : ProbabilisticTensorDictSequential
        The policy operator.
    critic_network : TensorDictModule, optional
        The value operator, if using a critic.
    loss_weighting_type : str, optional
        The type of weighting to use in the loss. Can be one of "advantage" or
        "reward_to_go". The former requires a critic network. Defaults to "advantage"
        when a critic is used, otherwise "reward_to_go".
    delay_value : bool, optional
        If ``True``, a target network is needed for the critic. Defaults to ``False``.
        Incompatible with ``functional=False``.
    loss_critic_type : str, default="smooth_l1"
        Loss function for the value discrepancy. Can be one of "l1", "l2" or
        "smooth_l1".
    gamma : float, optional
        The discount factor. Required if ``loss_weighting_type="reward_to_go"``.
    separate_losses : bool, default=False
        If ``True``, shared parameters between policy and critic will only be trained on
        the policy loss. Defaults to ``False``, ie. gradients are propagated to shared
        parameters for both policy and critic losses.
    functional : bool, default=True
        Whether modules should be functionalized. Functionalizing permits features like
        meta-RL, but makes it impossible to use distributed models (DDP, FSDP, ...) and
        comes with a little cost. Defaults to ``True``.
    normalize_advantage : bool, default=True
        Whether to normalise the advantage. Defaults to ``True``.
    clip_value (float, optional):
        If provided, it will be used to compute a clipped version of the value
        prediction with respect to the input tensordict value estimate and use it to
        calculate the value loss. The purpose of clipping is to limit the impact of
        extreme value predictions, helping stabilize training and preventing large
        updates. However, it will have no impact if the value estimate was done by the
        current version of the value estimator. Defaults to ``None``.
    """

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        critic_network: Optional[TensorDictModule] = None,
        *,
        loss_weighting_type: Optional[str] = None,
        delay_value: bool = False,
        loss_critic_type: str = "smooth_l1",
        gamma: Optional[float] = None,
        advantage_key: Optional[str] = None,
        value_target_key: Optional[str] = None,
        separate_losses: bool = False,
        functional: bool = True,
        normalize_advantage: bool = True,
        clip_value: Optional[float] = None,
    ):
        if actor_network is None:
            raise TypeError("Missing positional argument actor_network.")
        if not functional and delay_value:
            raise RuntimeError(
                "delay_value and ~functional are incompatible, as delayed value currently relies on functional calls."
            )

        if loss_weighting_type is None:
            if critic_network is None:
                loss_weighting_type = "reward_to_go"
            else:
                loss_weighting_type = "advantage"
        elif loss_weighting_type == "advantage" and critic_network is None:
            raise ValueError("Cannot use advantage weighting without a critic network.")
        elif loss_weighting_type == "reward_to_go" and critic_network is not None:
            warnings.warn(
                "Using reward-to-go weighting but a critic network is provided. "
                "This will result in the critic being ignored."
            )
        if loss_weighting_type not in ["advantage", "reward_to_go"]:
            raise ValueError(
                f"loss_weighting_type must be one of 'advantage' or 'reward_to_go', but "
                f"got {loss_weighting_type!r}."
            )
        if loss_weighting_type == "reward_to_go":
            if gamma is None:
                raise ValueError(
                    f"gamma must be provided when using 'reward_to_go' weighting."
                )
            self.gamma = gamma

        self.loss_weighting_type = loss_weighting_type
        self._functional = functional
        self.normalize_advantage = normalize_advantage
        self.clip_value = clip_value

        # A hacky way to all the grandparent class's __init__ method
        super(ReinforceLoss, self).__init__()
        self.in_keys = None
        self._set_deprecated_ctor_keys(
            advantage=advantage_key, value_target=value_target_key
        )

        self.delay_value = delay_value
        self.loss_critic_type = loss_critic_type

        # Actor
        if self.functional:
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=False,
            )
        else:
            self.actor_network = actor_network

        if separate_losses:
            # we want to make sure there are no duplicates in the hyper_params: the
            # hyper_params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        # Value
        if critic_network is not None:
            if self.functional:
                self.convert_to_functional(
                    critic_network,
                    "critic_network",
                    create_target_params=self.delay_value,
                    compare_against=policy_params,
                )
            else:
                self.critic_network = critic_network
                self.target_critic_network_params = None
        else:
            self.critic_network = None
            self.target_critic_network_params = None
            self.critic_network_params = None

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the given input TensorDict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        TensorDictBase
            The output TensorDict containing the loss values.
        """

        # Compute the weighting used in the loss, which is either the reward-to-go or
        # the advantage, depending on whether a critic is used
        if self.loss_weighting_type == "reward_to_go":
            loss_weighting = tensordict.get(("agents", "reward_to_go"))
        else:
            loss_weighting = self._get_advantage(tensordict)

        # Compute the log-prob
        log_prob, _, _ = self._log_weight(tensordict)
        if log_prob.shape == loss_weighting.shape[:-1]:
            log_prob = log_prob.unsqueeze(-1)

        # Compute the loss
        loss_actor_per_agent = -log_prob * loss_weighting.detach()
        loss_actor_per_agent = loss_actor_per_agent.mean(dim=0)
        td_out = TensorDict(
            {
                "loss_actor": loss_actor_per_agent.sum(),
                "agents": {
                    "loss_actor": loss_actor_per_agent,
                },
            },
            [],
        )

        if self.critic_network is not None:
            critic_loss_per_agent = self._loss_critic(tensordict).mean(dim=0)
            td_out.set("loss_critic", critic_loss_per_agent.sum())
            td_out.set(("agents", "loss_value"), self._loss_critic(tensordict).mean())

        if self.loss_weighting_type == "reward_to_go":
            td_out.set("reward_to_go", loss_weighting.mean(dim=0).sum())
            td_out.set(("agents", "reward_to_go"), loss_weighting.mean(dim=0))

        return td_out

    def backward(self, loss_vals: TensorDictBase):
        """Perform the backward pass for the loss.

        Parameters
        ----------
        loss_vals : TensorDictBase
            The loss values.
        """
        if self.critic_network is not None:
            loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]
        else:
            loss_value = loss_vals["loss_actor"]
        loss_value.backward()

    def _loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Get the critic loss without the clip fraction.

        TorchRL's `loss_critic` method returns a tuple with the critic loss and the
        clip fraction. This method returns only the critic loss.
        """
        return self.loss_critic(tensordict)[0]
