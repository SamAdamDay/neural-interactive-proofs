"""Extensions of TorchRL objectives to overcome limitations of the library."""

from dataclasses import dataclass, make_dataclass, field, fields
from typing import Iterable

import torch

from tensordict import TensorDictBase, TensorDict
from tensordict.nn.distributions import CompositeDistribution
from tensordict.utils import NestedKey

from pvg.parameters import SpgVariant
from pvg.utils.maths import dot, ihvp

from torch import distributions as d

from torchrl.objectives import PPOLoss, ClipPPOLoss


class PPOLossMultipleActions(PPOLoss):
    """Parent PPO loss class which allows multiple actions keys

    The implementation is a bit of a hack. We change the _AcceptedKeys class dynamically
    to allow for multiple action keys.

    See `torchrl.objectives.PPOLoss` for more details
    """

    action_keys: Iterable[NestedKey] = ("action",)

    @dataclass
    class _BaseAcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using
        '.set_keys(key_name=key_value)' and their default values

        Attributes:
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
            *self.actor.in_keys,
            *[("next", key) for key in self.actor.in_keys],
            *self.critic.in_keys,
        ]
        self._in_keys = list(set(keys))

    # Modified to output both log probabilities and log_weights
    def _log_weight(
        self, sample: TensorDictBase
    ) -> tuple[torch.Tensor, torch.distributions.Distribution]:
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
            if actions_batch_size is None:
                actions_batch_size = action.shape[0]
            elif actions_batch_size != action.shape[0]:
                raise RuntimeError(
                    f"tensordict stored {key} has batch size {action.shape[0]} "
                    f"but expected {actions_batch_size}."
                )

        action_tensordict = TensorDict(actions, batch_size=actions_batch_size)

        with self.actor_params.to_module(self.actor):
            dist = self.actor.get_dist(sample)

        if not isinstance(dist, CompositeDistribution):
            raise RuntimeError(
                f"Actor must return a CompositeDistribution to work with "
                f"{self.__name__}, but got {dist}."
            )

        log_prob = dist.log_prob(action_tensordict).get(
            self.tensor_keys.sample_log_prob
        )
        prev_log_prob = sample.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensordict prev_log_prob requires grad.")

        log_weight = log_prob - prev_log_prob  # .unsqueeze(-1)

        return log_prob, log_weight, dist

    # We modify the loss function to normalise per agent
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean(dim=0).item()
            scale = advantage.std(dim=0).clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        _, log_weight, dist = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same
            # source. Here we sample according to different, unrelated trajectories,
            # which is not standard. Still it can give a idea of the dispersion of the
            # weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = log_weight_clip.exp() * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]

        td_out = TensorDict({"loss_objective": -gain.mean(dim=0).sum()}, [])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean(dim=0).sum().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean(dim=0).sum())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean(dim=0).sum())
        td_out.set("ESS", ess.mean(dim=0).sum() / batch)

        return td_out

    def compute_grads(self, loss_vals: TensorDictBase):
        """Compute the gradients of the loss.

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


class ClipPPOLossMultipleActions(PPOLossMultipleActions, ClipPPOLoss):
    """Clipped PPO loss class which allows multiple actions keys

    See `torchrl.objectives.ClipPPOLoss` for more details
    """

    pass


class SpgLoss(PPOLossMultipleActions, ClipPPOLoss):
    """Loss for Stackelberg Policy Gradient and several variants, including LOLA and
    POLA.

    We return losses per agent, as well as the sum of the log probabilities, in order to
    then compute the scores later on in the compute_grads function.
    """

    def __init__(
        self,
        actor,
        critic,
        variant: SpgVariant,
        stackelberg_sequence: tuple[tuple[int]],
        names: list[str],
        ihvp: dict,
        clip_epsilon,
        entropy_coef,
        normalize_advantage,
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            normalize_advantage=normalize_advantage,
        )
        self.variant = variant
        self.stackelberg_sequence = stackelberg_sequence
        self.agent_indices = range(sum(len(g) for g in stackelberg_sequence))
        self.agent_groups = {
            i: stackelberg_sequence.index(g) for g in stackelberg_sequence for i in g
        }
        self.followers = self.get_followers()
        self.leaders = {
            i: tuple(j for j in self.agent_indices if i in self.followers[j])
            for i in self.agent_indices
        }
        self.ihvp = ihvp
        self.names = names

        try:
            self.device = next(self.parameters()).device
        except AttributeError:
            self.device = torch.device("cpu")

        self.register_buffer(
            "clip_epsilon", torch.tensor(clip_epsilon, device=self.device)
        )

    def get_followers(self):
        """
        Returns a dictionary of followers for each element in the stackelberg_sequence.

        Returns:
            dict: A dictionary where the keys are elements in the stackelberg_sequence
            and the values are the followers.
        """

        followers = {}

        for g in range(len(self.stackelberg_sequence)):
            if g != len(self.stackelberg_sequence) - 1:
                for i in self.stackelberg_sequence[g]:
                    followers[i] = self.stackelberg_sequence[g + 1]
            else:
                for i in self.stackelberg_sequence[g]:
                    followers[i] = ()

        return followers

    def get_actor_params(
        self, grads=False
    ):  # This probably isn't the most memory efficient way to do things
        """
        Returns a dictionary containing the split gradients for each agent.

        Returns:
            actor_params (dict): A dictionary where the keys are agent indices and the
            values are dictionaries
                                containing the parameters or the gradients thereof for
                                each agent.
        """
        actor_params = {i: {} for i in self.agent_indices}
        for i in self.agent_indices:
            for param_name, param in self.named_parameters():
                if self.names[i] in param_name and "actor" in param_name:
                    if grads:
                        actor_params[i][
                            param_name
                        ] = param.grad.clone()  # Gradients are cloned
                        param.grad.zero_()  # Zero the gradient after it's been recorded
                    else:
                        actor_params[i][param_name] = param  # Parameters are not cloned

        return actor_params

    # Define the (variants of the) SPG loss @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        # Normalise advantages per agent
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean(dim=0).item()
            scale = advantage.std(dim=0).clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        log_prob, log_weight, dist = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same
            # source. Here we sample according to different, unrelated trajectories,
            # which is not standard. Still it can give a idea of the dispersion of the
            # weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        # Use vanilla A2C losses for SPG and LOLA
        if self.variant == SpgVariant.SPG or self.variant == SpgVariant.LOLA:
            probs = log_prob.exp()
            gains = {}
            for i in self.agent_indices:
                gains[i] = (probs * advantage[:, i].unsqueeze(dim=1)).mean(dim=0)

            log_prob_sum = (
                log_prob.sum()
            )  # TODO we don't take a mean across samples (because the score multiplicands have already been averaged across samples), which I believe is correct in theory, but might need changed in practice

        # Otherwise use the clipped PPO loss for PSPG and POLA
        elif self.variant == SpgVariant.PSPG or self.variant == SpgVariant.POLA:
            gains = {}
            for i in self.agent_indices:
                gain1 = log_weight.exp() * advantage[:, i].unsqueeze(dim=1)
                log_weight_clip = log_weight.clamp(*self._clip_bounds)
                gain2 = log_weight_clip.exp() * advantage[:, i].unsqueeze(dim=1)
                gains[i] = torch.stack([gain1, gain2], -1).min(dim=-1)[0].mean(dim=0)

            log_prob_sum = (
                log_weight.sum()
            )  # TODO maybe this should be log_prob.sum(), also note that we don't take a mean across samples because the score multiplicands have already been averaged across samples

        # Return losses per agent and set of followers

        td_out = TensorDict(
            {
                **{"sum_log_probs": log_prob_sum},
                **{
                    f"{self.names[i]}_loss_objectives": -gains[i]
                    for i in self.agent_indices
                },
            },
            [],
        )

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean(dim=0).sum().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean(dim=0).sum())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean(dim=0).sum())
        td_out.set("ESS", ess.mean(dim=0).sum() / batch)

        return td_out

    def compute_grads(
        self, loss_vals: TensorDictBase
    ):  # TODO add learning rates in (to gradient calculation and updates)
        """Compute and assign the gradients of the loss for each agent.

        Parameters
        ----------
        loss_vals : TensorDictBase
            The loss values.
        """

        # Compute scores first
        loss_vals["sum_log_probs"].backward(retain_graph=True)
        scores = self.get_actor_params(grads=True)

        # Then compute objective gradients for all agents
        objective_loss_grads = {}
        for i in self.agent_indices:
            loss_vals[f"{self.names[i]}_loss_objectives"].sum().backward(
                retain_graph=True
            )
            objective_loss_grads[i] = self.get_actor_params(grads=True)

        # Recursive function to compute elements of the Jacobian (of agent j's loss with
        # respect to agent j's parameters then agent i's parameters) using the chain
        # rule – we maintain separate coefficients for the score term and the policy
        # gradient term to avoid computing full Jacobian matrices
        def jacobian_terms(i, j):
            if j in self.followers[i]:
                score_coefficient = objective_loss_grads[j][i]
                pg_coefficient = scores[i]

                return (score_coefficient, pg_coefficient)

            else:
                for k in self.leaders[j]:
                    p = jacobian_terms(i, k)
                    temp_score_coefficient = (
                        dot(objective_loss_grads[j][k], scores[k]) * p[0]
                    ) + (
                        dot(objective_loss_grads[j][k], objective_loss_grads[k][k])
                        * p[1]
                    )
                    temp_pg_coefficient = (dot(scores[k], scores[k]) * p[0]) + (
                        dot(scores[k], objective_loss_grads[k][k]) * p[1]
                    )

                    if self.leaders[j](k) == 0:
                        score_coefficient = temp_score_coefficient
                        pg_coefficient = temp_pg_coefficient
                    else:
                        for key in temp_score_coefficient.keys():
                            score_coefficient[key] += temp_score_coefficient[key]
                            pg_coefficient[key] += temp_pg_coefficient[key]

                return (score_coefficient, pg_coefficient)

        # TODO This is not very memory-efficient but it does prevent the gradients of
        # the parameters getting messed up between calulcating JVPs and assigning
        # gradients
        if self.variant == SpgVariant.SPG or self.variant == SpgVariant.PSPG:
            actor_params = self.get_actor_params()
            ihvps = {
                i: {
                    j: ihvp(
                        loss_vals[f"{self.names[j]}_loss_objectives"][j],
                        loss_vals[f"{self.names[i]}_loss_objectives"][j],
                        actor_params[j],
                        actor_params[i],
                        self.ihvp["variant"],
                        self.ihvp["num_iterations"],
                        self.ihvp["rank"],
                        self.ihvp["rho"],
                    )
                    for j in self.followers[i]
                }
                for i in self.agent_indices
                if len(self.followers[i]) > 0
            }

        for i in self.agent_indices:
            grad = objective_loss_grads[i][i]

            for j in self.followers[i]:
                p = jacobian_terms(i, j)
                # If using LOLA or POLA we effectively assume that the inverse Hessian
                # is the identity matrix
                if self.variant == SpgVariant.LOLA or self.variant == SpgVariant.POLA:
                    multiplier = objective_loss_grads[i][j]
                # Compute (an approximation of) the true Stackelberg gradient using an
                # inverse Hessian vector product
                elif self.variant == SpgVariant.SPG or self.variant == SpgVariant.PSPG:
                    multiplier = ihvps[i][j]
                score_term = dot(multiplier, scores[j])
                pg_term = dot(multiplier, objective_loss_grads[j][j])
                for k in grad.keys():
                    grad[k] += (score_term * p[0][k]) + (pg_term * p[1][k])

            # All the gradient updates needed for agent i are stored in grad
            for param_name, param in self.named_parameters():
                if param_name in grad.keys():
                    param.grad = grad[
                        param_name
                    ]  # TODO check this actually sets the gradients correctly

        additional_loss = loss_vals["loss_critic"] + loss_vals["loss_entropy"]
        additional_loss.backward()

        return
