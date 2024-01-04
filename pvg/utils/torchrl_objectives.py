from dataclasses import dataclass, make_dataclass, field, fields
from typing import Iterable

import torch

from tensordict import TensorDictBase, TensorDict
from tensordict.nn.distributions import CompositeDistribution
from tensordict.utils import NestedKey

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

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        # current log_prob of actions
        actions = {key: tensordict.get(key) for key in self.action_keys}
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

        dist = self.actor.get_dist(tensordict, params=self.actor_params)

        if not isinstance(dist, CompositeDistribution):
            raise RuntimeError(
                f"Actor must return a CompositeDistribution to work with "
                f"{self.__name__}, but got {dist}."
            )

        log_prob = dist.log_prob(action_tensordict)

        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensordict prev_log_prob requires grad.")

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        return log_weight, dist


class ClipPPOLossMultipleActions(PPOLossMultipleActions, ClipPPOLoss):
    """Clipped PPO loss class which allows multiple actions keys

    See `torchrl.objectives.ClipPPOLoss` for more details
    """

    pass
