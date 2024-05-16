"""PyTorch distributions"""

from torch.distributions import Categorical

from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import CompositeDistribution


class CompositeCategoricalDistribution(CompositeDistribution):
    """A composition of categorical distributions.

    Allows specifying the parameters of the categorical distributions either as logits
    or as probabilities.

    The `log_prob` method is reimplemented with the following changes:

    - The log-probability can be stored in a different key (specified by the
      "log_prob_key" parameter)
    - It only computes stores the total log-probability, not the individual ones
    - It doesn't reduce all non-batch dimensions in the log-probability
    - It has a method to compute the entropy of the distribution

    Parameter names must be strings ending in "_logits" or "_probs". However, the
    suffix-stripped can be changed by passing a key transform function or a lookup
    table. For example, to specify the parameters of a categorical distribution over key
    `("agents", "action")` using logits, you can pass the following:

    >>> CompositeCategoricalDistribution(
    ...     action_logits=...,
    ...     key_transform=lambda x: ("agents", x)
    ... )

    Parameters
    ----------
    **categorical_params : dict[str, Tensor]
        The parameters of the categorical distributions. Each key is the name of a
        categorical parameter appended with "_logits" or "_probs" and each value is a
        Tensor containing the logits or probabilities of the categorical distribution.
    key_transform : callable[[str], [str | tuple[str]]] | dict[str, str | tuple[str]],
    optional
        A function that transforms the keys of the categorical parameters. If a dict is
        given, it is used as a lookup table. If a callable is given, it is applied to
        each key. Defaults to the identity function.
    log_prob_key: NestedKey, default="sample_log_prob"
        The tensordict key to use for the log-probability of the sample.
    """

    def __init__(self, **kwargs):
        # Get the key transform
        try:
            key_transform = kwargs.pop("key_transform")
            if isinstance(key_transform, dict):
                key_transform = lambda x: key_transform[x]
            elif not callable(key_transform):
                raise ValueError("key_transform must be a callable or a dict.")
        except KeyError:
            key_transform = lambda x: x

        # Get the log-probability key
        self.log_prob_key = kwargs.pop("log_prob_key", "sample_log_prob")

        batch_size = None
        composite_params = {}
        name_suffixes = ("logits", "probs")
        for name, param_value in kwargs.items():
            for name_suffix in name_suffixes:
                if name.endswith("_" + name_suffix):
                    # Set the parameters of the categorical distribution
                    resolved_param_name = key_transform(name[: -len(name_suffix) - 1])
                    composite_params[resolved_param_name] = {name_suffix: param_value}

                    # Make sure all the categorical parameters have the same batch size
                    if batch_size is None:
                        batch_size = param_value.shape[:2]
                    elif batch_size != param_value.shape[:2]:
                        raise ValueError(
                            "All categorical parameters must have the same batch size."
                        )

        composite_params_td = TensorDict(composite_params, batch_size=batch_size)

        super().__init__(
            params=composite_params_td,
            distribution_map={key: Categorical for key in composite_params},
        )

    def log_prob(self, sample: TensorDictBase) -> TensorDictBase:
        """Computes the log probability of a sample for the composite distribution

        Adapted from `tensordict.nn.distributions.CompositeDistribution.log_prob`.

        Parameters
        ----------
        sample: TensorDictBase
            A tensordict containing the sample

        Returns
        -------
        updated_sample: TensorDictBase
            The sample tensordict updated with the log probability of the sample
        """
        slp = 0.0
        d = {}
        for name, dist in self.dists.items():
            slp += dist.log_prob(sample.get(name))
        d[self.log_prob_key] = slp
        sample.update(d)
        return sample

    def entropy(self) -> float:
        """Computes the entropy of the composite distribution

        Returns
        -------
        entropy: float
            The entropy of the composite distribution
        """
        entropy = 0.0
        for dist in self.dists.values():
            entropy += dist.entropy()
        return entropy
