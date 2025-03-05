"""Utilities to help with training models."""

from typing import Any, Iterable

from torch.optim import Optimizer


class ParamGroupFreezer:
    """A class for freezing groups of parameters during training.

    The ``param_group_groups`` collects the optimizer's parameter groups into named
    collections. Each collection can be frozen and unfrozen at any point.

    Freezing is achieved by setting removing the parameter groups from the optimizer and
    optionally setting the ``requires_grad`` attribute to False.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer which has the parameter groups to freeze.
    param_group_collections : dict[str, Iterable[dict[str, Any]]]
        A dictionary whose keys name collections of parameter groups which can be frozen
        during training. The values are lists of dictionaries which define the parameter
        groups (as stored in ``optimizer.param_groups``). The keys of the dictionaries
        are "params" and "lr".
    use_required_grad : bool, default=True
        Whether to set the ``requires_grad`` to False when freezing the parameters.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_group_collections: dict[str, Iterable[dict[str, Any]]],
        use_required_grad: bool = True,
    ):
        self.optimizer = optimizer
        self.param_group_collections = param_group_collections
        self.use_required_grad = use_required_grad

        # Check that all the parameter groups are part of the optimizer and get the
        # original required_grad values for each parameter
        self._original_requires_grad = {}
        for collection_name, collection in param_group_collections.items():
            for i, group in enumerate(collection):
                for optim_group in optimizer.param_groups:
                    if group["params"] is optim_group["params"]:
                        break
                else:
                    raise ValueError(
                        f"{type(self).__name__} requires all parameter groups to be "
                        f"part of the optimizer, but the {i}th parameter group in "
                        f"{collection_name!r} is not in the optimizer."
                    )
                if use_required_grad:
                    for param in group["params"]:
                        self._original_requires_grad[id(param)] = param.requires_grad

    def freeze(self, collection_name: str):
        """Freeze the parameter groups in the named collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection of parameter groups to freeze.
        """

        for group in self.param_group_collections[collection_name]:
            for i, optim_group in enumerate(self.optimizer.param_groups):
                if group["params"] is optim_group["params"]:
                    del self.optimizer.param_groups[i]
            if self.use_required_grad:
                for param in group["params"]:
                    param.requires_grad = False

    def unfreeze(self, collection_name: str):
        """Unfreeze the parameter groups in the named collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection of parameter groups to unfreeze.
        """

        for group in self.param_group_collections[collection_name]:
            for optim_group in self.optimizer.param_groups:
                if group["params"] is optim_group["params"]:
                    break
            else:
                self.optimizer.add_param_group(group)
            if self.use_required_grad:
                for param in group["params"]:
                    param.requires_grad = self._original_requires_grad[id(param)]
