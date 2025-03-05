"""Utilities for basing the current experiment on a previous W&B run.

The hyper-parameters and/or log statistics of a previous run can be used to initialize
the current experiment.

This module contains a function which loads the previous run and creates a new
hyper-parameters object with the hyper-parameters of the previous run. If the
hyper-parameters come from an older version of the package, it will attempt to convert
them to be compatible.
"""

from typing import Optional, Annotated, get_origin
from types import GenericAlias
import dataclasses
from warnings import warn
from inspect import isclass

import wandb
from wandb.apis.public import Run as WandbRun

from nip.parameters import (
    HyperParameters,
    BaseHyperParameters,
    SubParameters,
    AgentsParameters,
    BaseRunPreserve,
    convert_hyper_param_dict,
)
from nip.utils.types import get_union_elements


def get_base_wandb_run_and_new_hyper_params(
    hyper_params: HyperParameters,
) -> tuple[Optional[WandbRun], HyperParameters]:
    """Get the base W&B run and create a new hyper-parameters object.

    Depending on the ``base_run_type`` in the ``hyper_params``, this function either
    returns ``None`` and the original ``hyper_params`` or the base W&B run and a new
    hyper-parameters object with (some of) the hyper-parameters of the base run.

    Parameters
    ----------
    hyper_params : HyperParameters
        The hyper-parameters of the current experiment.

    Returns
    -------
    wandb_run : WandbRun or None
        The base W&B run object, specified by ``hyper_params``. This is an already
        finished run loaded using the W&B API. If ``base_run_type`` is "none", this is
        ``None``, because we are not basing the current experiment on a previous run.
    new_hyper_params : HyperParameters
        The new hyper-parameters object, based on the base run. If ``base_run_type`` is
        "none", this is the original ``hyper_params`` object.
    """

    base_run_type = hyper_params.base_run.base_run_type

    if base_run_type == "none":
        return None, hyper_params

    if hyper_params.base_run.run_id is None:
        raise ValueError(
            "If ``base_run_type`` is not 'none', ``run_id`` must be provided."
        )

    wandb_api = wandb.Api()

    run = wandb_api.run(
        f"{hyper_params.base_run.wandb_entity}"
        f"/{hyper_params.base_run.wandb_project}"
        f"/{hyper_params.base_run.run_id}"
    )

    # Convert the config dict to be compatible with the current package version
    hyper_param_dict = convert_hyper_param_dict(run.config)

    # First create the new hyper-parameters object by using the W&B run config dict
    new_hyper_params = HyperParameters.from_dict(
        hyper_param_dict, ignore_extra_keys=True
    )

    def revert_preserved_hyper_params(
        hyper_params: BaseHyperParameters,
        new_hyper_params: BaseHyperParameters,
    ):
        """Revert hyper-param values which should be preserved, recursively.

        Modifies ``new_hyper_params`` in place, using the values from ``hyper_params``
        """

        for field in dataclasses.fields(hyper_params):

            hyper_params_value = getattr(hyper_params, field.name)
            new_hyper_params_value = getattr(new_hyper_params, field.name)

            if get_origin(field.type) is Annotated:

                # When the type is annotated, check through the annotations for a
                # ``BaseRunPreserve`` instance. If one is found, and the current base
                # run type is in its list of base run types, we revert the value in the
                # original hyper-params
                preserve_annotation_found = False
                for annotation in field.type.__metadata__:
                    if (
                        isinstance(annotation, BaseRunPreserve)
                        and base_run_type in annotation.base_run_types
                    ):
                        setattr(
                            new_hyper_params,
                            field.name,
                            hyper_params_value,
                        )
                        preserve_annotation_found = True
                        break

                if preserve_annotation_found:
                    continue

                origin_type = field.type.__origin__

            else:
                origin_type = field.type

            # Look through the elements of the origin type considered as a union.
            for sub_type in get_union_elements(origin_type):

                if not isclass(sub_type) or type(sub_type) is GenericAlias:
                    continue

                # If any the union element subclasses ``SubParameters``, this means the
                # current field is a gives sub-hyper-parameters, so recurse down
                if issubclass(sub_type, SubParameters):
                    revert_preserved_hyper_params(
                        hyper_params_value,
                        new_hyper_params_value,
                    )
                    break

                # ``AgentsParameters`` is a special case: it is a dictionary of
                # ``SubParameters`` objects. We recurse into each key
                if issubclass(sub_type, AgentsParameters):

                    for agent_name in new_hyper_params_value.keys():

                        if agent_name not in hyper_params_value:
                            warn(
                                f"The agent {agent_name!r} is present in the base run "
                                f"hyper-parameters but not in the parameters passed to "
                                f"the current run. If any parameters are required to "
                                f"to be preserved from the current parameters, this "
                                f"will not happen for agent {agent_name!r}"
                            )
                            continue

                        revert_preserved_hyper_params(
                            hyper_params_value[agent_name],
                            new_hyper_params_value[agent_name],
                        )

    revert_preserved_hyper_params(hyper_params, new_hyper_params)

    # We never want to overwrite the base run parameters
    new_hyper_params.base_run = hyper_params.base_run

    return run, new_hyper_params
