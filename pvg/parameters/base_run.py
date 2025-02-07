"""Parameters for basing the current experiment on a previous W&B run.

The hyper-parameters and/or log statistics of a previous run can be used to initialize
the current experiment.
"""

from dataclasses import dataclass
import dataclasses
from typing import Optional, Literal

from pvg.parameters.parameters_base import SubParameters, register_parameter_class
from pvg.utils.env import env_var_default_factory


BaseRunType = Literal["none", "parameters", "rerun_tests"]
"""Enum for how to base the current experiment on a previous W&B run.

Values
------
none
    Do not base the current experiment on a previous run.
parameters
    Use the hyper-parameters of a previous run to initialize the current experiment.
rerun_tests
    Rerun the tests of a previous run. The hyper-parameters controlling the tests
    can be different.
"""


@register_parameter_class
@dataclass
class BaseRunParameters(SubParameters):
    """Parameters for basing the current experiment on a previous W&B run.

    Parameters
    ----------
    base_run_type : BaseRunType
        How to base the current experiment on a previous W&B run.
    run_id : str, optional
        The run ID of the run to base the current experiment on. This must be provided
        if `base_run_type` is not `BaseRunType.NONE`.
    wandb_project : str
        The W&B project of the run to base the current experiment on. If not provided,
        the default project is used.
    wandb_entity : str
        The W&B entity of the run to base the current experiment on. If not provided,
        the default entity is used.
    rerun_tests_force_test_during_training_state, bool
        When `base_run_type` is set to "rerun_tests", if True this forces the existence
        of a "test_during_training" state, even when it was not present during training.
        Older runs did not have this state, so without this option it would be
        impossible to redo a run testing every iteration. If `test_every_iteration` is
        not True, this option has no effect, so it's safe to leave it as True.
    """

    base_run_type: BaseRunType = "none"

    run_id: Optional[str] = None
    wandb_project: str = dataclasses.field(
        default_factory=env_var_default_factory("WANDB_PROJECT")
    )
    wandb_entity: str = dataclasses.field(
        default_factory=env_var_default_factory("WANDB_ENTITY")
    )

    rerun_tests_force_test_during_training_state: bool = True


class BaseRunPreserve:
    """Type annotation to preserve a parameter when initializing from a base run.

    Any parameter with this type annotation will be preserved when initializing from a
    base run of the given type.

    Example
    -------
    >>> from typing import Annotated
    >>> @register_parameter_class
    >>> @dataclass
    >>> class TestParameters(SubParameters):
    >>>     preserved_param: Annotated[int, BaseRunPreserve("rerun_tests")]

    Parameters
    ----------
    *base_run_types : BaseRunType
        The base run types to preserve the parameter for.
    """

    def __init__(self, *base_run_types: BaseRunType):
        self.base_run_types = base_run_types
