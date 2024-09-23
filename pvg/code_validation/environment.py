"""The RL environment for code validation."""

from typing import Optional, ClassVar
from functools import cached_property

from numpy import bool_
from numpy.typing import NDArray

from pvg.scenario_base import Environment, PureTextEnvironment
from pvg.factory import register_scenario_class
from pvg.parameters import ScenarioType
from pvg.utils.data import VariableDataCycler
from pvg.utils.nested_array_dict import CompositeSpec, NestedArrayDict, StringArraySpec


@register_scenario_class(ScenarioType.CODE_VALIDATION, Environment)
class CodeValidationEnvironment(PureTextEnvironment):
    """The RL environment for code validation."""

    # We don't batch over environments at this level. Batches of environments are
    # handled by the data collector.
    batch_size: ClassVar[tuple[int, ...]] = (1,)

    @cached_property
    def observation_spec(self) -> CompositeSpec:
        """The specification for the observation keys."""

        observation_spec = super().observation_spec

        observation_spec["question"] = StringArraySpec(*self.batch_size, "batch")
        observation_spec["solution"] = StringArraySpec(*self.batch_size, "batch")

        return observation_spec

    def _masked_reset(
        self,
        env_state: NestedArrayDict,
        mask: NDArray[bool_],
        data_batch: NestedArrayDict,
    ) -> NestedArrayDict:

        env_state = super()._masked_reset(env_state, mask, data_batch)

        env_state["question"] = data_batch["question"]
        env_state["solution"] = data_batch["solution"]

        return env_state
