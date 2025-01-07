"""The RL environment for code validation."""

from typing import Optional, ClassVar
from functools import cached_property

from numpy import bool_
from numpy.typing import NDArray

from pvg.scenario_base import Environment, PureTextEnvironment
from pvg.factory import register_scenario_class
from pvg.parameters import ScenarioType
from pvg.utils.data import VariableDataCycler
from pvg.utils.nested_array_dict import (
    CompositeSpec,
    NestedArrayDict,
    StringArraySpec,
    IntArraySpec,
)


@register_scenario_class("code_validation", Environment)
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
        observation_spec["prover_stance"] = IntArraySpec(*self.batch_size, "batch")

        return observation_spec

    def get_datapoint_from_env_state_as_dict(self, env_state: NestedArrayDict) -> dict:
        """Get the datapoint from a single-element environment state as a dictionary.

        This returns a dictionary which specifies the datapoint for the environment
        state.

        Parameters
        ----------
        env_state : NestedArrayDict
            The environment state.

        Returns
        -------
        datapoint : dict
            The datapoint.
        """

        datapoint = super().get_datapoint_from_env_state_as_dict(env_state)

        datapoint["question"] = str(env_state["question"])
        datapoint["solution"] = str(env_state["solution"])
        datapoint["prover_stance"] = int(env_state["prover_stance"])

        return datapoint

    def _masked_reset(
        self,
        env_state: NestedArrayDict,
        mask: NDArray[bool_],
        data_batch: NestedArrayDict,
    ) -> NestedArrayDict:

        env_state = super()._masked_reset(env_state, mask, data_batch)

        env_state["question"] = data_batch["question"]
        env_state["solution"] = data_batch["solution"]
        env_state["prover_stance"] = data_batch["prover_stance"]

        return env_state
