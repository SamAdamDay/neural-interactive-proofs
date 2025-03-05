"""Parameters specifying the schedule for updating agents.

Each agent follows its own update schedule, which specifies on which iterations the
agent should be updated by the optimizer. On all other iterations, the agent is frozen
and does not update.
"""

from abc import ABC, abstractmethod
import dataclasses

from nip.parameters.parameters_base import (
    ParameterValue,
    register_parameter_value_class,
    get_parameter_or_parameter_value_class,
)


class AgentUpdateSchedule(ParameterValue, ABC):
    """Base class for agent update schedules.

    An agent update schedule specifies on which iterations an agent should be updated by
    the optimizer. On all other iterations, the agent is frozen and does not update.
    """

    def to_dict(self) -> dict:
        """Convert the agent update schedule to a dictionary.

        Returns
        -------
        as_dict : dict
            A dictionary representation of the agent update schedule.
        """

        as_dict = dict(_type=type(self).__name__)

        if len(arguments := self._get_arguments_dict()) > 0:
            as_dict.update({type(self).__name__: arguments})

        return as_dict

    @classmethod
    def from_dict(
        cls, params_dict: dict, ignore_extra_keys: bool = False
    ) -> "AgentUpdateSchedule":
        """Create an agent update schedule from a dictionary.

        Parameters
        ----------
        params_dict : dict
            A dictionary of the agent update schedule.
        ignore_extra_keys : bool, default=False
            If True, ignore keys in the dictionary that do not correspond to fields in
            the parameters object.

        Returns
        -------
        schedule : AgentUpdateSchedule
            The agent update schedule.
        """

        # Get the class of the schedule
        try:
            class_name = params_dict["_type"]
        except KeyError:
            raise ValueError(
                "Missing agent update schedule class ('_type') in dictionary"
            )
        schedule_class = get_parameter_or_parameter_value_class(class_name)

        if ignore_extra_keys:
            params_dict = {
                key: value
                for key, value in params_dict.items()
                if key in {field.name for field in dataclasses.fields(schedule_class)}
            }

        # Create the schedule
        arguments = params_dict.get(class_name, {})
        return schedule_class(**arguments)

    def __repr__(self) -> str:
        argument_string = ", ".join(
            f"{key}={value}" for key, value in self._get_arguments_dict().items()
        )
        return f"{type(self).__name__}({argument_string})"

    def _get_arguments_dict(self) -> dict:
        """Get the constructor arguments of the schedule as a dictionary.

        Returns
        -------
        arguments : dict
            A dictionary of the arguments of the schedule.
        """
        return {}


@register_parameter_value_class
class ConstantUpdateSchedule(AgentUpdateSchedule):
    """An agent update schedule which updates the agent on all iterations."""


@register_parameter_value_class
class ContiguousPeriodicUpdateSchedule(AgentUpdateSchedule):
    """A periodic schedule where the agent is updated between start and stop iterations.

    The updates are scheduled in a cycle of length ``period``. The agent is updated from
    the ``start`` iteration to the ``stop`` iteration in each cycle.

    Parameters
    ----------
    period : int
        The length of the cycle.
    start : int
        The iteration of the cycle to start updating the agent.
    stop : int
        The iteration of the cycle to stop updating the agent.
    """

    def __init__(self, period: int, start: int, stop: int):
        self.period = period
        self.start = start
        self.stop = stop

    def _get_arguments_dict(self) -> dict:
        return dict(period=self.period, start=self.start, stop=self.stop)


@register_parameter_value_class
class AlternatingPeriodicUpdateSchedule(ContiguousPeriodicUpdateSchedule):
    """A periodic schedule for alternating updates between two agents.

    This schedule is to be used in pairs, where one agent is updated in the first part
    of the cycle and the other agent is updated in the second part of the cycle.

    The first agent is updated for the first ``first_agent_num_iterations`` iterations
    of each cycle. The second agent is updated for the remaining iterations of the
    cycle.

    The first agent should have a schedule with ``first_agent=True``, and the second
    agent should have a schedule with ``first_agent=False``.

    Parameters
    ----------
    period : int
        The length of the cycle.
    first_agent_num_iterations : int
        The number of iterations to update the first agent in each cycle.
    first_agent : bool, default=True
        Whether this schedule is for the first agent in the cycle. Otherwise, it is for
        the second agent.
    """

    def __init__(
        self, period: int, first_agent_num_rounds: int, first_agent: bool = True
    ):
        self.first_agent_num_rounds = first_agent_num_rounds
        self.first_agent = first_agent
        if first_agent:
            start = 0
            stop = first_agent_num_rounds
        else:
            start = first_agent_num_rounds
            stop = period
        super().__init__(period, start, stop)

    def _get_arguments_dict(self) -> dict:
        return dict(
            period=self.period,
            first_agent_num_rounds=self.first_agent_num_rounds,
            first_agent=self.first_agent,
        )
