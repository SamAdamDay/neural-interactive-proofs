"""Parameters specifying the schedule for updating agents.

Each agent follows its own update schedule, which specifies on which iterations the
agent should be updated by the optimizer. On all other iterations, the agent is frozen
and does not update.
"""

from abc import ABC

from pvg.parameters.base import ParameterValue


class AgentUpdateSchedule(ParameterValue, ABC):
    """Base class for agent update schedules.

    An agent update schedule specifies on which iterations an agent should be updated by
    the optimizer. On all other iterations, the agent is frozen and does not update.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def to_dict(self) -> dict:
        return self.__repr__()


class ConstantUpdateSchedule(AgentUpdateSchedule):
    """An agent update schedule which updates the agent on all iterations."""


class ContiguousPeriodicUpdateSchedule(AgentUpdateSchedule):
    """A periodic schedule where the agent is updated between start and stop iterations.

    The updates are scheduled in a cycle of length `period`. The agent is updated from
    the `start` iteration to the `stop` iteration in each cycle.

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.period}, {self.start}, {self.stop})"


class AlternatingPeriodicUpdateSchedule(ContiguousPeriodicUpdateSchedule):
    """A periodic schedule for alternating updates between two agents.

    This schedule is to be used in pairs, where one agent is updated in the first part
    of the cycle and the other agent is updated in the second part of the cycle.

    The first agent is updated for the first `first_agent_num_iterations` iterations of
    each cycle. The second agent is updated for the remaining iterations of the cycle.

    The first agent should have a schedule with `first_agent=True`, and the second agent
    should have a schedule with `first_agent=False`.

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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.period}, {self.first_agent_num_rounds}, "
            f"{self.first_agent})"
        )
