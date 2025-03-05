"""Base classes for analysing rollouts."""

from abc import ABC, abstractmethod
from typing import ClassVar, Iterator, Any, TypeVar, Callable

from tensordict import TensorDictBase

from numpy import ma

from nip.parameters import HyperParameters, ScenarioType
from nip.experiment_settings import ExperimentSettings
from nip.protocols import ProtocolHandler
from nip.utils.nested_array_dict import NestedArrayDict


class RolloutAnalyser(ABC):
    """Base class for analysing rollouts.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The experiment settings.
    protocol_handler : ProtocolHandler
        The protocol handler, which controls in interaction between agents.
    """

    name: ClassVar[str]

    @property
    @abstractmethod
    def system_prompt_template_filename(self) -> str:
        """The filename of the system prompt template."""

    @abstractmethod
    def relevant_agents_and_channels(self) -> Iterator[tuple[str, str]]:
        """Return an iterator over agent names and channel names to be analysed.

        Yields
        ------
        agent_name : str
            The name of the agent.
        channel_name : str
            The name of the channel.
        """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
    ):
        self.hyper_params = hyper_params
        self.settings = settings
        self.protocol_handler = protocol_handler

    @abstractmethod
    def forward(
        self, rollouts: NestedArrayDict | TensorDictBase, use_tqdm: bool = False
    ) -> dict[tuple[str, str], Any]:
        """Evaluate the rollouts.

        Parameters
        ----------
        rollouts : NestedArrayDict | TensorDictBase
            The rollouts to evaluate.
        use_tqdm : bool
            Whether to use tqdm for progress bars.

        Returns
        -------
        evaluations : dict[tuple[str, str], Any]
            The evaluations. A dictionary indexed by agent name and channel name, where
            ``evaluations[agent_name, channel_name]`` is the evaluations.
        """


ROLLOUT_ANALYSERS: dict[tuple[ScenarioType, str], type[RolloutAnalyser]] = {}

A = TypeVar("A", bound=RolloutAnalyser)


def register_rollout_analyser(scenario: ScenarioType) -> Callable[[type[A]], type[A]]:
    """Register a rollout analyser."""

    def register(cls: type[A]) -> type[A]:
        ROLLOUT_ANALYSERS[scenario, cls.name] = cls
        return cls

    return register


class PureTextRolloutAnalyser(RolloutAnalyser, ABC):
    """Base class rollout analysers which work in pure-text domains, calling APIs.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The experiment settings.
    protocol_handler : ProtocolHandler
        The protocol handler, which controls in interaction between agents.
    model_name : str
        The name of the model to use to analyse the rollouts. This will by accessed
        using and API.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        model_name: str,
        *,
        use_dummy_api: bool = False,
    ):
        super().__init__(hyper_params, settings, protocol_handler)
        self.model_name = model_name
        self.use_dummy_api = use_dummy_api

    @abstractmethod
    def forward(
        self, rollouts: NestedArrayDict, use_tqdm: bool = False
    ) -> dict[tuple[str, str], ma.MaskedArray]:
        """Evaluate the rollouts.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to evaluate.
        use_tqdm : bool
            Whether to use tqdm for progress bars.

        Returns
        -------
        evaluations : dict[tuple[str, str], ma.MaskedArray]
            The evaluations. A dictionary indexed by agent name and channel name, where
            ``evaluations[agent_name, channel_name]`` is an array of evaluations of
            shape
            (...)
        """
