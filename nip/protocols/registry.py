"""Registry and factory functions for protocol handlers."""

from typing import TypeVar, Callable, Optional

from nip.parameters import HyperParameters, InteractionProtocolType, ScenarioType
from nip.experiment_settings import ExperimentSettings
from nip.protocols.protocol_base import ProtocolHandler
from nip.protocols.zero_knowledge import ZeroKnowledgeProtocol


PROTOCOL_HANDLER_REGISTRY: dict[
    tuple[InteractionProtocolType, ScenarioType | None], type[ProtocolHandler]
] = {}
"""A registry of protocol handlers.

Each item is a tuple of the protocol type and either a scenario type or None. Generic
protocol handlers are registered with None as the scenario type, while scenario-specific
handlers are registered with the scenario type.
"""

P = TypeVar("P", bound=ProtocolHandler)


def register_protocol_handler(
    protocol_handler: InteractionProtocolType,
    scenario_type: Optional[ScenarioType] = None,
) -> Callable[[type[P]], type[P]]:
    """Register a protocol handler."""

    def decorator(cls: type[P]) -> type[P]:
        PROTOCOL_HANDLER_REGISTRY[protocol_handler, scenario_type] = cls
        return cls

    return decorator


def build_protocol_handler(
    hyper_params: HyperParameters, settings: ExperimentSettings
) -> ProtocolHandler:
    """Build a trainer from the parameters.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    # Get the base protocol class, using the one specific to the scenario if available
    if (
        hyper_params.interaction_protocol,
        hyper_params.scenario,
    ) in PROTOCOL_HANDLER_REGISTRY:
        base_protocol_cls = PROTOCOL_HANDLER_REGISTRY[
            hyper_params.interaction_protocol, hyper_params.scenario
        ]
    else:
        base_protocol_cls = PROTOCOL_HANDLER_REGISTRY[
            hyper_params.interaction_protocol, None
        ]

    if hyper_params.protocol_common.zero_knowledge:
        return ZeroKnowledgeProtocol(
            hyper_params, settings, base_protocol_cls=base_protocol_cls
        )
    else:
        return base_protocol_cls(hyper_params, settings)
