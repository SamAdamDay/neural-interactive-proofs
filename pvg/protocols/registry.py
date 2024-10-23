"""Registry and factory functions for protocol handlers."""

from typing import TypeVar, Callable, Optional

from pvg.parameters import Parameters, InteractionProtocolType, ScenarioType
from pvg.experiment_settings import ExperimentSettings
from pvg.protocols.base import ProtocolHandler
from pvg.protocols.zero_knowledge import ZeroKnowledgeProtocol


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
    """Decorator to register a protocol handler."""

    def decorator(cls: type[P]) -> type[P]:
        PROTOCOL_HANDLER_REGISTRY[protocol_handler, scenario_type] = cls
        return cls

    return decorator


def build_protocol_handler(
    params: Parameters, settings: ExperimentSettings
) -> ProtocolHandler:
    """Factory function for building a trainer from parameters.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    # Get the base protocol class, using the one specific to the scenario if available
    if (params.interaction_protocol, params.scenario) in PROTOCOL_HANDLER_REGISTRY:
        base_protocol_cls = PROTOCOL_HANDLER_REGISTRY[
            params.interaction_protocol, params.scenario
        ]
    else:
        base_protocol_cls = PROTOCOL_HANDLER_REGISTRY[params.interaction_protocol, None]

    if params.protocol_common.zero_knowledge:
        return ZeroKnowledgeProtocol(
            params, settings, base_protocol_cls=base_protocol_cls
        )
    else:
        return base_protocol_cls(params, settings)
