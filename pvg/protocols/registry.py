"""Registry and factory functions for protocol handlers."""

from typing import TypeVar, Callable

from pvg.parameters import Parameters, InteractionProtocolType
from pvg.experiment_settings import ExperimentSettings
from pvg.protocols.base import ProtocolHandler
from pvg.protocols.zero_knowledge import ZeroKnowledgeProtocol


PROTOCOL_HANDLER_REGISTRY: dict[InteractionProtocolType, type[ProtocolHandler]] = {}

P = TypeVar("P", bound=ProtocolHandler)


def register_protocol_handler(
    protocol_handler: InteractionProtocolType,
) -> Callable[[type[P]], type[P]]:
    """Decorator to register a protocol handler."""

    def decorator(cls: type[P]) -> type[P]:
        PROTOCOL_HANDLER_REGISTRY[protocol_handler] = cls
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

    base_protocol_cls = PROTOCOL_HANDLER_REGISTRY[params.interaction_protocol]

    if params.protocol_common.zero_knowledge:
        return ZeroKnowledgeProtocol(
            params, settings, base_protocol_cls=base_protocol_cls
        )
    else:
        return base_protocol_cls(params, settings)
