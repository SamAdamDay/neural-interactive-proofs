"""Implementations of interaction protocols.

A protocol is implemented by a protocol handler, which specifies the agents present, how
they interact, and how the environment is updated.

Every protocol handler is a subclass of `ProtocolHandler` and registers itself with the
use of the `register_protocol_handler` decorator. The `build_protocol_handler` factory
function can then be used to build a protocol handler from parameters.
"""

from .protocol_base import (
    ProtocolHandler,
    SingleVerifierProtocolHandler,
    DeterministicSingleVerifierProtocolHandler,
)
from .zero_knowledge import ZeroKnowledgeProtocol
from .registry import register_protocol_handler, build_protocol_handler
from .main_protocols import (
    PvgProtocol,
    DebateProtocol,
    AdpProtocol,
    MnipProtocol,
    MultiChannelTestProtocol,
)
