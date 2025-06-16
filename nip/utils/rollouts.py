"""Utilities for handling rollouts."""

import numpy as np

from jaxtyping import Int, Float

from nip.protocols.protocol_base import ProtocolHandler
from nip.utils.types import String


def get_pretty_pure_text_round_message(
    protocol_handler: ProtocolHandler,
    decision: Int[np.ndarray, "agent"],
    raw_decision: String[np.ndarray, "agent"],
    continuous_decision: Float[np.ndarray, "agent"],
    message: String[np.ndarray, "agent channel"],
) -> dict[str, str]:
    """Get a pretty version of the messages sent in a round of a pure-text scenario.

    This function returns a dict of ``key: value`` pairs determined as follows:

    - If a verifier has made a decision, the dict has keys of the form
      "{verifier_name}.decision" and values the decision made by each verifier which
      made a decision, including the continuous decision value in parentheses.
    - If no verifier has made a decision, the keys are of the form
      "{agent_name}@{channel_name}" and the values are the messages sent. If no agent
      sent a message, the dict is empty.

    Parameters
    ----------
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    decision : Int[np.ndarray, "agent"]
        The discrete decision made by each agent, where any value other than 2 indicates
        that the agent has made a decision.
    raw_decision : String[np.ndarray, "agent"]
        The raw decision text sent by each agent (which may be None).
    continuous_decision : Float[np.ndarray, "agent"]
        A float version of the decision made by each agent, which is a value between -1
        and 1.
    message : String[np.ndarray, "agent channel"]
        The messages sent by each agent to each channel (which may be None).

    Returns
    -------
    pretty_message_dict : dict[str, str]
        A dictionary of the messages sent in the round, determined as described above.
    """

    pretty_message_dict = {}

    # We first check the decision made by a verifier, and if it is made, we
    # set the processed transcript to "Accept" or "Reject" based on the
    # decision.
    for verifier_name in protocol_handler.verifier_names:
        verifier_index = protocol_handler.agent_names.index(verifier_name)
        if decision[verifier_index] == 2:
            continue
        pretty_message_dict[f"{verifier_name}.decision"] = (
            f"{raw_decision[verifier_index]} ({continuous_decision[verifier_index]})"
        )

    if len(pretty_message_dict) > 0:
        return pretty_message_dict

    # Otherwise, we look at the message history for the message sent this timestep.
    # The key is the active agent name and channel name, with an "@" in between.
    for agent_id, agent_name in enumerate(protocol_handler.agent_names):
        for channel_id, channel_name in enumerate(
            protocol_handler.message_channel_names
        ):

            if message[agent_id, channel_id] is None:
                continue

            # Add the message to the processed transcript with the key
            # "{agent_name}@{channel_name}"
            pretty_message_dict[f"{agent_name}@{channel_name}"] = message[
                agent_id, channel_id
            ]

    return pretty_message_dict
