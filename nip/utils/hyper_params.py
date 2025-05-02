"""Functions for handling parameters and deriving properties from them."""

from nip.parameters import HyperParameters, PureTextAgentParameters


def get_agent_part_flags(
    hyper_params: HyperParameters,
) -> tuple[bool, bool]:
    """Get flags indicating which agent parts are used.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.

    Returns
    -------
    use_critic : bool | None
        Whether the experiment uses a critic.
    use_single_body : bool
        Whether to create a single body. When there is a critic and a single body, the
        critic and actor share the same body. When there is a critic and two bodies, the
        critic and actor have separate bodies.
    use_whole_agent : bool
        Whether agents are composed of a single part, and are not split body and heads.
    """

    if hyper_params.trainer == "solo_agent":
        return False, True, False
    elif hyper_params.trainer == "vanilla_ppo" or hyper_params.trainer == "spg":
        return True, hyper_params.rl.use_shared_body, False
    elif hyper_params.trainer == "reinforce":
        return (
            hyper_params.reinforce.use_advantage_and_critic,
            hyper_params.rl.use_shared_body
            or not hyper_params.reinforce.use_advantage_and_critic,
            False,
        )
    elif (
        hyper_params.trainer == "pure_text_ei"
        or hyper_params.trainer == "pure_text_malt"
    ):
        return False, False, True
    else:
        raise ValueError(f"Unknown trainer type: {hyper_params.trainer}")


def check_use_supervisor_message(
    agent_params: PureTextAgentParameters, round_id: int
) -> bool:
    """Check if we should include the supervisor message in chat history.

    The supervisor message is a message that is appended to the chat history before
    being sent to the model. Whether to include it or not is determined by the
    agent's parameters and the round ID.

    Parameters
    ----------
    agent_params : PureTextAgentParameters
        The parameters of the agent.
    round_id : int
        The current round number.

    Returns
    -------
    use_supervisor_message : bool
        Whether to include the supervisor message in the chat history.
    """

    if agent_params.use_supervisor_message == "none":
        return False
    elif agent_params.use_supervisor_message == "all":
        return True
    elif agent_params.use_supervisor_message == "first":
        return round_id == 0
    elif agent_params.use_supervisor_message == "all_but_first":
        return round_id != 0
    else:
        raise ValueError(
            f"Unknown use_supervisor_message value: "
            f"{agent_params.use_supervisor_message!r}"
        )
