"""Functions for handling parameters and deriving properties from them."""

from pvg.parameters import Parameters, TrainerType


def get_agent_part_flags(
    params: Parameters,
) -> tuple[bool, bool]:
    """Get flags indicating which agent parts are used.

    Parameters
    ----------
    params : Parameters
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

    if params.trainer == TrainerType.SOLO_AGENT:
        return False, True, False
    elif params.trainer == TrainerType.VANILLA_PPO or params.trainer == TrainerType.SPG:
        return True, params.rl.use_shared_body, False
    elif params.trainer == TrainerType.REINFORCE:
        return (
            params.reinforce.use_advantage_and_critic,
            params.rl.use_shared_body or not params.reinforce.use_advantage_and_critic,
            False,
        )
    elif params.trainer == TrainerType.PURE_TEXT_EI:
        return False, False, True
    else:
        raise ValueError(f"Unknown trainer type: {params.trainer}")
