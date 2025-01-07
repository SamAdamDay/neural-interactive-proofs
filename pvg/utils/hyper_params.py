"""Functions for handling parameters and deriving properties from them."""

from pvg.parameters import HyperParameters, TrainerType


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
    elif hyper_params.trainer == "pure_text_ei":
        return False, False, True
    else:
        raise ValueError(f"Unknown trainer type: {hyper_params.trainer}")
