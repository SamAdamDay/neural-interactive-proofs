"""Functions for handling parameters and deriving properties from them."""

from pvg.parameters import Parameters, TrainerType


def check_if_critic_and_single_body(params: Parameters) -> tuple[bool, bool]:
    """Check if we need a critic and if we need a single body.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.

    Returns
    -------
    use_critic : bool
        Whether the experiment uses a critic.
    use_single_body : bool
        Whether to create a single body. When there is a critic and a single body, the
        critic and actor share the same body. When there is a critic and two bodies, the
        critic and actor have separate bodies.
    """

    if params.trainer == TrainerType.SOLO_AGENT:
        return False, True
    if params.trainer == TrainerType.VANILLA_PPO or params.trainer == TrainerType.SPG:
        return True, params.rl.use_shared_body
    if params.trainer == TrainerType.REINFORCE:
        return (
            params.reinforce.use_advantage_and_critic,
            params.rl.use_shared_body or not params.reinforce.use_advantage_and_critic,
        )
