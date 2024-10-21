"""Parameters for the various ML trainers."""

from typing import NamedTuple, Optional
from dataclasses import dataclass

from pvg.parameters.base import SubParameters, register_parameter_class
from pvg.parameters.types import PpoLossType, SpgVariant, IhvpVariant
from pvg.parameters.agents import LrFactors


@register_parameter_class
@dataclass
class RlTrainerParameters(SubParameters):
    """Additional parameters common to all RL trainers.

    Parameters
    ----------
    frames_per_batch : int | None
        The number of frames to sample per training iteration. If `None` we set the
        number of frames so that `rollouts_per_iteration` rollouts are sampled per
        iteration.
    rollouts_per_iteration : int | None
        If `frames_per_batch` is `None`, we use this parameter to determine the number
        of rollouts to sample per iteration. `frames_per_batch` is then set to
        `rollouts_per_iteration * steps_per_env_per_iteration`. If `None`, this defaults
        to the dataset size, so that every training datapoint appears exactly once in
        each iteration.
    steps_per_env_per_iteration : int | None
        Each batch is divided into a number of environments which run trajectories for
        this many steps. Note that when a trajectory ends, a new one is started
        immediately. This must be a factor of `frames_per_batch`, since the number of
        environments is `frames_per_batch / steps_per_env_per_iteration`. If `None`,
        this defaults to `max_message_rounds`.
    num_iterations : int
        The number of sampling and training iterations. `num_iterations *
        frames_per_batch` is the total number of frames sampled during training.
    num_epochs : int
        The number of epochs per training iteration.
    minibatch_size : int
        The size of the minibatches in each optimization step.
    lr : float
        The learning rate.
    anneal_lr : bool
        Whether to (linearly) anneal the learning rate over time. Defaults to `False`.
    max_grad_norm : float
        The maximum norm of the gradients during optimization.
    loss_critic_type : str
        Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
    clip_value : float or bool, optional
        If a ``float`` is provided, it will be used to compute a clipped version of the
        value prediction with respect to the input tensordict value estimate and use it
        to calculate the value loss. The purpose of clipping is to limit the impact of
        extreme value predictions, helping stabilize training and preventing large
        updates. However, it will have no impact if the value estimate was done by the
        current version of the value estimator. If instead ``True`` is provided, the
        ``clip_epsilon`` parameter will be used as the clipping threshold. If not
        provided or ``False``, no clipping will be performed. Defaults to ``False``.
    normalize_observations : bool
        Whether to normalise the observations in the environment.
    num_normalization_steps : int
        The number of steps to use to calculate the mean and standard deviation of the
        observations for normalisation. The environment is run for this many steps in
        total with random actions.
    gamma : float
        The discount factor.
    lmbda : float
        The GAE lambda parameter.
    use_shared_body : bool
        Whether the actor and critic share the same body, when using a critic.
    num_test_iterations : int
        The number of iterations to run the test for.
    """

    # Sampling
    frames_per_batch: int | None = 1000
    rollouts_per_iteration: int | None = None
    steps_per_env_per_iteration: int | None = None
    num_iterations: int = 1000

    # Training
    num_epochs: int = 4
    minibatch_size: int = 64
    lr: float = 0.003
    anneal_lr: bool = False
    max_grad_norm: float = 1.0
    loss_critic_type: str = "smooth_l1"
    clip_value: Optional[float | bool] = False
    normalize_observations: bool = True
    num_normalization_steps: int = 1000

    # Reinforcement learning
    gamma: float = 0.9
    lmbda: float = 0.95

    # Agents
    body_lr_factor: Optional[LrFactors | dict] = None
    use_shared_body: bool = True

    # Testing
    num_test_iterations: int = 10


@register_parameter_class
@dataclass
class CommonPpoParameters(SubParameters):
    """Common parameters for PPO trainers.

    Parameters
    ----------
    loss_type : PpoLossType
        The type of PPO loss function to use. See `PpoLossType` for options.
    clip_epsilon : float
        The PPO clip range when using the clipped PPO loss.
    kl_target : float
        The target KL divergence when using the KL penalty PPO loss.
    kl_beta : float
        The coefficient of the KL penalty term in the PPO loss.
    kl_decrement : float
        The decrement factor for the KL penalty term in the PPO loss.
    kl_increment : float
        The increment factor for the KL penalty term in the PPO loss.
    critic_coef : float
        The coefficient of the critic term in the PPO loss.
    entropy_eps : float
        The coefficient of the entropy term in the PPO loss.
    normalize_advantage : bool
        Whether to normalise the advantages in the PPO loss.
    """

    # Loss function
    loss_type: PpoLossType = PpoLossType.CLIP
    clip_epsilon: float = 0.2
    kl_target: float = 0.01
    kl_beta: float = 1.0
    kl_decrement: float = 0.5
    kl_increment: float = 2.0
    critic_coef: float = 1.0
    entropy_eps: float = 0.001
    normalize_advantage: bool = True


@register_parameter_class
@dataclass
class VanillaPpoParameters(SubParameters):
    """Additional parameters for the vanilla PPO trainer."""


SosParams = NamedTuple("SosParams", [("a", float), ("b", float)])


@register_parameter_class
@dataclass
class SpgParameters(SubParameters):
    """Additional parameters for SPG and its variants.

    Parameters
    ----------
    variant : SpgVariant
        The variant of SPG to use.
    stackelberg_sequence : tuple[tuple[str]]
        The sequence of agents to use in the Stackelberg game. The leaders first then
        their respective followers, and so forth.
    additional_lola_term : bool
        Whether to add an additional term to the SPG loss to make it equivalent to the later version of LOLA (first introduced implicitly in LOLA-DICE) as opposed to the original version.
    sos_params : NamedTuple
        The parameters for the SOS loss.
    ihvp_variant : IhvpVariant
        The variant of IHVP to use.
    ihvp_num_iterations : int
        The number of iterations to use in the IHVP approximation.
    ihvp_rank : int
        The rank of the approximation to use in the IHVP approximation.
    ihvp_rho : float
        The damping factor to use in the IHVP approximation.
    """

    variant: SpgVariant = SpgVariant.PSOS
    stackelberg_sequence: tuple[tuple[int]] = (("verifier",), ("prover",))
    additional_lola_term: bool = True
    sos_params: NamedTuple = SosParams(
        a=0.5, b=0.1
    )  # Default values taken from the original paper

    # IHVP
    ihvp_variant: IhvpVariant = IhvpVariant.NYSTROM
    ihvp_num_iterations: int = 5  # Default value taken from hypergrad package example
    ihvp_rank: int = 5  # Default value taken from hypergrad package example
    ihvp_rho: float = 0.1  # Default value taken from hypergrad package example


@register_parameter_class
@dataclass
class ReinforceParameters(SubParameters):
    """Additional parameters for the REINFORCE trainer.

    Parameters
    ----------
    use_advantage_and_critic : bool
        Whether to use a critic in the REINFORCE trainer and use the advantage estimated
        using it in the loss function. Otherwise reward-to-go is used.
    """

    use_advantage_and_critic: bool = False


@register_parameter_class
@dataclass
class SoloAgentParameters(SubParameters):
    """Additional parameters for running agents in isolation.

    Parameters
    ----------
    num_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size.
    learning_rate : float
        The learning rate.
    body_lr_factor_override : bool
        If true, this overrides the learning rate factor for the body (for both the actor and critic), effectively setting it to 1.
    """

    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001

    # Agents
    body_lr_factor_override: bool = False


@register_parameter_class
@dataclass
class TextRlParameters(SubParameters):
    """Additional parameters for the text-based RL trainers.

    Parameters
    ----------
    save_transcripts : bool
        Whether to save the transcripts of the rollouts. Note that the raw rollouts are
        always saved, and the transcripts can be extracted from them. So this is mostly
        for convenience (and comes with a small processing overhead).
    """

    save_transcripts: bool = True


@register_parameter_class
@dataclass
class PureTextEiParameters(SubParameters):
    """Additional parameters for the Expert Iteration (EI) trainer.

    Parameters
    ----------
    reward_threshold : float
        The threshold on the reward for a rollout to be added to the fine-tuning
        dataset.
    run_test_loop : bool
        Whether to run the test loop after training.
    """

    reward_threshold: float = 0.9

    run_test_loop: bool = False
