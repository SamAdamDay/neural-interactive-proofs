"""Parameters for the various ML trainers."""

from typing import Optional, Literal, Annotated
from dataclasses import dataclass

from nip.parameters.parameters_base import SubParameters, register_parameter_class
from nip.parameters.types import (
    PpoLossType,
    SpgVariantType,
    IhvpVariantType,
    TestSchemeType,
)
from nip.parameters.base_run import BaseRunPreserve
from nip.parameters.agents import LrFactors


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
        The number of iterations to run the test for. In each iteration we sample
        `frames_per_batch` frames, as in training.
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
    num_test_iterations: Annotated[int, BaseRunPreserve("rerun_tests")] = 10


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
    loss_type: PpoLossType = "clip"
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


@register_parameter_class
@dataclass
class SpgParameters(SubParameters):
    """Additional parameters for SPG :cite:p:`Fiez2020` and its variants.

    Parameters
    ----------
    variant : SpgVariantType
        The variant of SPG to use.
    stackelberg_sequence : tuple[tuple[str, ...]], optional
        The sequence of agents to use in the Stackelberg game. The leaders first then
        their respective followers, and so forth. If `None`, the sequence is determined
        automatically based on the protocol.
    additional_lola_term : bool
        Whether to add an additional term to the SPG loss to make it equivalent to the later version of LOLA (first introduced implicitly in LOLA-DICE) as opposed to the original version.
    sos_scaling_factor: float
        The SOS scaling factor (between 0 and 1), used with Stable Opponent Shaping.
    sos_threshold_factor: float
        The SOS threshold factor (between 0 and 1), used with Stable Opponent Shaping.
    ihvp_variant : IhvpVariantType
        The variant of IHVP to use.
    ihvp_num_iterations : int
        The number of iterations to use in the IHVP approximation.
    ihvp_rank : int
        The rank of the approximation to use in the IHVP approximation.
    ihvp_rho : float
        The damping factor to use in the IHVP approximation.
    """

    variant: SpgVariantType = "psos"
    stackelberg_sequence: Optional[tuple[tuple[str, ...]]] = None
    additional_lola_term: bool = True
    sos_scaling_factor: float = 0.5
    sos_threshold_factor: float = 0.1

    # IHVP
    ihvp_variant: IhvpVariantType = "nystrom"
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
    fine_tune_on_all_previous_rollouts : bool
        Whether to fine-tune the agents on the rollouts from all iterations so far. If
        `False`, only the rollouts from the current iteration are used.
    verifier_guess_replacement_proportion : float
        When fine-tuning on the rollouts, replace the verifier's guess with the true
        label for this proportion of the rollouts. This only changes the last message of
        the verifier, and leaves the rest of the transcript unchanged.
    verifier_guess_replacement_annealing : Literal["none", "linear", "exponential"]
        The annealing schedule for the proportion of rollouts where the verifier's guess
        is replaced. Possible values are:

        - "none": No annealing.
        - "linear": Linear annealing with rate
          `verifier_guess_replacement_annealing_rate`.
        - "exponential": Exponential annealing with base
          `1-verifier_guess_replacement_annealing_rate`.

    verifier_guess_replacement_annealing_rate : float
        The rate of annealing for the proportion of rollouts where the verifier's guess
        is replaced.
    save_transcripts : bool
        Whether to save the transcripts of the rollouts. Note that the raw rollouts are
        always saved, and the transcripts can be extracted from them. So this is mostly
        for convenience (and comes with a small processing overhead).
    transcript_format : Literal["json", "yaml"]
        The format to save the transcripts in.
    test_scheme : TestSchemeType
        When to run the test loop during training. See `TestSchemeType` for options.
    test_on_whole_dataset : bool
        Whether to run the test loop on the whole dataset or only on a single
        iteration-worth of rollouts.
    test_every_iteration : bool
        Whether to run the test loop after every iteration. If `False`, the test loop is
        only run after training is complete.
    """

    fine_tune_on_all_previous_rollouts: bool = False

    verifier_guess_replacement_proportion: float = 0.0
    verifier_guess_replacement_annealing: Literal["none", "linear", "exponential"] = (
        "none"
    )
    verifier_guess_replacement_annealing_rate: float = 0.1

    save_transcripts: bool = True
    transcript_format: Literal["json", "yaml"] = "yaml"

    test_scheme: Annotated[TestSchemeType, BaseRunPreserve("rerun_tests")] = "none"
    test_on_whole_dataset: Annotated[bool, BaseRunPreserve("rerun_tests")] = True


@register_parameter_class
@dataclass
class PureTextEiParameters(SubParameters):
    """Additional parameters for the Expert Iteration (EI) trainer.

    See :cite:t:`Anthony2017` for more information on EI.

    Parameters
    ----------
    rollout_selection_method : Literal["threshold", "weighted_sampling"]
        The method to use for selecting rollouts for fine-tuning. Possible values are:

        - "threshold": Rollouts are selected if their reward is above a certain
          threshold.
        - "weighted_sampling": Rollouts are selected with a probability proportional to
          their reward.

    reward_threshold : float
        When using the threshold method, the threshold on the reward for a rollout to be
        added to the fine-tuning dataset.
    weighting_sample_size_factor : float
        When using the weighted sampling method, the number of rollouts to sample is
        computed as this factor times the number of rollouts.
    weighting_minimum : float | None
        When using the weighted sampling method, all rewards below this value are
        assigned this value before being used as weights. If `None`, no minimum is
        applied.
    weighting_use_replacement : bool
        Whether to sample with replacement when using the weighted sampling method.
    weighting_epsilon : float
        When using the weighted sampling method, this value, divided by the number of
        rollouts, is added to the normalised weights, which are then normalised again.
        This can be used to prevent the probabilities from becoming zero.
    """

    rollout_selection_method: Literal["threshold", "weighted_sampling"] = "threshold"
    reward_threshold: float = 0.9
    weighting_sample_size_factor: float = 0.5
    weighting_minimum: Optional[float] = None
    weighting_use_replacement: bool = True
    weighting_epsilon: float = 0.01


@register_parameter_class
@dataclass
class PureTextMaltParameters(SubParameters):
    """Additional parameters for Multi-Agent LLM Training (MALT) :cite:p:`Motwani2024`.

    Parameters
    ----------
    num_responses_per_timestep : int
        The number of responses to sample from the agents at each timestep. This yields
        a tree of size at most `num_responses_per_timestep ** max_message_rounds`.
    """

    num_responses_per_timestep: int = 2
