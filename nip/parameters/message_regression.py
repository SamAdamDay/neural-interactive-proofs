"""Parameters for doing regression analysis on the messages."""

from dataclasses import dataclass, field

from nip.parameters.parameters_base import SubParameters, register_parameter_class
from nip.parameters.types import MessageRegressionMethodType, ActivationType


@register_parameter_class
@dataclass
class MessageRegressionParameters(SubParameters):
    """Additional parameters for doing regression analysis on the messages.

    This allows doing testing to what extent the label can be inferred purely from the
    first message sent an agent, without considering the other data.

    This is useful in the shared reward setting to see if the prover is simply
    communicating the label to the verifier. If this is the case, we expect to be able
    to recover it just from the messages.

    Parameters
    ----------
    enabled : bool
        Whether to do regression analysis on the messages.
    agents : list of str
        The agents to do regression analysis on. If None, do it on all agents.
    regression_method : MessageRegressionMethodType
        The method to use to do the regression.
    test_size : float
        The proportion of the message data to use for testing.
    reset_on_fit : bool
        Whether to reset the regressor on each fit (i.e. to fit it from scratch).
    mlp_num_layers : int
        The number of layers in the MLP, when using.
    mlp_hidden_size : int
        The hidden size of the MLP, when using.
    mlp_activation : ActivationType
        The activation function to use in the MLP, when using.
    mlp_num_epochs : int
        The number of epochs to train the MLP for, when using.
    mlp_batch_size : int
        The batch size to use when training the MLP, when using.
    mlp_learning_rate : float
        The learning rate to use when training the MLP, when using.
    """

    enabled: bool = False
    agents: list[str] | None = None

    regression_method: MessageRegressionMethodType = "mlp"

    test_size: float = 0.2

    reset_on_fit: bool = True

    mlp_num_layers: int = 2
    mlp_hidden_size: int = 64
    mlp_activation: ActivationType = "relu"
    mlp_num_epochs: int = 10
    mlp_batch_size: int = 512
    mlp_learning_rate: float = 0.001
