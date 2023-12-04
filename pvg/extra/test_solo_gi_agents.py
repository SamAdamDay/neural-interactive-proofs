from abc import ABC
from typing import Optional, Callable
from functools import partial
import logging

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from jaxtyping import Float, Bool

from tqdm import tqdm

import wandb

from pvg.graph_isomorphism import (
    GraphIsomorphismAgent,
    GraphIsomorphismDataset,
    GraphIsomorphismData,
)
from pvg.parameters import Parameters, GraphIsomorphismParameters


class ScoreToBitTransform(BaseTransform):
    """A transform that converts the score to a bit indicating isomorphism."""

    def __call__(self, data):
        for store in data.node_stores:
            store.y = (store.wl_score == -1).long()
        return data


class GraphIsomorphismSoloAgent(GraphIsomorphismAgent, ABC):
    """A base class for an agent that tries to solve the graph isomorphism task solo."""

    def _build_model(
        self,
        num_layers: int,
        d_gnn: int,
        d_gin_mlp: int,
        d_decider: int,
        num_heads: int,
        noise_sigma: float,
        use_batch_norm: bool,
        use_pair_invariant_pooling: bool,
    ) -> nn.Module:
        # Build up the GNN module
        self.gnn, self.attention = self._build_gnn_and_attention(
            d_input=1,
            d_gnn=d_gnn,
            d_gin_mlp=d_gin_mlp,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Create the Gaussian noise layer
        self.global_pooling = self._build_global_pooling(
            d_gnn=d_gnn,
            d_decider=d_decider,
            noise_sigma=noise_sigma,
            use_batch_norm=use_batch_norm,
            use_invariantizer=use_pair_invariant_pooling,
        )

        # Build the decider, which decides whether the graphs are isomorphic
        self.decider = self._build_decider(
            d_decider=d_decider,
            d_out=2,
        )

    def forward(
        self,
        data: GraphIsomorphismData | GeometricBatch,
        output_callback: Optional[
            Callable[
                [
                    Float[Tensor, "batch 2 max_nodes d_gnn"],
                    Float[Tensor, "batch 2 max_nodes d_gnn"],
                    Float[Tensor, "batch 2 d_decider"],
                    Bool[Tensor, "batch max_nodes_a+max_nodes_b"],
                    GraphIsomorphismData | GeometricBatch,
                ],
                None,
            ]
        ] = None,
    ) -> Float[Tensor, "batch 2"]:
        gnn_output, attention_output, node_mask = self._run_gnn_and_attention(data)
        pooled_output = self.global_pooling(gnn_output)
        if output_callback is not None:
            output_callback(
                gnn_output, attention_output, pooled_output, node_mask, data
            )
        decider_logits = self.decider(pooled_output)
        return decider_logits

    def to(self, device: str | torch.device):
        self.gnn.to(device)
        self.attention.to(device)
        self.global_pooling.to(device)
        self.global_pooling[-1].to(device)
        self.decider.to(device)
        return self


class GraphIsomorphismSoloProver(GraphIsomorphismSoloAgent):
    """A class for a prover that tries to solve the graph isomorphism task solo."""

    def __init__(
        self,
        params: Parameters,
        d_decider: int,
        device: str | torch.device,
    ):
        super().__init__(params, device)
        self._build_model(
            num_layers=params.graph_isomorphism.prover_num_layers,
            d_gnn=params.graph_isomorphism.prover_d_gnn,
            d_gin_mlp=params.graph_isomorphism.prover_d_gin_mlp,
            d_decider=d_decider,
            num_heads=params.graph_isomorphism.prover_num_heads,
            noise_sigma=params.graph_isomorphism.prover_noise_sigma,
            use_batch_norm=params.graph_isomorphism.prover_use_batch_norm,
            use_pair_invariant_pooling=params.graph_isomorphism.prover_pair_invariant_pooling,
        )


class GraphIsomorphismSoloVerifier(GraphIsomorphismSoloAgent):
    """A class for a verifier that tries to solve the graph isomorphism task solo."""

    def __init__(
        self,
        params: Parameters,
        d_decider: int,
        device: str | torch.device,
    ):
        super().__init__(params, device)
        self._build_model(
            num_layers=params.graph_isomorphism.verifier_num_layers,
            d_gnn=params.graph_isomorphism.verifier_d_gnn,
            d_gin_mlp=params.graph_isomorphism.verifier_d_gin_mlp,
            d_decider=d_decider,
            num_heads=params.graph_isomorphism.verifier_num_heads,
            noise_sigma=params.graph_isomorphism.verifier_noise_sigma,
            use_batch_norm=params.graph_isomorphism.verifier_use_batch_norm,
            use_pair_invariant_pooling=params.graph_isomorphism.verifier_pair_invariant_pooling,
        )


def train_and_test_solo_gi_agents(
    dataset_name: str,
    d_gnn: int,
    d_decider: int,
    use_batch_norm: bool,
    noise_sigma: float,
    use_pair_invariant_pooling: bool,
    test_size: float,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_scheduler: str | None,
    learning_rate_scheduler_args: dict,
    freeze_encoder: bool,
    encoder_lr_factor: float,
    prover_num_layers: int,
    verifier_num_layers: int,
    seed: int,
    device: str | torch.device,
    wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None,
    tqdm_func: Callable = tqdm,
    logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
) -> tuple[nn.Module, nn.Module, dict]:
    """Train and test solo GI agents.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to use.
    d_gnn : int
        The dimensionality of the GNN hidden layers and of the attention embedding.
    d_decider : int
        The dimensionality of the final MLP hidden layers.
    use_batch_norm : bool
        Whether to use batch normalization in the global pooling layer.
    noise_sigma : float
        The relative standard deviation of the Gaussian noise added to the graph-level
        representations.
    use_pair_invariant_pooling : bool
        Whether to use pair-invariant pooling in the global pooling layer. This makes
        the graph-level representations invariant to the order of the graphs in the
        pair.
    dataset : GraphIsomorphismDataset
        The training dataset.
    test_size : float
        The size of the test set as a fraction of the dataset size.
    num_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size.
    learning_rate : float
        The learning rate.
    learning_rate_scheduler : "ReduceLROnPlateau" | "CyclicLR" | None
        The learning rate scheduler to use.
    learning_rate_scheduler_args : dict
        The arguments to pass to the learning rate scheduler.
    freeze_encoder : bool
        Whether to freeze the GNN and attention modules.
    encoder_lr_factor : float
        The factor by which to scale the learning rate of the encoder. Only makes sense
        when freeze_encoder is False.
    prover_num_layers : int
        The number of layers in the prover's GNN.
    verifier_num_layers : int
        The number of layers in the verifier's GNN.
    seed : int
        The random seed.
    device : str | torch.device
        The device to use.
    wandb_run : wandb.wandb_sdk.wandb_run.Run, optional
        The W&B run to log to, if any.
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    logger : logging.Logger | logging.LoggerAdapter, optional
        The logger to log to. If None, creates a new logger.

    Returns
    -------
    prover : nn.Module
        The trained prover.
    verifier : nn.Module
        The trained verifier.
    results : dict
        A dictionary containing the training losses and the training and testing
        accuracies.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_generator = torch.Generator().manual_seed(seed)

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading dataset and agents...")

    # Create parameters for the experiment
    params = Parameters(
        scenario="graph_isomorphism",
        trainer="test",
        dataset=dataset_name,
        max_message_rounds=1,
        graph_isomorphism=GraphIsomorphismParameters(
            prover_num_layers=prover_num_layers,
            prover_d_gnn=d_gnn,
            prover_use_batch_norm=use_batch_norm,
            prover_noise_sigma=noise_sigma,
            prover_pair_invariant_pooling=use_pair_invariant_pooling,
            verifier_num_layers=verifier_num_layers,
            verifier_d_gnn=d_gnn,
            verifier_use_batch_norm=use_batch_norm,
            verifier_noise_sigma=noise_sigma,
            verifier_pair_invariant_pooling=use_pair_invariant_pooling,
        ),
    )

    dataset = GraphIsomorphismDataset(params, transform=ScoreToBitTransform())
    train_dataset, test_dataset = random_split(dataset, (1 - test_size, test_size))

    # Create the prover and verifier
    prover = GraphIsomorphismSoloProver(params, d_decider, device)
    verifier = GraphIsomorphismSoloVerifier(params, d_decider, device)

    def encoder_parameter_selector(name: str) -> bool:
        return (
            name.startswith("gnn")
            or name.startswith("attention")
            or name.startswith("global_pooling")
        )

    # Divide the parameters into encoder and non-encoder parameters
    prover_encoder_params = (
        param
        for name, param in prover.named_parameters()
        if encoder_parameter_selector(name)
    )
    verifier_encoder_params = (
        param
        for name, param in verifier.named_parameters()
        if encoder_parameter_selector(name)
    )
    prover_non_encoder_params = (
        param
        for name, param in prover.named_parameters()
        if not encoder_parameter_selector(name)
    )
    verifier_non_encoder_params = (
        param
        for name, param in verifier.named_parameters()
        if not encoder_parameter_selector(name)
    )

    # Freeze the encoder if requested
    if freeze_encoder:
        for param in prover_encoder_params:
            param.requires_grad = False
        for param in verifier_encoder_params:
            param.requires_grad = False
        prover_param_dicts = [
            {"params": prover_non_encoder_params, "lr": learning_rate}
        ]
        verifier_param_dicts = [
            {
                "params": verifier_non_encoder_params,
                "lr": learning_rate,
            }
        ]
    # Otherwise, scale the learning rate of the encoder
    else:
        prover_param_dicts = [
            {"params": prover_encoder_params, "lr": learning_rate * encoder_lr_factor},
            {"params": prover_non_encoder_params, "lr": learning_rate},
        ]
        verifier_param_dicts = [
            {
                "params": verifier_encoder_params,
                "lr": learning_rate * encoder_lr_factor,
            },
            {
                "params": verifier_non_encoder_params,
                "lr": learning_rate,
            },
        ]

    # Create the optimizers and schedulers
    optimizer_prover = Adam(prover_param_dicts)
    optimizer_verifier = Adam(verifier_param_dicts)
    if learning_rate_scheduler == "ReduceLROnPlateau":
        scheduler_prover = ReduceLROnPlateau(
            optimizer_prover,
            verbose=False,
            **learning_rate_scheduler_args,
        )
        scheduler_verifier = ReduceLROnPlateau(
            optimizer_verifier,
            verbose=False,
            **learning_rate_scheduler_args,
        )
    elif learning_rate_scheduler == "CyclicLR":
        scheduler_prover = torch.optim.lr_scheduler.CyclicLR(
            optimizer_prover,
            verbose=False,
            **learning_rate_scheduler_args,
        )
        scheduler_verifier = torch.optim.lr_scheduler.CyclicLR(
            optimizer_verifier,
            verbose=False,
            **learning_rate_scheduler_args,
        )
    elif learning_rate_scheduler is None:
        scheduler_prover = None
        scheduler_verifier = None
    else:
        raise ValueError(f"Unknown learning rate scheduler {learning_rate_scheduler}.")

    # Create the data loaders
    test_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch_generator,
        follow_batch=["x_a", "x_b"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, follow_batch=["x_a", "x_b"]
    )

    # Define the training step
    def train_step(
        model: GraphIsomorphismSoloAgent,
        optimizer,
        data: GraphIsomorphismData,
    ) -> tuple[Tensor, float, float]:
        model.train()
        optimizer.zero_grad()

        # Function to compute the encoder equality accuracy
        encoder_eq_accuracy = torch.empty(data.y.shape[0], dtype=bool, device=device)

        def compute_encoder_eq_accuracy(
            gnn_output,
            attention_output,
            pooled_output: Float[Tensor, "batch 2 d_decider"],
            node_mask,
            data,
            encoder_eq_accuracy,
        ):
            if use_pair_invariant_pooling:
                close = torch.isclose(
                    pooled_output[:, 1],
                    torch.zeros_like(pooled_output[:, 1]),
                    rtol=1e-5,
                    atol=1e-5,
                )
            else:
                close = torch.isclose(pooled_output[:, 0], pooled_output[:, 1])
            encoder_eq_accuracy[:] = close.all(dim=-1) == data.y.bool()

        # Run the model and compute the loss and encoder equality accuracy
        pred = model(
            data,
            output_callback=partial(
                compute_encoder_eq_accuracy,
                encoder_eq_accuracy=encoder_eq_accuracy,
            ),
        )
        loss = F.cross_entropy(pred, data.y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = (pred.argmax(dim=1) == data.y).float().mean().item()

        return loss, accuracy, encoder_eq_accuracy.float().mean().item()

    prover.to(device)
    verifier.to(device)

    train_losses_prover = np.empty(num_epochs)
    train_accuracies_prover = np.empty(num_epochs)
    train_encoder_eq_accs_prover = np.empty(num_epochs)
    train_losses_verifier = np.empty(num_epochs)
    train_accuracies_verifier = np.empty(num_epochs)
    train_encoder_eq_accs_verifier = np.empty(num_epochs)

    # Train the prover and verifier
    with tqdm_func(total=num_epochs, desc="Training") as pbar:
        for epoch in range(num_epochs):
            total_loss_prover = 0
            total_accuracy_prover = 0
            total_encoder_eq_acc_prover = 0
            total_loss_verifier = 0
            total_accuracy_verifier = 0
            total_encoder_eq_acc_verifier = 0

            for data in test_loader:
                data = data.to(device)

                # Train the prover on the batch
                loss_prover, accuracy, encoder_eq_accuracy = train_step(
                    prover, optimizer_prover, data
                )
                total_loss_prover += loss_prover.item()
                total_accuracy_prover += accuracy
                total_encoder_eq_acc_prover += encoder_eq_accuracy

                # Train the verifier on the batch
                loss_verifier, accuracy, encoder_eq_accuracy = train_step(
                    verifier, optimizer_verifier, data
                )
                total_loss_verifier += loss_verifier.item()
                total_accuracy_verifier += accuracy
                total_encoder_eq_acc_verifier += encoder_eq_accuracy

                # Update the learning rate per batch if not using ReduceLROnPlateau
                if (
                    learning_rate_scheduler is not None
                    and learning_rate_scheduler != "ReduceLROnPlateau"
                ):
                    scheduler_prover.step()
                    scheduler_verifier.step()

            # Update the learning rate per epoch if using ReduceLROnPlateau
            if learning_rate_scheduler == "ReduceLROnPlateau":
                scheduler_prover.step(loss_prover)
                scheduler_verifier.step(loss_verifier)

            # Log the results
            train_losses_prover[epoch] = total_loss_prover / len(test_loader)
            train_accuracies_prover[epoch] = total_accuracy_prover / len(test_loader)
            train_encoder_eq_accs_prover[epoch] = total_encoder_eq_acc_prover / len(
                test_loader
            )
            train_losses_verifier[epoch] = total_loss_verifier / len(test_loader)
            train_accuracies_verifier[epoch] = total_accuracy_verifier / len(
                test_loader
            )
            train_encoder_eq_accs_verifier[epoch] = total_encoder_eq_acc_verifier / len(
                test_loader
            )

            # Log to W&B if using
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train_loss_prover": train_losses_prover[epoch],
                        "train_accuracy_prover": train_accuracies_prover[epoch],
                        "train_encoder_eq_accuracy_prover": train_encoder_eq_accs_prover[
                            epoch
                        ],
                        "train_loss_verifier": train_losses_verifier[epoch],
                        "train_accuracy_verifier": train_accuracies_verifier[epoch],
                        "train_encoder_eq_accuracy_verifier": train_encoder_eq_accs_verifier[
                            epoch
                        ],
                        "optimizer_lr_prover": optimizer_prover.param_groups[0]["lr"],
                        "optimizer_lr_verifier": optimizer_verifier.param_groups[0][
                            "lr"
                        ],
                    },
                    step=epoch,
                )

            # Update the progress bar
            pbar.update(1)

    # Define the testing step
    def test_step(
        model: GraphIsomorphismSoloAgent, data: GraphIsomorphismData
    ) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            pred = model(data)
            loss = F.cross_entropy(pred, data.y)
            accuracy = (pred.argmax(dim=1) == data.y).float().mean().item()
        return loss.item(), accuracy

    test_loss_prover = 0
    test_accuracy_prover = 0
    test_loss_verifier = 0
    test_accuracy_verifier = 0

    # Test the prover and verifier
    logger.info("Testing...")
    for data in test_loader:
        data = data.to(device)
        loss_verifier, accuracy = test_step(prover, data)
        test_loss_prover += loss_verifier
        test_accuracy_prover += accuracy
        loss_verifier, accuracy = test_step(verifier, data)
        test_loss_verifier += loss_verifier
        test_accuracy_verifier += accuracy
    test_loss_prover = test_loss_prover / len(test_loader)
    test_accuracy_prover = test_accuracy_prover / len(test_loader)
    test_loss_verifier = test_loss_verifier / len(test_loader)
    test_accuracy_verifier = test_accuracy_verifier / len(test_loader)

    # Record the final results with W&B if using
    if wandb_run is not None:
        wandb_run.log(
            {
                "test_loss_prover": test_loss_prover,
                "test_accuracy_prover": test_accuracy_prover,
                "test_loss_verifier": test_loss_verifier,
                "test_accuracy_verifier": test_accuracy_verifier,
            }
        )

    results = {
        "train_losses_prover": train_losses_prover,
        "train_accuracies_prover": train_accuracies_prover,
        "train_encoder_eq_accuracies_prover": train_encoder_eq_accs_prover,
        "train_losses_verifier": train_losses_verifier,
        "train_accuracies_verifier": train_accuracies_verifier,
        "train_encoder_eq_accuracies_verifier": train_encoder_eq_accs_verifier,
        "test_loss_prover": test_loss_prover,
        "test_accuracy_prover": test_accuracy_prover,
        "test_loss_verifier": test_loss_verifier,
        "test_accuracy_verifier": test_accuracy_verifier,
    }

    return prover, verifier, results
