"""Dummy graph isomorphism test script, for debugging purposes"""

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import torch

from pvg import (
    Parameters,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    ReinforceParameters,
    ScenarioType,
    TrainerType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    SpgParameters,
    MessageRegressionParameters,
    PpoLossType,
    SpgVariant,
    Guess,
    run_experiment,
    prepare_experiment,
)
from pvg.constants import WANDB_PROJECT, WANDB_ENTITY


def run(cmd_args: Namespace):
    if cmd_args.use_cpu or not torch.cuda.is_available():
        print("Using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Make sure W&B doesn't print anything
    os.environ["WANDB_SILENT"] = "true"

    # Create the parameters object
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="eru10000",
        agents=AgentsParameters(
            verifier=GraphIsomorphismAgentParameters(
                num_gnn_layers=1,
                num_transformer_layers=1,
                use_manual_architecture=False,
                agent_lr_factor={"actor": 1.0, "critic": 1.0},
                ortho_init=False,
            ),
            prover=GraphIsomorphismAgentParameters(
                num_gnn_layers=1,
                num_transformer_layers=1,
                use_manual_architecture=False,
                agent_lr_factor={"actor": 1.0, "critic": 1.0},
                ortho_init=False,
            ),
        ),
        rl=RlTrainerParameters(
            num_iterations=100,
            num_epochs=1,
            lr=0.001,
            anneal_lr=True,
            minibatch_size=4,
            frames_per_batch=16,
            use_shared_body=False,
            num_normalization_steps=10,
        ),
        spg=SpgParameters(
            variant=SpgVariant.PSOS,
            # stackelberg_sequence=combo["stackelberg_sequence"],
            # ihvp_variant=combo["ihvp_variant"],
            # ihvp_num_iterations=combo["ihvp_num_iterations"],
            # ihvp_rank=combo["ihvp_rank"],
            # ihvp_rho=combo["ihvp_rho"],
        ),
        ppo=CommonPpoParameters(
            loss_type=PpoLossType.CLIP,
            normalize_advantage=True,
        ),
        reinforce=ReinforceParameters(
            use_advantage_and_critic=False,
        ),
        protocol_common=CommonProtocolParameters(
            shared_reward=True,
            force_guess=None,
        ),
        pvg_protocol=PvgProtocolParameters(
            min_message_rounds=0,
        ),
        pretrain_agents=False,
        include_linear_message_space=False,
        message_regression=MessageRegressionParameters(
            enabled=True,
            mlp_num_layers=2,
            mlp_hidden_size=64,
            mlp_num_epochs=10,
            mlp_batch_size=512,
            mlp_learning_rate=0.001,
        ),
    )

    # Prepare the experiment if necessary
    if cmd_args.prepare_experiment:
        prepare_experiment(params)

    # Train and test the agents
    if cmd_args.use_wandb and cmd_args.run_id != "" and cmd_args.run_id is not None:
        run_id = cmd_args.run_id
    else:
        run_id = None
    run_experiment(
        params,
        device=device,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=run_id,
        allow_auto_generated_run_id=True,
        print_wandb_run_url=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run a dummy graph isomorphism test",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Whether to use W&B to log the experiment",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="The ID of the W&B run. By default, a random ID is auto-generated",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="The name of the W&B project to use",
        default=WANDB_PROJECT,
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="The name of the W&B entity to use",
        default=WANDB_ENTITY,
    )
    parser.add_argument(
        "--prepare-experiment",
        action="store_true",
        help="Whether to prepare the experiment before running it",
    )
    args = parser.parse_args()

    run(args)
