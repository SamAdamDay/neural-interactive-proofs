"""Dummy image classification test script, for debugging purposes.

This script runs the image classification scenario with very basic agents, and is intended
as a way to quickly test that the scenario is working as expected, without having to
run a full experiment.
"""

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import torch

from nip import (
    HyperParameters,
    AgentsParameters,
    ImageClassificationAgentParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    ReinforceParameters,
    CommonProtocolParameters,
    NipProtocolParameters,
    SpgParameters,
    ImageClassificationParameters,
    InteractionProtocolType,
    run_experiment,
)
from nip.utils.env import get_env_var


def run(cmd_args: Namespace):
    """Run the dummy image classification test.

    Parameters
    ----------
    cmd_args : Namespace
        The command-line arguments.
    """

    if cmd_args.use_cpu or not torch.cuda.is_available():
        print("Using CPU")  # noqa: T201
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Make sure W&B doesn't print anything
    os.environ["WANDB_SILENT"] = "true"

    # Create the parameters object
    interaction_protocol: InteractionProtocolType = "nip"
    hyper_params = HyperParameters(
        scenario="image_classification",
        trainer="vanilla_ppo",
        dataset="cifar10",
        agents=AgentsParameters(
            _default=ImageClassificationAgentParameters(
                building_block_type="conv2d",
                d_latent_pixel_selector=1,
                d_decider=1,
                num_decider_layers=1,
                d_value=1,
                num_value_layers=1,
                num_blocks_per_group=1,
                use_manual_architecture=False,
                agent_lr_factor={"actor": 1.0, "critic": 1.0},
                use_orthogonal_initialisation=False,
            )
        ),
        image_classification=ImageClassificationParameters(
            num_block_groups=1,
            initial_num_channels=1,
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
            variant="psos",
        ),
        ppo=CommonPpoParameters(
            loss_type="clip",
            normalize_advantage=True,
        ),
        reinforce=ReinforceParameters(
            use_advantage_and_critic=False,
        ),
        interaction_protocol=interaction_protocol,
        protocol_common=CommonProtocolParameters(
            shared_reward=True,
            force_guess=None,
        ),
        nip_protocol=NipProtocolParameters(
            min_message_rounds=1,
        ),
        pretrain_agents=False,
        include_linear_message_space=False,
    )

    # Train and test the agents
    if cmd_args.use_wandb and cmd_args.run_id != "" and cmd_args.run_id is not None:
        run_id = cmd_args.run_id
    else:
        run_id = None
    run_experiment(
        hyper_params,
        device=device,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=run_id,
        allow_auto_generated_run_id=True,
        print_wandb_run_url=True,
    )


parser = ArgumentParser(
    description="Run a dummy image classification test",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--ignore-cache", action="store_true")
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
    default=get_env_var("WANDB_PROJECT", ""),
)
parser.add_argument(
    "--wandb-entity",
    type=str,
    help="The name of the W&B entity to use",
    default=get_env_var("WANDB_ENTITY", ""),
)

if __name__ == "__main__":

    args = parser.parse_args()

    run(args)
