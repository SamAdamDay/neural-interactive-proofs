from argparse import Namespace
import os
import json
from pathlib import Path

import numpy as np

import torch

import wandb

from pvg.utils.experiments import HyperparameterExperiment
from pvg.extra.test_solo_gi_agents import train_and_test_solo_gi_agents
from pvg.constants import GI_SOLO_AGENTS_RESULTS_DATA_DIR

TEST_SIZE = 0.2

param_grid = dict(
    dataset_name=["er10000"],
    d_gnn=[16],
    d_decider=[16],
    use_batch_norm=[True],
    use_pair_invariant_pooling=[True],
    noise_sigma=[0.0],
    num_epochs=[500],
    batch_size=[256],
    learning_rate=[0.003],
    learning_rate_scheduler=[None],
    scheduler_factor=[0.5],
    freeze_encoder=[True],
    encoder_lr_factor=[0.1],
    prover_num_layers=[5],
    verifier_num_layers=[2],
    seed=[8144, 820, 4173, 3992],
)


def experiment_fn(combo: dict, run_id: str, cmd_args: Namespace):
    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Set up W&B
    use_wandb = cmd_args.wandb_project != ""
    if use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
        wandb_run = wandb.init(
            project=cmd_args.wandb_project, name=run_id, tags=wandb_tags
        )
        wandb_run.config.update(combo)

    # Train and test the agents to get the results
    learning_rate_scheduler_args = {
        arg[len("scheduler_") :]: value
        for arg, value in combo.items()
        if arg.startswith("scheduler_")
    }
    _, _, results = train_and_test_solo_gi_agents(
        dataset_name=combo["dataset_name"],
        d_gnn=combo["d_gnn"],
        d_decider=combo["d_decider"],
        use_batch_norm=combo["use_batch_norm"],
        noise_sigma=combo["noise_sigma"],
        use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
        test_size=TEST_SIZE,
        num_epochs=combo["num_epochs"],
        batch_size=combo["batch_size"],
        learning_rate=combo["learning_rate"],
        learning_rate_scheduler=combo["learning_rate_scheduler"],
        learning_rate_scheduler_args=learning_rate_scheduler_args,
        freeze_encoder=combo["freeze_encoder"],
        encoder_lr_factor=combo["encoder_lr_factor"],
        prover_num_layers=combo["prover_num_layers"],
        verifier_num_layers=combo["verifier_num_layers"],
        seed=combo["seed"],
        wandb_run=wandb_run if use_wandb else None,
        device=device,
    )

    if use_wandb:
        wandb_run.finish()
    else:
        # Convert any numpy arrays to lists
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()

        results["run_id"] = run_id
        results["combo"] = combo

        # Save the results
        print(f"Saving results locally...")
        filename = f"{run_id}.json"
        filepath = os.path.join(GI_SOLO_AGENTS_RESULTS_DATA_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(results, f)


def run_id_fn(combo_index: int, cmd_args: Namespace):
    return f"test_solo_gi_agents_{cmd_args.run_infix}_{combo_index}"


# Make sure the results directory exists
Path(GI_SOLO_AGENTS_RESULTS_DATA_DIR).mkdir(parents=True, exist_ok=True)

experiment = HyperparameterExperiment(
    param_grid=param_grid,
    experiment_fn=experiment_fn,
    run_id_fn=run_id_fn,
    experiment_name="TEST_SOLO_GI_AGENTS",
)
experiment.parser.add_argument(
    "--run-infix",
    type=str,
    help="The string to add in the middle of the run ID",
    default="a",
)
experiment.parser.add_argument(
    "--gpu-num", type=int, help="The (0-based) number of the GPU to use", default=0
)
experiment.parser.add_argument(
    "--wandb-project",
    type=str,
    help="The name of the W&B project to use. If not set saves the results locally",
    default="",
)
experiment.parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="An optional tag for the W&B run",
)
experiment.run()
