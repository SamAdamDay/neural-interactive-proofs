from argparse import Namespace
import os
import json
from pathlib import Path

import numpy as np

import torch

from pvg.utils.experiments import HyperparameterExperiment
from pvg.extra.test_solo_gi_agents import train_and_test_solo_gi_agents
from pvg.constants import GI_SOLO_AGENTS_RESULTS_DATA_DIR

TEST_SIZE = 0.2

param_grid = dict(
    dataset_name=["er10000"],
    d_gnn=[8],
    d_decider=[16],
    num_epochs=[1000],
    batch_size=[1024],
    learning_rate=[0.003],
    scheduler_patience=[2000],
    scheduler_factor=[0.5],
    freeze_encoder=[True, False],
    seed=[8144, 820, 4173, 3992],
)


def experiment_fn(combo: dict, run_id: str, cmd_args: Namespace):
    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Train and test the agents to get the results
    _, _, results = train_and_test_solo_gi_agents(
        dataset_name=combo["dataset_name"],
        d_gnn=combo["d_gnn"],
        d_decider=combo["d_decider"],
        test_size=TEST_SIZE,
        num_epochs=combo["num_epochs"],
        batch_size=combo["batch_size"],
        learning_rate=combo["learning_rate"],
        scheduler_patience=combo["scheduler_patience"],
        scheduler_factor=combo["scheduler_factor"],
        freeze_encoder=combo["freeze_encoder"],
        seed=combo["seed"],
        device=device,
    )

    # Convert any numpy arrays to lists
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    results["run_id"] = run_id
    results["combo"] = combo

    # Save the results
    print(f"Saving results")
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
    "--run-infix", type=str, help="The string to add in the middle of the run ID", default="a"
)
experiment.parser.add_argument(
    "--gpu-num", type=int, help="The (0-based) number of the GPU to use", default=0
)
experiment.run()
