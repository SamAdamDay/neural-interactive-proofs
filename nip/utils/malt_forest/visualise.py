"""Utilities for visualising the MALT forest of trees."""

from typing import Optional, Self, Literal
import json

from numpy.typing import NDArray

from wandb import Api as WandbApi

from jaxtyping import Int, Float

import yaml

import nip.run  # Importing this to ensure that environment classes are registered
from nip.parameters import HyperParameters
from nip.scenario_base.environment import Environment, PureTextEnvironment
from nip.protocols.protocol_base import ProtocolHandler
from nip.factory import get_scenario_class
from nip.protocols.registry import build_protocol_handler
from nip.experiment_settings import ExperimentSettings
from nip.utils.malt_forest.forest import MaltNode, MaltTree, reconstruct_malt_forest
from nip.utils.checkpoints import load_rollouts, load_run_hyper_parameters
from nip.utils.types import String
from nip.utils.rollouts import get_pretty_pure_text_round_message


class MaltForestVisualiser:
    """Class for visualising the MALT forest of trees.

    Parameters
    ----------
    malt_forest : list[MaltTree]
        The MALT forest to be visualised.
    hyper_params : HyperParameters
        The hyper-parameters of the experiment used to generate the rollouts from which
        the MALT forest was built.
    protocol_handler : ProtocolHandler
        The protocol handler constructed for the hyper-parameters. This will be
        functionally identical to the one used to generate the rollouts from which the
        MALT forest was built.
    environment_class : PureTextEnvironment
        The environment class determined from the hyper-parameters. If instantiated,
        this would be functionally identical to the one used to generate the rollouts
        from which the MALT forest was built.
    """

    def __init__(
        self,
        malt_forest: list[MaltTree],
        hyper_params: HyperParameters,
        protocol_handler: ProtocolHandler,
        environment_class: type[PureTextEnvironment],
    ):
        self.malt_forest = malt_forest
        self.hyper_params = hyper_params
        self.protocol_handler = protocol_handler
        self.environment_class = environment_class

    @classmethod
    def from_wandb_run(
        self,
        run_id: str,
        iteration: int | str,
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        wandb_api: Optional[WandbApi] = None,
    ) -> Self:
        """Build a MaltForestVisualiser by from a wandb run.

        This method downloads the rollouts from the wandb run, builds the MALT forest
        from them, and constructs the visualiser.

        Parameters
        ----------
        run_id : str
            The id of the wandb run to download the rollouts from.
        iteration : int | str
            The experiment iteration to use as the rollouts.
        wandb_project : str
            The project of the wandb run.
        wandb_entity : str, optional
            The entity of the wandb run. If not provided, the default entity will be
            used.
        wandb_api : WandbApi, optional
            The wandb API instance to use. If not provided, a new instance will be
            created.

        Returns
        -------
        malt_forest_visualiser : MaltForestVisualiser
            The MALT forest visualiser constructed from the wandb run.
        """

        if wandb_api is None:
            wandb_api = WandbApi()

        rollouts = load_rollouts(
            run_id=run_id,
            iterations=iteration,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_api=wandb_api,
        )
        hyper_params = load_run_hyper_parameters(
            run_id=run_id,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_api=wandb_api,
        )
        environment_class = get_scenario_class(hyper_params, Environment)
        protocol_handler = build_protocol_handler(hyper_params, ExperimentSettings())
        malt_forest = reconstruct_malt_forest(rollouts)

        return MaltForestVisualiser(
            malt_forest=malt_forest,
            hyper_params=hyper_params,
            protocol_handler=protocol_handler,
            environment_class=environment_class,
        )

    def visualise(self, format: Literal["yaml", "jsonl"] = "yaml") -> str:
        """Visualise the MALT forest in the specified format.

        Parameters
        ----------
        format : Literal["yaml", "jsonl"]
            The format to visualise the MALT forest in.

        Returns
        -------
        malt_forest_visualisation : str
            The MALT forest visualisation in the specified format. Each tree is
            represented as a nested dictionary.
        """
        if format == "yaml":
            return self.as_yaml()
        elif format == "jsonl":
            return self.as_jsonl()
        else:
            raise ValueError(f"Invalid format {format!r}. Must be 'yaml' or 'jsonl'.")

    def as_yaml(self) -> str:
        """Build a YAML representation of the MALT forest.

        Returns
        -------
        malt_yaml_forest : str
            The MALT forest represented as a YAML string. Each tree is represented as a
            nested dictionary.
        """

        malt_dict_forest = self.build_dict_forest()
        malt_yaml_forest = yaml.dump(malt_dict_forest, sort_keys=False)
        return malt_yaml_forest

    def as_jsonl(self) -> str:
        """Build a JSONL representation of the MALT forest.

        A JSONL (JSON Lines) file contains one JSON object per line. See `JSON Lines
        <https://jsonlines.org/>`_.

        Returns
        -------
        malt_jsonl_forest : str
            The MALT forest represented as a JSONL string. Each tree is represented as a
            nested dictionary.
        """

        malt_dict_forest = self.build_dict_forest()
        malt_jsonl_forest = "\n".join(json.dumps(tree) for tree in malt_dict_forest)
        return malt_jsonl_forest

    def build_dict_forest(self) -> list[dict]:
        """Build a dictionary representation of the MALT forest.

        Returns
        -------
        malt_dict_forest : list[dict]
            The MALT forest represented as a list of dictionaries. Each contains the
            whole tree as nested dictionaries.
        """

        malt_dict_forest: list[dict] = []

        for tree in self.malt_forest:
            a_root_env_state = tree[0].env_state
            current_dict_tree = (
                self.environment_class.get_datapoint_from_env_state_as_dict(
                    a_root_env_state
                )
            )
            current_dict_tree["tree"] = [self._build_dict_tree(child) for child in tree]
            malt_dict_forest.append(current_dict_tree)

        return malt_dict_forest

    def _build_dict_tree(self, malt_node: MaltNode) -> dict:
        """Build a dictionary representation of a MALT tree, recursively.

        Parameters
        ----------
        malt_node : MaltNode
            The root node of the MALT tree to be converted to a dictionary.

        Returns
        -------
        malt_dict_tree : dict
            The MALT tree represented as a dictionary. Each node is represented as a
            dictionary with keys for the messages sent, various metadata, and the
            children of the node.
        """

        decision: Int[NDArray, "agent"] = malt_node.env_state["agents", "decision"]
        continuous_decision: Float[NDArray, "agent"] = malt_node.env_state[
            "agents", "continuous_decision"
        ]
        raw_decision: String[NDArray, "agent"] = malt_node.env_state[
            "agents", "raw_decision"
        ]
        message: String[NDArray, "agent channel"] = malt_node.env_state[
            "agents", "message"
        ]

        agent_names = self.protocol_handler.agent_names

        malt_dict_tree = {
            "node_id": malt_node.env_state["_node_id"].item(),
        }

        # Reward info for the current node
        for agent_id, agent_name in enumerate(agent_names):
            malt_dict_tree[agent_name] = {
                "expected_reward": malt_node.env_state["agents", "expected_reward"][
                    agent_id
                ].item()
            }
            if malt_node.env_state["agents", "is_pair_positive"][agent_id]:
                malt_dict_tree[agent_name]["pair"] = "positive"
            elif malt_node.env_state["agents", "is_pair_negative"][agent_id]:
                malt_dict_tree[agent_name]["pair"] = "negative"

        malt_dict_tree = malt_dict_tree | get_pretty_pure_text_round_message(
            protocol_handler=self.protocol_handler,
            decision=decision,
            raw_decision=raw_decision,
            continuous_decision=continuous_decision,
            message=message,
        )

        children = [self._build_dict_tree(child) for child in malt_node.children]
        if len(children) > 0:
            malt_dict_tree["children"] = children

        return malt_dict_tree
