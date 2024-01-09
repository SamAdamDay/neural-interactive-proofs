from torchrl.envs.utils import check_env_specs

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.graph_isomorphism.environment import GraphIsomorphismEnvironment


def test_environment_specs():
    """Test that the environment has the correct specs."""

    scenario_types = [ScenarioType.GRAPH_ISOMORPHISM]
    environment_classes = [GraphIsomorphismEnvironment]
    for scenario_type, environment_class in zip(scenario_types, environment_classes):
        params = Parameters(scenario_type, TrainerType.PPO, "test")
        env = environment_class(params)
        check_env_specs(env)
