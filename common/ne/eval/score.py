from dataclasses import dataclass

import torch
from torchrl.data import TensorSpec
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv

from common.ne.eval.base import BaseEval
from common.ne.popu.base import BasePopu


@dataclass
class ScoreEvalConfig:
    env_name: str
    max_steps: int = 50
    num_workers: int = "${popu.config.size}"


class ScoreEval(BaseEval):
    def __init__(self: "ScoreEval", config: ScoreEvalConfig) -> None:
        self.config = config
        env_name = (
            config.env_name
        )  # Capture locally to avoid circular reference
        make_env = EnvCreator(lambda: GymEnv(env_name))
        self.env = ParallelEnv(config.num_workers, make_env)

    def retrieve_num_inputs_outputs(self: "ScoreEval") -> tuple[int, int]:
        return (
            self.env.observation_spec["observation"].shape[-1],
            self.env.action_spec.shape[-1],
        )

    def retrieve_input_output_specs(
        self: "ScoreEval",
    ) -> tuple[TensorSpec, TensorSpec]:
        return (
            self.env.observation_spec["observation"],
            self.env.action_spec,
        )

    def __call__(self: "ScoreEval", population: BasePopu) -> torch.Tensor:
        num_envs = self.env.num_workers
        fitness_scores = torch.zeros(num_envs)
        env_done = torch.zeros(num_envs, dtype=torch.bool)
        population.nets.reset()  # Initialize hidden states for recurrent networks
        x = self.env.reset()
        for step in range(self.config.max_steps):
            x["action"] = population(x["observation"])
            x = self.env.step(x)["next"]
            # Only accumulate reward for envs that haven't terminated
            reward = x["reward"].squeeze()
            fitness_scores += reward * (~env_done).float()
            env_done = env_done | x["done"].squeeze()
        return fitness_scores
