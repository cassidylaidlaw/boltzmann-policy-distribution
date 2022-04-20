from typing import Dict, Optional, Type
import numpy as np
from tqdm import tqdm
from typing_extensions import Literal
import json
import os
import pickle
from gym import spaces

# Sacred setup (must be before rllib imports)
from sacred import Experiment

import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.typing import PolicyID

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

from bpd.training_utils import load_trainer
from bpd.models.overcooked_models import (
    BPDFeaturizeFn,
    OvercookedPPODistributionModel,
)
from bpd.agents.bpd_trainer import BPDTrainer
from bpd.agents.distillation_prediction import (
    DistillationPredictionTrainer,
)
from bpd.envs.overcooked import EpisodeInformation, evaluate
from bpd.agents.bc import BCTrainer
from bpd.agents.gail import GailTrainer

ex = Experiment("evaluate_overcooked")


RunStr = Literal["ppo", "bpd", "bc", "distill", "gail"]
trainer_classes: Dict[RunStr, Type[Trainer]] = {
    "ppo": PPOTrainer,
    "bpd": BPDTrainer,
    "bc": BCTrainer,
    "distill": DistillationPredictionTrainer,
    "gail": GailTrainer,
}

default_policy_ids: Dict[RunStr, PolicyID] = {
    "ppo": "ppo",
    "bpd": "ppo",
    "bc": DEFAULT_POLICY_ID,
    "distill": "ppo_distilled",
    "gail": "ppo",
}


@ex.config
def config():
    run_0: RunStr = "ppo"
    checkpoint_path_0 = None
    policy_id_0 = default_policy_ids[run_0]  # noqa: F841
    run_1: RunStr = run_0
    checkpoint_path_1 = checkpoint_path_0  # noqa: F841
    policy_id_1 = default_policy_ids[run_1]  # noqa: F841

    layout_name = "cramped_room"
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    mdp_params = {  # noqa: F841
        "layout_name": layout_name,
        "rew_shaping_params": rew_shaping_params,
    }
    ep_length = 400
    num_games = 1
    display = False
    eval_params = {  # noqa: F841
        "ep_length": ep_length,
        "num_games": num_games,
        "display": display,
    }

    # Whether to evaluate with flipped starting positions.
    evaluate_flipped = False  # noqa: F841

    render_path = None  # noqa: F841
    render_action_probs = False  # noqa: F841

    out_tag = None  # noqa: F841
    if out_tag is not None:
        out_path = os.path.join(os.path.dirname(checkpoint_path_0), out_tag)
    else:
        out_path = None  # noqa: F841


@ex.automain
def main(
    run_0: RunStr,
    checkpoint_path_0: str,
    policy_id_0: PolicyID,
    run_1: RunStr,
    checkpoint_path_1: str,
    policy_id_1: PolicyID,
    eval_params: dict,
    mdp_params: dict,
    evaluate_flipped: bool,
    render_path: Optional[str],
    render_action_probs: bool,
    out_path: Optional[str],
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer_0 = load_trainer(
        checkpoint_path_0, trainer_classes[run_0], config_overrides={"input": "sampler"}
    )
    policy_0 = trainer_0.get_policy(policy_id_0)
    trainer_1 = load_trainer(
        checkpoint_path_1, trainer_classes[run_1], config_overrides={"input": "sampler"}
    )
    policy_1 = trainer_1.get_policy(policy_id_1)
    assert isinstance(policy_0, TorchPolicy) and isinstance(policy_1, TorchPolicy)

    mdp_params = dict(mdp_params)
    evaluator = AgentEvaluator.from_layout_name(
        mdp_params=mdp_params,
        env_params={"horizon": eval_params["ep_length"], "num_mdp": 1},
    )
    env: OvercookedEnv = evaluator.env

    bc_obs_shape = env.featurize_state_mdp(env.mdp.get_standard_start_state())[0].shape

    bpd_featurize_fn: Optional[BPDFeaturizeFn] = None

    def get_featurize_fn(policy: TorchPolicy):
        if policy.observation_space.shape == bc_obs_shape:
            return lambda state: env.featurize_state_mdp(state)
        elif isinstance(policy.model, OvercookedPPODistributionModel):
            nonlocal bpd_featurize_fn
            if bpd_featurize_fn is None:
                bpd_featurize_fn = BPDFeaturizeFn(
                    evaluator.env,
                    latent_size=policy.model.latent_size,
                    use_tuple=isinstance(policy.model.obs_space, spaces.Tuple),
                )
            return bpd_featurize_fn
        else:
            return env.lossless_state_encoding_mdp

    results = evaluate(
        eval_params=dict(eval_params),
        mdp_params=mdp_params,
        outer_shape=None,
        agent_0_policy=policy_0,
        agent_1_policy=policy_1,
        agent_0_featurize_fn=get_featurize_fn(policy_0),
        agent_1_featurize_fn=get_featurize_fn(policy_1),
    )

    ep_returns = [int(ep_return) for ep_return in results["ep_returns"]]
    simple_results = {
        "ep_returns": ep_returns,
        "mean_return": float(np.mean(ep_returns)),
    }
    all_results = {
        "results": results,
    }
    ep_returns_all = list(ep_returns)

    if evaluate_flipped:
        results_flipped = evaluate(
            eval_params=dict(eval_params),
            mdp_params=mdp_params,
            outer_shape=None,
            agent_0_policy=policy_1,
            agent_1_policy=policy_0,
            agent_0_featurize_fn=get_featurize_fn(policy_1),
            agent_1_featurize_fn=get_featurize_fn(policy_0),
        )
        ep_returns_flipped = [
            int(ep_return) for ep_return in results_flipped["ep_returns"]
        ]
        simple_results.update(
            {
                "ep_returns_flipped": list(ep_returns_flipped),
                "mean_return_flipped": float(np.mean(ep_returns_flipped)),
            }
        )
        ep_returns_all.extend(ep_returns_flipped)
        all_results["results_flippped"] = results_flipped

    simple_results.update(
        {
            "ep_returns_all": ep_returns_all,
            "mean_return_all": float(np.mean(ep_returns_all)),
        }
    )

    if render_path:
        import skvideo.io
        import pygame
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        for episode_index, (episode_states, episode_infos) in enumerate(
            zip(results["ep_states"], results["ep_infos"])
        ):
            video_writer = skvideo.io.FFmpegWriter(
                f"{render_path}_{episode_index}.mp4",
                outputdict={
                    "-filter:v": "setpts=5*PTS",
                    "-pix_fmt": "yuv420p",
                },
            )
            state: OvercookedState
            info: EpisodeInformation
            for state, info in tqdm(
                zip(episode_states, episode_infos),
                desc=f"Rendering episode {episode_index}",
            ):
                state_frame = pygame.surfarray.array3d(
                    StateVisualizer(tile_size=60).render_state(
                        state,
                        grid=evaluator.env.mdp.terrain_mtx,
                        action_probs=[
                            agent_info["action_probs"][0]
                            for agent_info in info["agent_infos"]
                        ]
                        if render_action_probs
                        else None,
                    )
                ).transpose((1, 0, 2))
                video_writer.writeFrame(state_frame)
            video_writer.close()

    if out_path is not None:
        with open(f"{out_path}_results.pickle", "wb") as results_file:
            _log.info(f"Saving full results to {out_path}_results.pickle")
            pickle.dump(all_results, results_file)
        with open(f"{out_path}_simple_results.json", "w") as simple_results_file:
            _log.info(f"Saving simple results to {out_path}_simple_results.json")
            json.dump(simple_results, simple_results_file)

    return simple_results
