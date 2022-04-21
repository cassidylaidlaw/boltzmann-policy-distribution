import csv
import glob
import json
import os
import random
import time
from typing import Optional

import pytest

from bpd.experiments.train_overcooked import (
    ex as train_overcooked_ex,
)
from bpd.experiments.train_overcooked_bc import (
    ex as train_overcooked_bc_ex,
)
from bpd.experiments.evaluate_overcooked import (
    ex as evaluate_overcooked_ex,
)
from bpd.experiments.evaluate_overcooked_prediction import (
    ex as evaluate_overcooked_prediction_ex,
)
from bpd.experiments.rollout import ex as rollout_ex
from bpd.experiments.train_pickup_ring import ex as train_pickup_ring_ex


LAYOUT_NAMES = ["cramped_room", "forced_coordination", "coordination_ring"]
HUMAN_TRAIN_DATA_FNAME = "data/human_human_data/human_data_state_dict_and_action_by_traj_train_inserted_fixed.pkl"
HUMAN_TEST_DATA_FNAME = "data/human_human_data/human_data_state_dict_and_action_by_traj_test_inserted_fixed.pkl"


@pytest.fixture
def random_seed():
    random.seed(time.time_ns())


def get_metric_change(metric: str, run_dir) -> float:
    run_dir = glob.glob(str(run_dir))[0]
    with open(os.path.join(run_dir, "progress.csv"), "r") as progress_file:
        progress_csv = csv.DictReader(progress_file)
        first_row = next(iter(progress_csv))
        for row in progress_csv:
            last_row = row
    return float(last_row[metric]) - float(first_row[metric])


def test_bc(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    run = train_overcooked_bc_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "layout_name": layout_name,
            "validation_prop": 0.1,
            "num_training_iters": 10,
            "human_data_fname": HUMAN_TRAIN_DATA_FNAME,
        }
    )
    assert (
        run.result["info"]["learner"]["default_policy"]["learner_stats"]["bc_loss"]
        < 1.2
    )
    assert (
        run.result["info"]["learner"]["default_policy"]["learner_stats"][
            "validation/cross_entropy"
        ]
        < 1.2
    )


def test_bc_ppo_features(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    run = train_overcooked_bc_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "layout_name": layout_name,
            "validation_prop": 0.1,
            "num_training_iters": 10,
            "use_bc_features": False,
            "human_data_fname": HUMAN_TRAIN_DATA_FNAME,
        }
    )
    assert (
        run.result["info"]["learner"]["default_policy"]["learner_stats"]["bc_loss"]
        < 1.2
    )
    assert (
        run.result["info"]["learner"]["default_policy"]["learner_stats"][
            "validation/cross_entropy"
        ]
        < 1.2
    )


def test_bc_online(random_seed, tmp_path):
    # First, we have to make a BC trainer checkpoint which hasn't been trained at all.
    layout_name = random.choice(LAYOUT_NAMES)
    train_overcooked_bc_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "layout_name": layout_name,
            "num_training_iters": 1,
            "lr": 0,
            "human_data_fname": HUMAN_TRAIN_DATA_FNAME,
        }
    )

    # Now, we evaluate using the checkpoint.
    run_dir = glob.glob(str(tmp_path / "logs" / "bc" / layout_name / "*"))[0]
    checkpoint_fname = os.path.join(run_dir, "checkpoint_000001", "checkpoint-1")
    evaluate_run = evaluate_overcooked_prediction_ex.run(
        config_updates={
            "run": "bc",
            "checkpoint_path": checkpoint_fname,
            "trajectory_index": 0,
            "run_bc_online": True,
            "human_data_fname": HUMAN_TRAIN_DATA_FNAME,
        }
    )

    assert evaluate_run.result["cross_entropy"] < 3


def test_gail(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    run = train_overcooked_ex.run(
        config_updates={
            "run": "gail",
            "num_workers": 0,
            "layout_name": layout_name,
            "log_dir": tmp_path / "logs",
            "num_training_iters": 10,
            "train_batch_size": 2000,
            "num_sgd_iter": 1,
            "entropy_coeff_start": 0.1,
            "entropy_coeff_end": 0.1,
            "human_data_fname": HUMAN_TRAIN_DATA_FNAME,
        }
    )

    assert (
        run.result["info"]["learner"]["ppo"]["learner_stats"][
            "demonstration_cross_entropy"
        ]
        < 1.77
    )


def test_self_play(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    train_overcooked_ex.run(
        config_updates={
            "run": "ppo",
            "layout_name": layout_name,
            "log_dir": tmp_path / "logs",
            "num_training_iters": 10,
            "train_batch_size": 2000,
            "entropy_coeff_start": 0,
            "entropy_coeff_end": 0,
        }
    )

    assert (
        get_metric_change(
            "policy_reward_mean/ppo",
            tmp_path / "logs" / "self_play" / "ppo" / layout_name / "*",
        )
        > 1
    )


def test_bpd(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    train_overcooked_ex.run(
        config_updates={
            "run": "bpd",
            "num_workers": 0,
            "layout_name": layout_name,
            "log_dir": tmp_path / "logs",
            "num_training_iters": 10,
            "train_batch_size": 2000,
            "prior_concentration": 1,
            "latents_per_iteration": 5,
            "share_dense_reward": True,
            "discriminate_sequences": True,
            "max_seq_len": 10,
            "entropy_coeff_start": 0,
            "entropy_coeff_end": 0,
            "latent_size": 100,
            "temperature": 0.2,
            "use_latent_attention": True,
        }
    )

    log_dir = (
        tmp_path
        / "logs"
        / "self_play"
        / "bpd"
        / layout_name
        / "temperature_0.2"
        / "concentration_1"
        / "*"
    )
    assert get_metric_change("policy_reward_mean/ppo", log_dir) > 0.5
    assert (
        get_metric_change("info/learner/ppo/learner_stats/discriminator/loss", log_dir)
        < -0.5
    )


def test_train_br(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    # Choose a random run that we know we trained a best response to and try
    # training it here.
    all_runs = [
        os.path.basename(run_dir)
        for run_dir in glob.glob(os.path.join("data/checkpoints", layout_name, "*"))
    ]
    runs_with_br = [run for run in all_runs if f"{run}_br" in all_runs]
    run_name = random.choice(runs_with_br)
    run_dir = os.path.join("data/checkpoints", layout_name, run_name)
    checkpoint_fname = glob.glob(os.path.join(run_dir, "*", "checkpoint-*[0-9]"))[0]

    config_updates = {
        "run": "ppo",
        "log_dir": tmp_path / "logs",
        "layout_name": layout_name,
        "multiagent_mode": "cross_play",
        "checkpoint_to_load_policies": checkpoint_fname,
        "evaluation_interval": None,
        "experiment_tag": f"{run_name}_br",
        "entropy_coeff_start": 0,
        "entropy_coeff_end": 0,
        "num_training_iters": 10,
        "train_batch_size": 2000,
        "num_workers": 0,
    }
    if run_name.startswith("random"):
        config_updates["loaded_policy_id"] = "random"
    br_run = train_overcooked_ex.run(config_updates=config_updates)

    log_dir = (
        tmp_path / "logs" / "cross_play" / "ppo" / layout_name / f"{run_name}_br" / "*"
    )
    assert (
        get_metric_change("episode_reward_mean", log_dir) > 0
        or br_run.result["episode_reward_mean"] > 5
    )


def test_train_recurrent_br(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    # Choose a random run that we know we trained a recurrent best response to and try
    # training it here.
    all_runs = [
        os.path.basename(run_dir)
        for run_dir in glob.glob(os.path.join("data/checkpoints", layout_name, "*"))
    ]
    runs_with_br = [run for run in all_runs if f"{run}_br_recurrent" in all_runs]
    run_name = random.choice(runs_with_br)
    run_dir = os.path.join("data/checkpoints", layout_name, run_name)
    checkpoint_fname = glob.glob(os.path.join(run_dir, "*", "checkpoint-*[0-9]"))[0]
    lstm_checkpoint_fname = glob.glob(
        os.path.join(
            "data/checkpoints",
            layout_name,
            f"{run_name}_lstm",
            "*",
            "checkpoint-*[0-9]",
        )
    )[0]

    config_updates = {
        "run": "ppo",
        "log_dir": tmp_path / "logs",
        "layout_name": layout_name,
        "multiagent_mode": "cross_play",
        "checkpoint_to_load_policies": checkpoint_fname,
        "evaluation_interval": None,
        "experiment_tag": f"{run_name}_br_recurrent",
        "entropy_coeff_start": 0,
        "entropy_coeff_end": 0,
        "num_training_iters": 10,
        "train_batch_size": 2000,
        "recurrent_checkpoint": lstm_checkpoint_fname,
        "max_seq_len": 1,
    }
    if run_name.startswith("random"):
        config_updates["loaded_policy_id"] = "random"
    br_run = train_overcooked_ex.run(config_updates=config_updates)

    log_dir = (
        tmp_path
        / "logs"
        / "cross_play"
        / "ppo"
        / layout_name
        / f"{run_name}_br_recurrent"
        / "*"
    )
    assert (
        get_metric_change("episode_reward_mean", log_dir) > 0
        or br_run.result["episode_reward_mean"] > 5
    )


def test_distill(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    # Choose a random run that we know we trained a distilled predictive model from.
    all_runs = [
        os.path.basename(run_dir)
        for run_dir in glob.glob(os.path.join("data/checkpoints", layout_name, "*"))
    ]
    runs_with_distill = [run for run in all_runs if f"{run}_transformer" in all_runs]
    run_name = random.choice(runs_with_distill)
    run_dir = os.path.join("data/checkpoints", layout_name, run_name)
    checkpoint_fname = glob.glob(os.path.join(run_dir, "*", "checkpoint-*[0-9]"))[0]
    model_type = random.choice(["lstm", "transformer"])

    # Next, generate some rollouts.
    rollout_dir = str(tmp_path / "rollouts")
    rollout_ex.run(
        config_updates={
            "num_workers": 0,
            "checkpoint": checkpoint_fname,
            "run": "distill" if run_name.startswith("random") else "bpd",
            "episodes": 10,
            "out_dir": rollout_dir,
        }
    )

    distill_run = train_overcooked_ex.run(
        config_updates={
            "run": "distill",
            "log_dir": tmp_path / "logs",
            "layout_name": layout_name,
            "num_workers": 0,
            "num_training_iters": 20,
            "checkpoint_to_load_policies": None
            if run_name.startswith("random")
            else checkpoint_fname,
            "distill_random_policies": run_name.startswith("random"),
            "use_sequence_model": True,
            "use_lstm": model_type == "lstm",
            "train_batch_size": 2000,
            "num_sgd_iter": 1,
            "size_hidden_layers": 256,
            "experiment_tag": f"{run_name}_{model_type}",
            "input": rollout_dir,
        }
    )
    policy_id = next(iter(distill_run.result["info"]["learner"]))

    log_dir = (
        tmp_path
        / "logs"
        / "self_play"
        / "distill"
        / layout_name
        / f"{run_name}_{model_type}"
        / "*"
    )
    assert (
        get_metric_change(
            f"info/learner/{policy_id}/learner_stats/cross_entropy", log_dir
        )
        < 0.001
    )


def get_trainer(run_name: str) -> str:
    if "_br" in run_name:
        return "ppo"
    elif run_name.startswith("bc_"):
        return "bc"
    elif run_name.startswith("gail_"):
        return "gail"
    elif (
        run_name.endswith("_lstm")
        or run_name.endswith("_transformer")
        or run_name.startswith("random")
    ):
        return "distill"
    elif run_name.startswith("bpd_"):
        return "bpd"
    else:
        return "ppo"


def test_evaluate_prediction(random_seed):
    layout_name = random.choice(LAYOUT_NAMES)
    # Choose a random run that is used for prediction.
    all_runs = [
        os.path.basename(run_dir)
        for run_dir in glob.glob(os.path.join("data/checkpoints", layout_name, "*"))
    ]
    run_name: Optional[str] = None
    while run_name is None:
        run_name = random.choice(all_runs)
        run_dir = os.path.join("data/checkpoints", layout_name, run_name)
        checkpoint_fname = glob.glob(os.path.join(run_dir, "*", "checkpoint-*[0-9]"))[0]
        try:
            prev_results_fname = glob.glob(
                os.path.join(run_dir, "*", "test_prediction_results.json")
            )[0]
        except IndexError:
            run_name = None

    config_updates = {
        "checkpoint_path": checkpoint_fname,
        "run": get_trainer(run_name),
        "human_data_fname": HUMAN_TEST_DATA_FNAME,
        "trajectory_index": 0,
        "run_bc_online": "online" in run_name,
    }
    if run_name.startswith("random"):
        config_updates["policy_id"] = "random_distilled"
    eval_run = evaluate_overcooked_prediction_ex.run(config_updates=config_updates)

    with open(prev_results_fname, "r") as prev_results_file:
        prev_results = json.load(prev_results_file)

    for metric_name, metric_values in eval_run.result.items():
        if isinstance(metric_values, dict):
            for episode_id in metric_values:
                assert (
                    abs(
                        eval_run.result[metric_name][episode_id]
                        - prev_results[metric_name][str(episode_id)]
                    )
                    < 0.1
                )


def test_evaluate_hproxy(random_seed):
    layout_name = random.choice(LAYOUT_NAMES)
    # Choose a random run that was previously evaluated with the human proxy.
    all_runs = [
        os.path.basename(run_dir)
        for run_dir in glob.glob(os.path.join("data/checkpoints", layout_name, "*"))
    ]
    run_name: Optional[str] = None
    while run_name is None:
        run_name = random.choice(all_runs)
        run_dir = os.path.join("data/checkpoints", layout_name, run_name)
        checkpoint_fname = glob.glob(os.path.join(run_dir, "*", "checkpoint-*[0-9]"))[0]
        try:
            prev_results_fname = glob.glob(
                os.path.join(run_dir, "*", "hproxy_simple_results.json")
            )[0]
        except IndexError:
            run_name = None
    hproxy_checkpoint_path = glob.glob(
        os.path.join(
            "data/checkpoints", layout_name, "bc_test", "*", "checkpoint-*[0-9]"
        )
    )[0]

    config_updates = {
        "layout_name": layout_name,
        "run_0": get_trainer(run_name),
        "checkpoint_path_0": checkpoint_fname,
        "run_1": "bc",
        "checkpoint_path_1": hproxy_checkpoint_path,
        "num_games": 3,
        "evaluate_flipped": True,
        "ep_length": 400,
    }
    if "_br" in run_name:
        config_updates["policy_id_0"] = "ppo_0"

    eval_run = evaluate_overcooked_ex.run(config_updates=config_updates)

    with open(prev_results_fname, "r") as prev_results_file:
        prev_results = json.load(prev_results_file)

    for returns_key in ["ep_returns", "ep_returns_flipped"]:
        for ep_return in eval_run.result[returns_key]:
            assert ep_return in prev_results[returns_key]


def test_render(random_seed, tmp_path):
    layout_name = random.choice(LAYOUT_NAMES)
    # Choose a random run that was previously evaluated with the human proxy.
    all_runs = [
        os.path.basename(run_dir)
        for run_dir in glob.glob(os.path.join("data/checkpoints", layout_name, "*"))
    ]
    run_name = random.choice(
        [run_name for run_name in all_runs if get_trainer(run_name) == "bpd"]
    )
    run_dir = os.path.join("data/checkpoints", layout_name, run_name)
    checkpoint_fname = glob.glob(os.path.join(run_dir, "*", "checkpoint-*[0-9]"))[0]

    render_path = str(tmp_path / "episode")

    evaluate_overcooked_ex.run(
        config_updates={
            "layout_name": layout_name,
            "run_0": "bpd",
            "checkpoint_path_0": checkpoint_fname,
            "run_1": "bpd",
            "checkpoint_path_1": checkpoint_fname,
            "num_games": 1,
            "evaluate_flipped": False,
            "render_path": render_path,
            "ep_length": 400,
        }
    )
    assert os.path.exists(render_path + "_0.mp4")


def test_pickup_ring(tmp_path):
    train_pickup_ring_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "num_workers": 0,
            "temperature": 0.1,
            "prior_concentration": 0.2,
            "discriminate_sequences": True,
            "num_training_iters": 10,
        }
    )
    log_dir = (
        tmp_path
        / "logs"
        / "ring_size_8"
        / "temperature_0.1"
        / "concentration_0.2"
        / "*"
    )
    assert get_metric_change("episode_reward_mean", log_dir) > 0.5
    assert (
        get_metric_change(
            "info/learner/default_policy/learner_stats/discriminator/loss", log_dir
        )
        < -0.5
    )
