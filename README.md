# The Boltzmann Policy Distribution

This repository contains code and data for the ICLR 2022 paper [The Boltzmann Policy Distribution: Accounting for Systematic Suboptimality in Human Models](https://openreview.net/forum?id=_l_QjPGN5ye). In particular, the repository contains an implementation of our algorithm for computing the Boltzmann Policy Distribution (BPD) which is based around [RLlib](https://www.ray.io/rllib).

## Installation

The code can be downloaded as this GitHub repository or installed as a pip package.

### As a repository


1. Install [Python](https://www.python.org/) 3.8 or later (3.7 might work but may not be able to load pretrained checkpoints).
2. Clone the repository:

        git clone https://github.com/cassidylaidlaw/boltzmann-policy-distribution.git
        cd boltzmann-policy-distribution

2. Install pip requirements:

        pip install -r requirements.txt

### As a package

1. Install [Python 3](https://www.python.org/).
2. Install from PyPI:
    
        pip install boltzmann-policy-distribution

2. Import the package as follows:

        from bpd.agents.bpd_trainer import BPDTrainer

   See [getting_started.ipynb](getting_started.ipynb) or the Colab notebook below for examples of how to use the package.

## Data and Pretrained Models

Download human-human data from [here](https://boltzmann-policy-distribution.s3.us-east-2.amazonaws.com/human_data.zip).

Download pretrained models from [here](https://boltzmann-policy-distribution.s3.us-east-2.amazonaws.com/checkpoints.zip). The download includes a README describing which checkpoints are used where in the paper.

## Usage

This section explains how to get started with using the code and how to run the Overcooked experiments from the paper.

### Getting Started

The [getting_started.ipynb](getting_started.ipynb) notebook shows how to use the BPD to predict human behavior in a new environment. It is also available on Google Colab via the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cassidylaidlaw/boltzmann-policy-distribution/blob/main/getting_started.ipynb)

### Experiments

Each of the subsections below describes how to various experiments from the paper. All experiment configuration is done using [Sacred](https://sacred.readthedocs.io/en/stable/), and parameters can be updated from the command line by adding `param=value` after the command. For instance, most of the experiments require setting the Overcooked layout by, for instance, writing `layout_name="cramped_room"`.

We used [RLlib](https://www.ray.io/rllib) for reinforcement learning (RL) and many experiments output an RLlib checkpoint as the result. If a checkpoint from one experiment is needed for another experiment, you can find the checkpoint by looking at the output of the training run, which should look something like this:

    INFO - main - Starting training iteration 0
    INFO - main - Starting training iteration 1
    ...
    INFO - main - Saved final checkpoint to data/logs/self_play/ppo/cramped_room/2022-01-01_12-00-00/checkpoint_000500/checkpoint-500

Many experiments also log metrics to TensorBoard during training. Logs and checkpoints are stored in `data/logs` by default. You can open TensorBoard by running

    pip install tensorboard
    tensorboard --logdir data/logs

#### Calculating the BPD

To calculate the BPD for Overcooked, we used the following command:

    python -m bpd.experiments.train_overcooked with run="bpd" num_workers=25 num_training_iters=2000 layout_name="cramped_room" temperature=0.1 prior_concentration=0.2 reward_shaping_horizon=20000000 latents_per_iteration=250  share_dense_reward=True train_batch_size=100000 discriminate_sequences=True max_seq_len=10 entropy_coeff_start=0 entropy_coeff_end=0 latent_size=1000 sgd_minibatch_size=8000 use_latent_attention=True

Some useful parameters include

 * `temperature`: the parameter $1 / \beta$ from the paper, which controls how irrational or suboptimal the human is.
 * `prior_concentration`: the parameter $\alpha$ from the paper, which controls how inconsistent the human is.
 * `latent_size`: $n$, the size of the Gaussian latent vector $z$.

#### Training a predictive model for the BPD

In the paper, we describe training a sequence model (transformer) to do online prediction of human actions using the BPD. We also experimented with using an RNN, and the command to train either is as follows. To train a prediction model, the first step is to rollout many episodes from the BPD:

    python -m bpd.experiments.rollout with checkpoint=data/checkpoints/cramped_room/bpd_0.1_0.2_1000/checkpoint_000500/checkpoint-500 run=bpd num_workers=10 episodes=5000

Replace the `checkpoint=` parameter with the path to your BPD checkpoint. Then, look for a directory called `rollouts_2022-...` under the checkpoint directory. Use this to run the sequence model training:

    python -m bpd.experiments.train_overcooked with run="distill" num_training_iters=5000 distill_random_policies=True layout_name="cramped_room" use_sequence_model=True use_lstm=False train_batch_size=16000 sgd_minibatch_size=16000 num_sgd_iter=1 size_hidden_layers=256 input="data/checkpoints/cramped_room/bpd_0.1_0.2_1000/checkpoint_000500/rollouts_2022-01-01_12-00-00" save_freq=1000

You can set `use_lstm=True` to use an LSTM instead of a transformer for prediction.

#### Evaluating prediction

We haven't used any human data up until now to train the BPD and the predictive model! However, to evaluate the predictive power of the BPD, we'll need the human trajectories included in data download above. Assuming you've extracted them to `data/human_data`, you can run:

    python -m bpd.experiments.evaluate_overcooked_prediction with checkpoint_path=data/checkpoints/cramped_room/bpd_0.1_0.2_1000_transformer/checkpoint_005000/checkpoint-5000 run=distill human_data_fname="data/human_data/human_data_state_dict_and_action_by_traj_test_inserted_fixed.pkl" out_tag="test"

You should replace the `run=distill` parameter with whatever `run` parameter you used to **train** the model you want to evaluate. For instance, to evaluate the BPD policy distribution directly using mean-field variational inference (MFVI), you could run

    python -m bpd.experiments.evaluate_overcooked_prediction with checkpoint_path=data/checkpoints/cramped_room/bpd_0.1_0.2_1000/checkpoint_000500/checkpoint-500 run=bpd human_data_fname="data/human_data/human_data_state_dict_and_action_by_traj_test_inserted_fixed.pkl" out_tag="test"

#### Training a best response

Besides using the BPD to predict human actions, we might also want to use it to enable human-AI cooperation. We can do this by training a *best response* to the BPD which will learn to cooperate with all the policies in the BPD and thus hopefully with real humans as well. To train a best response, run:

    python -m bpd.experiments.train_overcooked with run="ppo" num_workers=10 num_training_iters=500 multiagent_mode="cross_play" checkpoint_to_load_policies=data/checkpoints/cramped_room/bpd_0.1_0.2_1000/checkpoint_000500/checkpoint-500 layout_name=cramped_room evaluation_interval=None entropy_coeff_start=0 entropy_coeff_end=0 share_dense_reward=True train_batch_size=100000 sgd_minibatch_size=8000

You can replace the `checkpoint_to_load_policies` parameter with any other checkpoint you want to train a best response to. For instance, [human-aware RL](https://github.com/HumanCompatibleAI/human_aware_rl) (HARL) is just a best response to a behavior cloned (BC) policy. To train a HARL policy, you can follow the instructions below to train a BC policy and then use that checkpoint with the command above.

#### Training a behavior cloning/human proxy policy

To train a behavior-cloned (BC) human policy from the human data, run:

    python -m bpd.experiments.train_overcooked_bc with layout_name="cramped_room" human_data_fname="data/human_data/human_data_state_dict_and_action_by_traj_train_inserted_fixed.pkl" save_freq=10 num_training_iters=100 validation_prop=0.1

By default, this will use special, hand-engineered features as the input to the policy network. To use the normal Overcooked features add `use_bc_features=False` to the command. To train a BC policy on the test set, replace `human_data_fname="data/human_data/human_data_state_dict_and_action_by_traj_test_inserted_fixed.pkl"` in the command.

#### Evaluating with a human proxy

We evaluated cooperative AI policies in the paper by testing how well they performed when paired with a human proxy policy trained via behavior cloning on the test set of human data. To test a best response policy, run:

    python -m bpd.experiments.evaluate_overcooked with layout_name=cramped_room run_0=ppo checkpoint_path_0=data/checkpoints/cramped_room/bpd_0.1_0.2_1000_br/checkpoint_002000/checkpoint-2000 policy_id_0=ppo_0 run_1=bc checkpoint_path_1=data/checkpoints/cramped_room/bc_test/checkpoint_000500/checkpoint-500 num_games=100 evaluate_flipped=True ep_length=400 out_tag=hproxy

If you want to test a policy which *isn't* a best response with the human proxy, remove the `policy_id_0=ppo_0` parameter and update the `run_0` parameter to whatever `run` parameter you used when training the policy.

#### Baselines

To train a **self-play policy**, run:

    python -m bpd.experiments.train_overcooked with run="ppo" num_workers=10 num_training_iters=500 layout_name="cramped_room" prior_concentration=1 reward_shaping_horizon=20000000 share_dense_reward=True train_batch_size=100000 entropy_coeff_start=0 entropy_coeff_end=0 sgd_minibatch_size=8000

To train a **Boltzmann rational policy**, use the same command but change the parameters `entropy_coeff_start=0.1 entropy_coeff_end=0.1` for $1 / \beta = 0.1$.

To train a human model using **generative adversarial imitation learning (GAIL)**, run:

    python -m bpd.experiments.train_overcooked with run="gail" num_workers=10 num_training_iters=500 layout_name=cramped_room prior_concentration=1 reward_shaping_horizon=20000000 share_dense_reward=True train_batch_size=100000 num_sgd_iter=1 entropy_coeff_start=0.1 entropy_coeff_end=0.1 human_data_fname="data/human_data/human_data_state_dict_and_action_by_traj_train_inserted_fixed.pkl" sgd_minibatch_size=8000

## Citation

If you find this repository useful for your research, please cite our paper as follows:

    @inproceedings{laidlaw2022boltzmann,
      title={The Boltzmann Policy Distribution: Accounting for Systematic Suboptimality in Human Models},
      author={Laidlaw, Cassidy and Dragan, Anca},
      booktitle={ICLR},
      year={2022}
    }

## Contact

For questions about the paper or code, please contact cassidy_laidlaw@berkeley.edu.
