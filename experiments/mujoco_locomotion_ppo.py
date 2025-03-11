from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Ant, HalfCheetah, Hopper, Walker2D
from mushroom_rl.policy import GaussianTorchPolicy

from tqdm import trange




def experiment(env, n_epochs, n_steps, n_episodes_test):
    np.random.seed()

    logger = Logger(PPO.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + PPO.__name__)

    mdp = env()

    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = 32
    batch_size = 64
    n_epochs_policy = 10
    eps = 0.2
    lam = 0.95
    std_0 = 1.0
    n_steps_per_fit = 2000

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    alg_params = dict(
        actor_optimizer={"class": optim.Adam, "params": {"lr": actor_lr}},
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps,
        lam=lam,
        critic_params=critic_params,
    )

    policy_params = dict(std_0=std_0, n_features=n_features)

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        **policy_params,
    )

    agent = PPO(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy().item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    import wandb
    config = {}
    mode = "online"
    results_dir = "results"
    name = "plainppo_test_trial_2"
    group_name = "PPO"
    wandb_run = wandb.init(project="corl25_trials", config=config, dir=results_dir, name=name, entity="kicai",
              group=f'{group_name}', mode=mode)

    for epoch in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy().item()
        wandb.log({"J": J, "R": R, "entropy":E, "epoch": epoch})

        logger.epoch_info(epoch + 1, J=J, R=R, entropy=E)

    logger.info("Press a button to visualize")
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    #envs = [Ant, HalfCheetah, Hopper, Walker2D]
    envs = [Ant]
    for env in envs:
        experiment(env=env, n_epochs=50, n_steps=30000, n_episodes_test=10)
