import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import PPO

from frequency_rl.policy.entropy_projection import LowPassGaussianTorchPolicy
from frequency_rl.utils.network import Network
from mushroom_rl.policy import GaussianTorchPolicy


def agent_builder(policy_name, mdp, agent_params):
    alg = PPO
    policy_type = GaussianTorchPolicy
    if policy_name == "lowpass":
        policy_type = LowPassGaussianTorchPolicy

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": agent_params["critic_lr"]}},
        loss=F.mse_loss,
        n_features=agent_params["n_features"],
        batch_size=agent_params["batch_size"],
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    actor_optimizer={"class": optim.Adam, "params": {"lr": agent_params["actor_lr"]}}

    policy = policy_type(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        n_features=agent_params["n_features"],
        std_0=agent_params["std_0"],
        cutoff_freq=agent_params["cutoff_freq"],
        order=agent_params["order"],
        sampling_freq=1. / mdp.info.dt,
        entropy_projection_method=agent_params["ent_projection_type"],
        normalize_std=agent_params["normalize_std"],
    )

    agent = alg(mdp.info, policy, actor_optimizer=actor_optimizer,
                critic_params=critic_params,
                n_epochs_policy=agent_params["n_epochs_policy"],
                batch_size=agent_params["batch_size"], eps_ppo=agent_params["eps"],
                lam=agent_params["lam"])
    return agent
