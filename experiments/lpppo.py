import numpy as np
import torch
import wandb

from mushroom_rl.core import Core, Logger
from frequency_rl.utils.env_builder import env_builder
from frequency_rl.utils.agent_builder import agent_builder

from tqdm import trange

from experiment_launcher import single_experiment, run_experiment


@single_experiment
def experiment(env_name: str = "ant",
               policy_name: str = "default",
               #policy_name: str = "lowpass",
               n_epochs: int = 50,
               n_steps: int = 30_000,
               n_episodes_test: int = 10,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               n_features: int = 32,
               batch_size: int = 64,
               n_epochs_policy: int = 10,
               eps: float = 0.2,
               lam: float = 0.95,
               std_0: float = 1.0,
               n_steps_per_fit: int = 2000,
               #ent_projection_type: str = "default",
               ent_projection_type: str = "indep",
               #initial_entropy_lb: float = 11.35,
               initial_entropy_lb: float = 0,
               entropy_lb: float = -11.35,
               entropy_lb_ep: int = 20,
               order: int = 1,
               cutoff_freq: float = 9.,
               normalize_std: bool = True,
               results_dir:str = "results",
               #debug: bool = False,
               debug: bool = True,
               group_name_prefix: str = "",
               seed: int = 0,
    ):
    np.random.seed(seed)
    torch.manual_seed(seed)

    training_params = dict(
        n_epochs=n_epochs,
        n_steps=n_steps,
        n_episodes_test=n_episodes_test,
        n_steps_per_fit=n_steps_per_fit
    )

    env_params = dict()
    env = env_builder(env_name, env_params)

    agent_params = dict(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        n_features=n_features,
        batch_size=batch_size,
        n_epochs_policy=n_epochs_policy,
        eps=eps,
        lam=lam,
        std_0=std_0,
        order=order,
        cutoff_freq=cutoff_freq,
        ent_projection_type=ent_projection_type,
        normalize_std=normalize_std,
    )
    agent = agent_builder(policy_name, env, agent_params)

    config = {**env_params, **agent_params, **training_params}
    mode = "disabled" if debug else "online"
    group_name = f"{env_name}_{policy_name}_o{order}_f{cutoff_freq}_nep{n_epochs}_nsteps{n_steps}_nstpf{n_steps_per_fit}_alr{actor_lr}_clr{critic_lr}_" \
                 f"nf{n_features}_bs{batch_size}_eps{eps}_lam{lam}_std0{std_0}_elb{initial_entropy_lb}_{entropy_lb}_{entropy_lb_ep}_ept{ent_projection_type}{'_normstd' if normalize_std else ''}"
    if group_name_prefix:
        group_name = f"{group_name_prefix}_{group_name}"
    wandb.init(project="corl25_initial_experiments", config=config, dir=results_dir, entity="kicai",
              group=f'{group_name}', mode=mode)

    logger = Logger(type(agent).__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + type(agent).__name__)

    core = Core(agent, env)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy().item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for epoch in trange(n_epochs, leave=False):
        # entropy lowe bound update
        if hasattr(core.agent.policy, "e_lb"):
            current_entropy_lb = np.maximum(initial_entropy_lb +
                (entropy_lb - initial_entropy_lb) * epoch / entropy_lb_ep, entropy_lb)
            core.agent.policy.e_lb = current_entropy_lb

        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy().item()
        wandb.log({"J": J, "R": R, "entropy":E, "epoch": epoch})

        logger.epoch_info(epoch + 1, J=J, R=R, entropy=E)

    #logger.info("Press a button to visualize")
    #input()
    #core.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    run_experiment(experiment)
