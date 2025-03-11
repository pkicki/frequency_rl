from itertools import product

from experiment_launcher import Launcher, is_local
import os

os.environ["WANDB_API_KEY"] = "a9819ac569197dbd24b580d854c3041ad75efafd"

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 10
if LOCAL:
    N_EXPS_IN_PARALLEL = 10
else:
    N_EXPS_IN_PARALLEL = 80

N_CORES = 1
MEMORY_SINGLE_JOB = 2000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'standard'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = "freq_rl"

policy_name = "lowpass"
#policy_name = "default"

env_name = "ant"

experiment_name = f'{policy_name}_ppo_{env_name}'

launcher = Launcher(
    exp_name=experiment_name,
    exp_file='lpppo',
    n_seeds=N_SEEDS,
    start_seed=0,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=6,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

bs = [1.]
cs = [1.]

#for mu_lr, value_lr, constraint_lr in product(mu_lrs, value_lrs, constraint_lrs):
for b, c, in product(bs, cs):
    launcher.add_experiment(
        env_name=env_name,
        policy_name=policy_name,
        debug=False,
    )
launcher.run(LOCAL, TEST)