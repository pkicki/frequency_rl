from mushroom_rl.environments import Ant, HalfCheetah, Hopper, Walker2D

def env_builder(env_name, env_params):
    if env_name == "ant":
        return Ant()