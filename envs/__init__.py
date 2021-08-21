from envs.kinova_gym_env import KinovaGymEnv, KinovaReach
from envs.task_block_gym_env import TaskblockgymEnv
import gym

gym.register("blockenv-v0", entry_point = KinovaGymEnv)
env_kwargs =dict(render = True)

gym.register("reachenv-v0", entry_point = KinovaReach)
env_kwargs =dict(render = True)

gym.register("bintaskenv-v0", entry_point = TaskblockgymEnv)
env_kwargs = dict(render = True)