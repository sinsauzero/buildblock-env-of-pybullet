from kinova_gym_env import *
from stable_baselines3 import  PPO
import gym
env = KinovaGymEnv(render=True)
# env = gym.make('Pendulum-v0')
model = PPO(policy= 'MlpPolicy', env = env)
# model.learn(total_timesteps=1000)
ob = env.reset()

while True:
    env.render("rgb_array")
    a,s = model.predict(ob)

    ob, r, done, _ = env.step(a)
    if done: break
