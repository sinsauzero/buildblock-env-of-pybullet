from envs import __init__

import gym
# env =gym.make("blockenv-v0", render = True, double_agent = True)
env = gym.make("bintaskenv-v0", render = True, using_binary_task = True, double_agent = True)
# env = gym.make('Pendulum-v0')
# model = PPO(policy= 'MlpPolicy', env = env)
# model.learn(total_timesteps=1000)
ob = env.reset()

while True:
    env.render("rgb_array")
    # a,s = model.predict(ob)

    ob, r, done, _ = env.step(env.action_space.sample())
    if done: break
