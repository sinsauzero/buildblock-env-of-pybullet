#The RL env of double mechanical arms which can achieve the task of building blocks based on Pybullet

## overview
I use kinova-arms as my based agent. And the env is packaged as a gym env. I use PPO to train the task of reach and it seems have great result.
However, using PPO to train the stack task may not get good result. It seems there's still a lot to improve. So I decided to post it first, and invite you to contribute.
And feel free to do any adjustment and use this project!
Thank all of you!

Here is how it looks like..
![](./t.gif )

## Requirements
To run the env or my RL algorithm, you need pybullet, gym, and torch

~~~
pip install -r requirments.txt
~~~
To test if installed successfully, you can test the env.
~~~
python test_env.py
~~~

##How to use the env
In this project, I defined 3 gym envs of different tasks in ```envs/__init__.py```, you can use it like other gym envs
~~~
env = gym.make('bintaskenv-v0')
env = gym.make('blockenv-v0')
env = gym.make("bintaskenv-v0")
~~~
 Note: the default env may not be shown when running, because the render is set to False.
 You can change the apis when defining the env.
example:
 ~~~
env = gym.make('bintaskenv-v0', render = True, double_agent = True, using_binary_task = True, need_GUI = True)
~~~
```render```: if env is shown. Recommended to be False when training.

```double_agent``` : whether using two arms or not. 

```using_banary_task``` : whether running subtasks or not. 

```need_GUI``` : whether u need GUI which can show some graphs from robot view.

Some other apis can be changed you can find it in ```/envs```to see the source code.

I run some RL algorithms to test my envs, I use PPO to test reach env and bintask env.
~~~
python ppo-reach-1.py
python ppo continuous.py
~~~
reach env shows great result and succeed. the Successful model is saved in ```/save_file/save_success.pth```
you can load this file to see the successful result.

However, the bintask env shows disappointed result. Could be the training time not enough, or the env definition not good.
And also I use the Disentangled_attention model to train the env but same results.
The model is derived from this paper.
> https://arxiv.org/abs/2106.05907
>
You can run this model
~~~
python Disentangled_attention.py
~~~

## binary-task 
I build this section in my env based on this paper
> https://arxiv.org/abs/1909.13874

In this paper, it comes up with an idea which called subtasks. Shortly, all task of multiple arms can be split into subtasks. 

Based on this, I design the task using bin_task of [lift, pick, rotate] as subtask. So when run task, it should decide which subtask to choose and then how to achieve it.
So in bintask_env, the action_space should be (spaces.Discrete(len(task_family)), spaces.Box...)
You can see the detail in ```envs/task_block_gym_env.py```