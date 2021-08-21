import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from envs import __init__
from tensorboardX import SummaryWriter
import numpy as np

# Hyperparameters
learning_rate = 0.0003
gamma = 0.9
lmbda = 0.9
eps_clip = 0.2
K_epoch = 10
rollout_len = 40
buffer_size = 20
minibatch_size = 2

# writer = SummaryWriter('./runs/log_4')
class PPO(nn.Module):
    def __init__(self, ob_space, action_space):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(ob_space, 64)

        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_space)
        self.fc_std = nn.Linear(64, action_space)
        self.fc_v = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim=0):
        x = self.fc1(x)
        # para = self.fc1.parameters()
        # print([p.grad for p in para])
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std





    def v(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):

        data = []

        for j in range(buffer_size):
            s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

                    s_lst.append(s)
                    a_lst.append([a])


                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])

                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)

                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)

                done_batch.append(done_lst)

            # mini_batch = list(map(lambda x: torch.tensor(x, dtype=torch.float), s_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), a_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), r_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), s_prime_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), done_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), prob_a_batch))
            s_batch2tensor = torch.tensor(s_batch, dtype=torch.float)
            # a_batch2numpy = []
            # for i in a_batch:
            #     a_batch2numpy.append([item[0].detach().numpy() for item in i])

            a_batch2tensor = torch.tensor(a_batch, dtype=torch.float)

            r_batch2tensor = torch.tensor(r_batch, dtype=torch.float)
            s_prime_batch2tensor = torch.tensor(s_prime_batch, dtype=torch.float)
            done_batch2tensor = torch.tensor(done_batch, dtype=torch.float)
            prob_a_batch2tensor = torch.tensor(prob_a_batch, dtype=torch.float)

            # s_batch, a_batch,r_batch, s_prime_batch, done_batch, prob_a_batch = map(np.array, [s_batch, a_batch,r_batch, s_prime_batch, done_batch, prob_a_batch])
            # mini_batch = torch.tensor([item.detach().numpy() for item in s_batch], dtype=torch.float), torch.tensor([item.detach().numpy() for item in a_batch], dtype=torch.float), \
            #               # torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
            #               # torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            mini_batch = s_batch2tensor, a_batch2tensor, r_batch2tensor, s_prime_batch2tensor, done_batch2tensor, prob_a_batch2tensor

            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob= mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask

                delta = td_target - self.v(s)

            delta = delta.numpy()


            advantage_batch_lst = []

            i = 0
            for delta_one in delta:
                advantage_lst = []
                advantage = 0.0

                j = -1
                for delta_t in delta_one[::-1]:
                    done = done_mask[i, j, 0]
                    advantage = gamma * lmbda * advantage * done + delta_t[0]
                    j -=1
                    advantage_lst.append([advantage])

                advantage_lst.reverse()
                advantage_batch_lst.append(advantage_lst)

                i +=1


            advantage = torch.tensor(advantage_batch_lst, dtype=torch.float)

            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob,
                                  td_target, advantage))

        return data_with_adv

    def train_net(self):
        print("length of data ", len(self.data))

        if len(self.data) == minibatch_size * buffer_size:
            print("start training")
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):

                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)

                    # index =torch.tensor(a_mask, dtype = torch.int)


                    dist = Normal(mu, std)
                    a = torch.squeeze(a, dim=2)
                    log_prob = dist.log_prob(a)
                    prob = log_prob.exp()
                    old_log_prob = torch.squeeze(old_log_prob, dim=2)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target)
                    # v_m1, v_m2 = self.v_mask(s)


                    # print(print([parameters for parameters in self.parameters()]))
                    self.optimizer.zero_grad()
                    loss.mean().backward()

                    # loss2_2.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1

            print("training done")


def main():

    # env = gym.make('bintaskenv-v0', double_agent=False, using_binary_task=False, need_GUI=False)

    env = gym.make("reachenv-v0", double_agent=False, need_GUI = False)
    s = env.reset()
    s = np.concatenate([s["observation"], s["achieved_goal"], s["desired_goal"]])
    model = PPO(ob_space=s.shape[0],
                action_space=( env.action_space.sample().shape[0]))
    CKPT = torch.load("save_file/save_success.pth")
    model.load_state_dict(CKPT["ckpt"])
    # writer.add_graph(model)
    tt = (env.action_space.sample().shape)
    score = 0.0
    print_interval = 20
    rollout = []
    rewards_all = 0.0

    # imgs = []
    # imgs.append(env.render("rgb_array"))

    time_step = 0
    success = 0
    for n_epi in range(10000):
        s = env.reset()
        s = np.concatenate([s["observation"], s["achieved_goal"], s["desired_goal"]])
        done = False
        rewards = 0.0
        ep = 0
        while not done and ep<200:

            for t in range(rollout_len):

                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()

                log_prob = dist.log_prob(a)
                prob = log_prob.exp()
                s_prime, r, done, info = env.step((a.numpy()))
                time_step +=1
                s_prime = np.concatenate([s_prime["observation"], s_prime["achieved_goal"], s_prime["desired_goal"]])
                rewards += r
                rewards_all +=r
                avg_rewards = rewards_all / time_step

                # writer.add_scalar("avg_score", avg_rewards, time_step)

                rollout.append((s, a.numpy(), r , s_prime,
                                log_prob.detach().numpy(),  done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                ep +=1
                score += r
                if done:
                    print("done!!")
                    success +=1
                    break
            #     imgs.append(env.render("rgb_array"))
            # imageio.mimsave("t.mp4", imgs)
            model.train_net()
            rate = success / (n_epi+1)
            # torch.save({"ckpt": model.state_dict(), "episode": n_epi}, "save_1.pth")
        # writer.add_scalar("rewards", rewards, n_epi)
        # writer.add_scalar("success_rate", rate, n_epi)
        print('reset!')

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score / print_interval,
                                                                              model.optimization_step))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()