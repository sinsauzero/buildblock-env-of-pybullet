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
writer = SummaryWriter('./runs/stack/log_1')
#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 4
buffer_size    = 2
minibatch_size = 2

class PPO(nn.Module):
    def __init__(self, ob_space, action_space):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(ob_space,64)
        self.fc_mask = nn.Linear(64,action_space[0])
        self.fc_mask2 = nn.Linear(64, action_space[1])
        self.fc2   = nn.Linear(64,128)
        self.fc3   = nn.Linear(128,64)
        self.fc_mu = nn.Linear(64,action_space[2])
        self.fc_std  = nn.Linear(64,action_space[2])
        self.fc_v = nn.Linear(64,1)
        self.fc_vmask2 = nn.Linear(64,1)
        self.fc_vmask = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def choose_act(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        a = self.fc_mask(x)
        a2 = self.fc_mask2(x)

        # para = self.fc_mask2.parameters()
        # print([p for p in para])
        prob = F.softmax(a, dim = -1)
        prob2 = F.softmax(a2, dim= -1)
        pi, pi2 = Categorical(prob), Categorical(prob2)

        # x = torch.argmax(x).reshape((1,))
        x, x2 = pi.sample(), pi2.sample()
        return x, x2, prob, prob2

    # def v_mask(self,x ):
    #     x = self.fc1(x)
    #     v = self.fc_vmask(x)
    #     v2 = self.fc_vmask2(x)
    #     # para = self.fc_vmask.parameters()
    #     # print([p for p in para])
    #     return v,v2

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
            s_batch, a_batch, a_mask_batch, a_mask2_batch, r_batch, s_prime_batch, prob_a_batch, prob_am_batch, prob_am2_batch, done_batch = [], [], [], [], [], [], [], [], [], []
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, a_mask_lst, a_mask2_lst, r_lst, s_prime_lst, prob_a_lst, prob_am_lst, prob_am2_lst, done_lst = [], [], [], [], [], [], [], [], [], []

                for transition in rollout:
                    s, a, a_mask, a_mask2, r, s_prime, prob_a, prob_am, prob_am2, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    a_mask_lst.append([a_mask.reshape(1,)])
                    a_mask2_lst.append([a_mask2.reshape(1,)])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    prob_am_lst.append([prob_am])
                    prob_am2_lst.append([prob_am2])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                a_mask_batch.append((a_mask_lst))
                a_mask2_batch.append((a_mask2_lst))
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                prob_am_batch.append(prob_am_lst)
                prob_am2_batch.append(prob_am2_lst)
                done_batch.append(done_lst)

            # mini_batch = list(map(lambda x: torch.tensor(x, dtype=torch.float), s_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), a_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), r_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), s_prime_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), done_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), prob_a_batch))
            s_batch2tensor = torch.tensor(s_batch, dtype = torch.float)
            # a_batch2numpy = []
            # for i in a_batch:
            #     a_batch2numpy.append([item[0].detach().numpy() for item in i])

            a_batch2tensor = torch.tensor(a_batch, dtype=torch.float)
            a_mask_batch2tensor = torch.tensor(a_mask_batch, dtype=torch.int64)
            a_mask2_batch2tensor = torch.tensor(a_mask2_batch, dtype=torch.int64)
            r_batch2tensor = torch.tensor(r_batch, dtype = torch.float)
            s_prime_batch2tensor = torch.tensor(s_prime_batch, dtype = torch.float)
            done_batch2tensor = torch.tensor(done_batch,dtype = torch.float)
            prob_a_batch2tensor = torch.tensor(prob_a_batch,dtype = torch.float)
            prob_am_batch2tensor = torch.tensor(prob_am_batch, dtype= torch.float)
            prob_am2_batch2tensor = torch.tensor(prob_am2_batch, dtype= torch.float)
            # s_batch, a_batch,r_batch, s_prime_batch, done_batch, prob_a_batch = map(np.array, [s_batch, a_batch,r_batch, s_prime_batch, done_batch, prob_a_batch])
            # mini_batch = torch.tensor([item.detach().numpy() for item in s_batch], dtype=torch.float), torch.tensor([item.detach().numpy() for item in a_batch], dtype=torch.float), \
            #               # torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
            #               # torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            mini_batch = s_batch2tensor, a_batch2tensor, a_mask_batch2tensor, a_mask2_batch2tensor, r_batch2tensor, s_prime_batch2tensor, done_batch2tensor, prob_a_batch2tensor, prob_am_batch2tensor, prob_am2_batch2tensor

            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, a_mask, a_mask2, r, s_prime, done_mask, old_log_prob, old_prob_am, old_prob_am2 = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                # v_mask1, v_mask2 = self.v_mask(s_prime)
                # td_target_mask = r + gamma * v_mask1 * done_mask
                # td_target_mask2 = r + gamma * v_mask2 * done_mask
                delta = td_target - self.v(s)
                # v_mask1, v_mask2 = self.v_mask(s)
                # delta_mask = td_target_mask - v_mask1
                # delta_mask2 = td_target_mask2 - v_mask2
            delta = delta.numpy()
            # delta_mask, delta_mask2 = delta_mask.numpy(), delta_mask2.numpy()

            advantage_batch_lst = []

            advantage_mask2_lst = []
            advantage_mask_blst = []
            advantage_mask2_blst = []

            i = 0
            for delta_one in delta:
                advantage_lst = []
                advantage = 0.0
                j = -1
                for delta_t in delta_one[::-1]:
                    done = done_mask[i, j, 0]
                    advantage = gamma * lmbda * advantage * done + delta_t[0]
                    j -= 1
                    advantage_lst.append([advantage])

                advantage_lst.reverse()
                advantage_batch_lst.append(advantage_lst)

                i +=1


            # for delta_mone in delta_mask:
            #     advantage_mask_lst = []
            #     advantage_mask = 0.0
            #     for delta_m in delta_mone[::-1]:
            #         advantage_mask = gamma * lmbda * advantage_mask + delta_m[0]
            #         advantage_mask_lst.append([advantage_mask])
            #
            #     advantage_mask_lst.reverse()
            #     advantage_mask_blst.append(advantage_mask_lst)
            # for delta_m in delta_mask2[::-1]:
            #     advantage_mask2 = gamma * lmbda * advantage_mask2 + delta_m[0]
            #     advantage_mask2_lst.append([advantage_mask2])
            #
            #
            #
            # advantage_mask2_lst.reverse()
            advantage = torch.tensor(advantage_batch_lst, dtype=torch.float)
            # advantage_mask = torch.tensor(advantage_mask_lst, dtype=torch.float)
            # advantage_mask2 = torch.tensor(advantage_mask2_lst, dtype = torch.float)
            data_with_adv.append((s, a, a_mask, a_mask2, r, s_prime, done_mask, old_log_prob, old_prob_am, old_prob_am2, td_target, advantage))


        return data_with_adv

        
    def train_net(self):
        print("length of data ", len(self.data))

        if len(self.data) == minibatch_size * buffer_size:
            print("start training")
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):

                for mini_batch in data:
                    s, a, a_mask, a_mask2, r, s_prime, done_mask, old_log_prob, old_prob_am, old_prob_am2, td_target, advantage  = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    _, _, prob_m, prob_m2 = self.choose_act(s)
                    # index =torch.tensor(a_mask, dtype = torch.int)
                    a_mask = torch.squeeze(a_mask, dim = 3)
                    a_mask2 = torch.squeeze(a_mask2, dim = 3)
                    old_prob_am = torch.squeeze(old_prob_am, dim=3)
                    old_prob_am2 = torch.squeeze(old_prob_am2, dim=3)
                    prob_am = prob_m.gather(2, a_mask)
                    prob_am2 = prob_m2.gather(2, a_mask2)
                    dist = Normal(mu, std)
                    a = torch.squeeze(a, dim=2)
                    log_prob = dist.log_prob(a)
                    old_log_prob = torch.squeeze(old_log_prob, dim=2)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))
                    ratio2 = torch.exp(torch.log(prob_am) - torch.log(old_prob_am))
                    ratio2_2 = torch.exp(torch.log(prob_am2) - torch.log(old_prob_am2))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    surr1_mask , surr1_mask2 = ratio2 * advantage, ratio2_2 *advantage
                    surr2_mask, surr2_mask2 = torch.clamp(ratio2, 1 - eps_clip, 1 + eps_clip) * advantage, torch.clamp(ratio2_2, 1 - eps_clip, 1 + eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)
                    # v_m1, v_m2 = self.v_mask(s)
                    loss2 = -torch.min(surr1_mask, surr2_mask)
                    loss2_2 = -torch.min(surr1_mask2, surr2_mask2)
                    loss2 = loss2 + loss2_2
                    # print(print([parameters for parameters in self.parameters()]))
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    loss2.mean().backward()
                    # loss2_2.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1


            print("training done")
        
def main():

    env = gym.make('bintaskenv-v0', render = True, double_agent = True, using_binary_task = True, need_GUI = True)
    # env = gym.make("blockenv-v0", render=True, double_agent=True)
    s = env.reset()
    s = np.concatenate([s["observation"], s["achieved_goal"], s["desired_goal"]])
    model = PPO(ob_space= s.shape[0], action_space=(env.action_space[0].n, env.action_space[1].n, env.action_space.sample()[2].shape[0]))
    tt = (env.action_space.sample()[0],env.action_space[1].n, env.action_space.sample()[2].shape)
    score = 0.0
    print_interval = 20
    rollout = []
    rewards_all = 0.0

    time_step = 0
    success = 0
    for n_epi in range(100000):
        s = env.reset()
        s = np.concatenate([s["observation"], s["achieved_goal"], s["desired_goal"]])
        done = False
        rewards = 0.0
        ep = 0
        while not done and ep < 400:

            for t in range(rollout_len):

                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                a_mask, a_mask2, prob, prob2 = model.choose_act(torch.from_numpy(s).float())

                # index_mask = torch.tensor(a_mask.copy(), dtype = torch.float)
                prob_am = prob.gather(0, a_mask).reshape((1,))
                prob_am2 = prob2.gather(0, a_mask2).reshape((1,))
                log_prob = dist.log_prob(a)


                s_prime, r, done, info = env.step((a_mask.numpy(), a_mask2.numpy(), a.numpy()))

                time_step +=1
                s_prime = np.concatenate([s_prime["observation"], s_prime["achieved_goal"], s_prime["desired_goal"]])
                rewards += r
                rewards_all +=r

                avg_rewards = rewards_all / time_step
                writer.add_scalar("avg_score", avg_rewards, time_step)

                rollout.append((s, a.numpy(), a_mask.numpy(), a_mask2.numpy(), r/10.0, s_prime, log_prob.detach().numpy(), prob_am.detach().numpy(), prob_am2.detach().numpy(), done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                ep +=1
                # score += r
                if done:
                    print("done!!")
                    success += 1
                    break
            #     imgs.append(env.render("rgb_array"))
            # imageio.mimsave("t.mp4", imgs)
            model.train_net()
            rate = success / (n_epi + 1)
            torch.save({"ckpt": model.state_dict(), "episode": n_epi}, "save_1.pth")

        writer.add_scalar("rewards", rewards, n_epi)
        writer.add_scalar("success_rate", rate, n_epi)
        print("reset!")


        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score/print_interval, model.optimization_step))
        #     score = 0.0

    env.close()

if __name__ == '__main__':
    main()