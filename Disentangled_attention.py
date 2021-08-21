import math

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

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_tau      = 0.01
gamma           = 0.98

buffer_limit    = 50000
# tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_tau        = 0.0001  # for automated alpha update
rollout_len    = 3
buffer_size    = 2
minibatch_size = 2
K_epoch         = 10

tb_epoch = 0

writer = SummaryWriter('./runs/log')
class DA(nn.Module):
    def __init__(self, ob_space : dict, action_space:tuple):
        super(DA, self).__init__()
        self.data = []
        self.ob_space = np.concatenate([ob_space["observation"], ob_space["achieved_goal"], ob_space["desired_goal"]]).shape[0]
        self.robot_state_space =int (ob_space["observation"].shape[0] / 2)
        self.block_state_space = int(ob_space["achieved_goal"].shape[0])
        self.target_state_space = int(ob_space["desired_goal"].shape[0])
        self.action_space_per = int(action_space[2] / 2)
        # policy_init
        self.fch = nn.Linear(512, 64)

        self.fii = nn.Linear(self.robot_state_space, 64)
        self.fii_2 = nn.Linear(64,512)

        self.fi2 = nn.Linear(self.robot_state_space, 64)
        self.fi2_2 = nn.Linear(64, 512)

        self.fij = nn.Linear(self.block_state_space, 64)
        self.fij_2 = nn.Linear(64, 512)

        self.fc_mask = nn.Linear(64, action_space[0])
        self.fc_mask2 = nn.Linear(64, action_space[1])

        self.wq = nn.Linear(512, 1)
        self.wk = nn.Linear(512, 1)
        self.g = nn.Linear(512, 512)

        self.fc_mu = nn.Linear(64, action_space[2])
        self.fc_std = nn.Linear(64, action_space[2])

        # Q init
        action = int(1 + action_space[2] / 2)
        self.fch_Q = nn.Linear(512, 64)

        self.fii_Q = nn.Linear(self.robot_state_space + action, 64)
        self.fii_2Q = nn.Linear(64, 512)

        self.fi2_Q = nn.Linear(self.robot_state_space, 64)
        self.fi2_2Q = nn.Linear(64, 512)

        self.fij_Q = nn.Linear(self.block_state_space, 64)
        self.fij_2Q = nn.Linear(64, 512)



        self.wq_Q = nn.Linear(512, 1)
        self.wk_Q = nn.Linear(512, 1)
        self.g_Q = nn.Linear(512, 512)



        self.fc_v = nn.Linear(64, 1)
        # self.fc_vmask2 = nn.Linear(64, 1)
        # self.fc_vmask = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.log_tau1= torch.tensor(np.log(init_tau))
        self.log_tau2 = torch.tensor(np.log(init_tau))
        self.log_tau1.requires_grad = True
        self.log_tau2.requires_grad = True
        self.log_tau_optimizer = optim.Adam([self.log_tau1], lr=lr_tau)

    def _state_encode(self, i, j, state : torch.Tensor):
        """
        state(1d) -> 512D representation   (f(i*j))
        """
        if isinstance(state, np.ndarray):

            state = torch.from_numpy(state)
            state = state.float()

        if (j == i):
            # x = state.shape
            assert state.shape[0] == self.robot_state_space

            s = self.fii(state)

            # para = self.fii.parameters()
            # print([p.grad for p in para])
            state = self.fii_2(s)

        if (j<=2 and j!=i):
            assert  state.shape[0] == self.robot_state_space
            s = self.fi2(state)

            state = self.fi2_2(s)

        if (j>2 and j!=i):
            assert state.shape[0] == self.block_state_space
            s = self.fij(state)

            state = self.fij_2(s)

        return state
    def _state_encode_Q(self, i, j, state, a):
        """
        state(1d) -> 512D representation   (f(i*j))
        """
        if (j == i):

            assert state.shape[0] == self.robot_state_space
            if isinstance(a, torch.Tensor) :state = torch.cat([state, a])
            else: state = torch.cat([state, torch.zeros(self.action_space_per + 1)])
            s = self.fii_Q(state)

            state = self.fii_2Q(s)

        if (j<=2and j!=i):
            assert  state.shape[0] == self.robot_state_space
            s = self.fi2_Q(state)
            state = self.fi2_2Q(s)

        if (j>2and j!=i):
            assert state.shape[0] == self.block_state_space
            s = self.fij_Q(state)
            state = self.fij_2Q(s)

        return state

    def _compute_beta(self, i, j, state_i, state_j, q = False, action = None):
        """
        state_i/j : 1d
        action :1d
        return : tensor
        """
        if q:
            x1 = self.wq(self._state_encode_Q(i, i, state_i,action))
            x2 = self.wk(self._state_encode_Q(i, j, state_j,action))
            beta = x1 * x2 / math.sqrt(512)
            return beta

        x1 = self.wq(self._state_encode(i,i,state_i))
        x2 = self.wk(self._state_encode(i, j, state_j))
        beta = x1 * x2 / math.sqrt(512)
        return beta

    def _attention_embbed(self, i, ob: tuple, Q = False, action = None):
        """
        ob: (rob[0], rob[1], achieved_goal, desired_goal)
        """
        exp = 0.0

        if Q:

            for j in range(1, 5):
                exp += torch.exp(self._compute_beta(i, j, ob[i - 1], ob[j - 1], q=Q, action = action))
            alphals = []
            v = 0.0
            for j in range(1, 5):
                e = torch.exp(self._compute_beta(i, j, ob[i - 1], ob[j - 1],q=Q, action = action))
                alpha = e / exp

                v += self._state_encode_Q(i, j, ob[j - 1], action) * alpha
                alphals.append(alpha)


            alphals = torch.cat(alphals, dim=0)
            return v, alphals

        for j in range(1, 5):
            exp += torch.exp(self._compute_beta(i, j, ob[i - 1], ob[j - 1]))
        alphals = []
        v = 0.0
        for j in range(1,5):
            e = torch.exp(self._compute_beta(i , j, ob[i- 1] , ob[j- 1] ))
            alpha = e / exp
            v += self._state_encode(i, j, ob[j- 1] ) * alpha
            alphals.append(alpha)

        alphals = torch.cat(alphals, dim=0)
        return v, alphals

    def policy_net(self, i, ob_state: dict):
        rob = ob_state["observation"].reshape([2, -1])
        ob = (rob[0], rob[1], ob_state["achieved_goal"], ob_state["desired_goal"])
        v, alpha_i = self._attention_embbed(i, ob)
        v = self.g(v)
        v = nn.LayerNorm(512)(v)
        f = self._state_encode(i,i,ob[i-1])
        h = f + v
        h = self.fch(h)
        return h, alpha_i

    def compute_Q(self,  i, obs:dict, action):
        Q_batch = []
        action = torch.squeeze(action)
        if isinstance(obs, list):
            ind1 = 0

            for ob_states in obs:
                Q = []
                ind2 = 0
                for ob_state in ob_states:
                    rob = ob_state["observation"].reshape([2, -1])
                    ob = (rob[0], rob[1], ob_state["achieved_goal"], ob_state["desired_goal"])
                    v, alpha_i = self._attention_embbed(i, ob, Q= True, action = action[ind1][ind2])
                    v = self.g_Q(v)
                    v = nn.LayerNorm(512)(v)
                    f = self._state_encode_Q(i, i, ob[i - 1],action[ind1][ind2])
                    h = f + v
                    h = self.fch_Q(h)
                    Q.append(self.fc_v(h))
                    ind2 +=1
                Q = torch.cat(Q, dim=0).reshape([1,-1])
                Q_batch.append(Q)
                ind1 +=1
            Q_batch = torch.cat(Q_batch, dim = 0)
        else:
            ob_state = obs
            rob = ob_state["observation"].reshape([2, -1])
            ob = (rob[0], rob[1], ob_state["achieved_goal"], ob_state["desired_goal"])
            v, alpha_i = self._attention_embbed(i, ob, Q=True, action=action)
            v = self.g_Q(v)
            v = nn.LayerNorm(512)(v)
            f = self._state_encode_Q(i, i, ob[i - 1], action)
            h = f + v
            h = self.fch_Q(h)
            Q_batch = self.fc_v(h)


        return Q_batch


    def choose_action(self, ob_state : dict):
        if isinstance(ob_state, list):
            x, x2, alpha1, alpha2 = [], [], [], []
            for obs in ob_state:
                x_lst, x2_lst, a1_lst, a2_lst = [], [], [], []
                for ob in obs:
                    x_, a1 = self.policy_net(1, ob)
                    x2_, a2 = self.policy_net(2, ob)

                    x_lst.append(x_.reshape([1,-1]))
                    a1_lst.append(a1.reshape([1,-1]))
                    x2_lst.append(x2_.reshape([1,-1]))
                    a2_lst.append(a2.reshape([1,-1]))
                x_lst = torch.unsqueeze(torch.cat(x_lst, dim=0), dim = 0)
                a1_lst =torch.unsqueeze (torch.cat(a1_lst, dim=0), dim = 0)
                x2_lst = torch.unsqueeze(torch.cat(x2_lst, dim=0), dim = 0)
                a2_lst = torch.unsqueeze(torch.cat(a2_lst, dim=0), dim = 0)
                x.append(x_lst)
                x2.append(x2_lst)
                alpha1.append(a1_lst)
                alpha2.append(a2_lst)
            x = torch.cat(x, dim = 0)
            x2 =torch.cat(x2, dim = 0)
            alpha1 = torch.cat(alpha1, dim = 0)
            alpha2 = torch.cat(alpha2, dim = 0)
        else:
            x, alpha1 = self.policy_net(1, ob_state)
            x2, alpha2 = self.policy_net(2, ob_state)
        x = F.relu(x)
        a = self.fc_mask(x)
        x2 = F.relu(x2)
        a2 = self.fc_mask2(x2)


        # para = self.fc_mask2.parameters()
        # print([p for p in para])
        prob = F.softmax(a, dim=-1)
        prob2 = F.softmax(a2, dim=-1)
        pi, pi2 = Categorical(prob), Categorical(prob2)

        # x = torch.argmax(x).reshape((1,))
        x, x2 = pi.sample(), pi2.sample()
        return x, x2, prob, prob2, alpha1, alpha2

    def pi(self, ob_state : dict, softmax_dim=0):
        if isinstance(ob_state, list):
            x, x2= [], []
            for obs in ob_state:
                x_lst, x2_lst= [], []
                for ob in obs:
                    x_, a1 = self.policy_net(1, ob)
                    x2_, a2 = self.policy_net(2, ob)

                    x_lst.append(x_.reshape([1, -1]))

                    x2_lst.append(x2_.reshape([1, -1]))

                x_lst = torch.unsqueeze(torch.cat(x_lst, dim=0), dim=0)

                x2_lst = torch.unsqueeze(torch.cat(x2_lst, dim=0), dim=0)

                x.append(x_lst)
                x2.append(x2_lst)

            x = torch.cat(x, dim=0)
            x2 = torch.cat(x2, dim=0)


        else:
            x, alpha1 = self.policy_net(1, ob_state)
            x2, alpha2 = self.policy_net(2, ob_state)

        x = x + x2
        x = F.leaky_relu(x, 0.1)


        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        return mu, std

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_batch, a_batch, a_mask_batch, a_mask2_batch, r_batch, s_prime_batch, prob_a_batch, prob_am_batch, prob_am2_batch, done_batch=[], [], [], [], [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, a_mask_lst, a_mask2_lst, r_lst, s_prime_lst, prob_a_lst, prob_am_lst, prob_am2_lst, done_lst=[], [], [], [], [], [], [], [], [], []

                for transition in rollout:
                    s, a, a_mask, a_mask2, r, s_prime, prob_a, prob_am, prob_am2, done= transition

                    for key,value in s.items():
                            s[key] = torch.tensor(value, dtype=torch.float)


                    for key, value in s_prime.items():
                            s_prime[key] = torch.tensor(value, dtype = torch.float)
                    s_lst.append(s)
                    a_lst.append([a])
                    a_mask_lst.append([a_mask.reshape(1, )])
                    a_mask2_lst.append([a_mask2.reshape(1, )])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    prob_am_lst.append([prob_am])
                    prob_am2_lst.append([prob_am2])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])
                    # alpha1_lst.append([alpha1])
                    # alpha2_lst.append([alpha2])

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
                # alpha1_batch.append(alpha1_lst)
                # alpha2_batch.append(alpha2_lst)

            # mini_batch = list(map(lambda x: torch.tensor(x, dtype=torch.float), s_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), a_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), r_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), s_prime_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), done_batch)), \
            #              list(map(lambda x: torch.tensor(x, dtype=torch.float), prob_a_batch))
            # s_batch2tensor = torch.tensor(s_batch, dtype=torch.float)
            # a_batch2numpy = []
            # for i in a_batch:
            #     a_batch2numpy.append([item[0].detach().numpy() for item in i])
            s_batch2tensor = s_batch.copy()
            a_batch2tensor = torch.tensor(a_batch, dtype=torch.float)
            a_mask_batch2tensor = torch.tensor(a_mask_batch, dtype=torch.int64)
            a_mask2_batch2tensor = torch.tensor(a_mask2_batch, dtype=torch.int64)
            r_batch2tensor = torch.tensor(r_batch, dtype=torch.float)
            s_prime_batch2tensor = s_prime_batch.copy()
            done_batch2tensor = torch.tensor(done_batch, dtype=torch.float)
            prob_a_batch2tensor = torch.tensor(prob_a_batch, dtype=torch.float)
            prob_am_batch2tensor = torch.tensor(prob_am_batch, dtype=torch.float)
            prob_am2_batch2tensor = torch.tensor(prob_am2_batch, dtype=torch.float)
            # alpha1_batch2tensor = torch.tensor(alpha1_batch, dtype=torch.float)
            # alpha2_batch2tensor = torch.tensor(alpha2_batch, dtype=torch.float)
            # s_batch, a_batch,r_batch, s_prime_batch, done_batch, prob_a_batch = map(np.array, [s_batch, a_batch,r_batch, s_prime_batch, done_batch, prob_a_batch])
            # mini_batch = torch.tensor([item.detach().numpy() for item in s_batch], dtype=torch.float), torch.tensor([item.detach().numpy() for item in a_batch], dtype=torch.float), \
            #               # torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
            #               # torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            mini_batch = s_batch2tensor, a_batch2tensor, a_mask_batch2tensor, a_mask2_batch2tensor, r_batch2tensor, s_prime_batch2tensor, done_batch2tensor, prob_a_batch2tensor, prob_am_batch2tensor, prob_am2_batch2tensor

            data.append(mini_batch)

        return data


    def calc_loss(self, mini_batch):


            s, a, a_mask, a_mask2, r, s_prime, done_mask,  log_prob, prob_am, prob_am2 = mini_batch

            a_1q = torch.cat([ a_mask, a[..., :self.action_space_per]] , dim= -1)
            a_2q = torch.cat([ a_mask2, a[..., self.action_space_per:]] , dim= -1)

            # a_mask, a_mask2 = a_mask[..., 0], a_mask2[..., 0]
            mu, std = self.pi(s)
            dist = Normal(mu, std)
            a = dist.sample()

            log_prob = dist.log_prob(a)
            prob_a = log_prob.exp()

            a_mask, a_mask2, prob_m, prob_m2, alpha1, alpha2 = self.choose_action(s)

            a_mask, a_mask2 = torch.unsqueeze(a_mask, 2), torch.unsqueeze(a_mask2, 2)
            a_1 = torch.cat([a_mask, a[..., :self.action_space_per]], dim=-1)
            a_2 = torch.cat([a_mask2, a[..., self.action_space_per:]], dim=-1)

            prob_am = torch.gather(prob_m, 2, a_mask)
            prob_am2 = torch.gather(prob_m2, 2, a_mask2)
            q_1 = torch.unsqueeze(self.compute_Q(1, s, a_1q), dim= 2)
            q_2 = torch.unsqueeze(self.compute_Q(2, s, a_2q), dim=2)
            q1 = torch.unsqueeze(self.compute_Q(1, s, a_1), dim=2)
            q2 = torch.unsqueeze(self.compute_Q(2, s, a_2), dim=2)

            entropy1 = self.log_tau1.exp() * torch.cat([torch.log(prob_am), log_prob[..., :self.action_space_per]], dim = -1)
            entropy2 = self.log_tau2.exp() * torch.cat([torch.log(prob_am2), log_prob[..., self.action_space_per:]], dim= -1)
            log_prob1 =torch.cat([torch.log(prob_am), log_prob[..., :self.action_space_per]], dim = -1)

            log_prob2 = torch.cat([torch.log(prob_am2), log_prob[..., self.action_space_per:]], dim= -1)
            entropy1 = torch.squeeze(entropy1)
            entropy2 = torch.squeeze(entropy2)

            with torch.no_grad():

                a_prime_m1, a_prime_m2, probpm1, probpm2,_,_ = self.choose_action(s_prime)
                a_prime_m1 = torch.unsqueeze(a_prime_m1,2)
                a_prime_m2 = torch.unsqueeze(a_prime_m2,2)
                mu_, std_ = self.pi(s_prime)
                pi_prime = Normal(mu_, std_)
                a_prime = pi_prime.sample()
                a_prime1 = torch.cat([a_prime_m1, a_prime[..., :self.action_space_per]], dim=-1)
                a_prime2 = torch.cat([a_prime_m2, a_prime[..., self.action_space_per:]], dim=-1)
                q_prime1 = self.compute_Q(1, s_prime, a_prime1)
                q_prime2 = self.compute_Q(2, s_prime, a_prime2)
                q_prime1 = torch.unsqueeze(q_prime1, dim= 2)
                q_prime2= torch.unsqueeze(q_prime2, dim=2)
                q_target1 = r + gamma * done_mask * q_prime1
                q_target2 = r + gamma * done_mask * q_prime2



            pi_loss = entropy1 - q1 + entropy2 - q2

            tau1_loss = -(self.log_tau1.exp() * (log_prob1 + target_entropy).detach()).mean()
            tau2_loss = -(self.log_tau2.exp() * (log_prob2 + target_entropy).detach()).mean()
            tau_loss = tau1_loss + tau2_loss
            q_loss = F.mse_loss(q_1, q_target1) + F.mse_loss(q_2, q_target2)

            if torch.isnan(pi_loss.mean()).sum()>0 : print("1")

            if torch.isnan(tau1_loss).sum()>0: print("2")

            att_loss = []
            a_size = list(alpha1.shape[:-1])
            alpha1, alpha2 = alpha1.reshape([-1,4]), alpha2.reshape([-1, 4])
            for n in range(alpha1.shape[0]):

                att_loss.append((2 * (torch.dot(alpha1[n], alpha2[n].t())) **2).reshape((1,)))

            att_loss = torch.cat(att_loss, 0)
            att_loss = att_loss.view(a_size+[1])
            pi_loss += 0.05 * att_loss
            q_loss += 0.05 * att_loss.mean()

            return tau_loss, pi_loss, q_loss

    def train_net(self):
        print("length of data ", len(self.data))
        global  tb_epoch
        if len(self.data) == minibatch_size * buffer_size:
            print("start training")
            data = self.make_batch()


            for i in range(K_epoch):

                for mini_batch in data:

                    tau_loss, pi_loss, q_loss = self.calc_loss(mini_batch)
                    print(pi_loss)

                    writer.add_scalar('policy_loss', pi_loss.mean().item(), tb_epoch)
                    writer.add_scalar('q_loss', q_loss.item(), tb_epoch)
                    tb_epoch +=1
                    # print(print([parameters for parameters in self.parameters()]))
                    self.optimizer.zero_grad()
                    (pi_loss.mean()+q_loss).backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()

                    # loss2_2.mean().backward()




                    self.log_tau_optimizer.zero_grad()

                    tau_loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.log_tau_optimizer.step()




            print("training done")


def main():
    import imageio
    env = gym.make('bintaskenv-v0', double_agent=True, using_binary_task=True, need_GUI=False)
    # env = gym.make("blockenv-v0", render=True, double_agent=True)
    s = env.reset()
    # s = np.concatenate([s["observation"], s["achieved_goal"], s["desired_goal"]])
    model = DA(ob_space=s,
                action_space=(env.action_space[0].n, env.action_space[1].n, env.action_space.sample()[2].shape[0]))
    # tt = (env.action_space.sample()[0], env.action_space[1].n, env.action_space.sample()[2].shape)
    score = 0.0
    print_interval = 20
    rollout = []
    rewards_episode = []
    # imgs = []
    # imgs.append(env.render("rgb_array"))

    for n_epi in range(10000):
        s = env.reset()
        # s = np.concatenate([s["observation"], s["achieved_goal"], s["desired_goal"]])
        done = False
        rewards = 0

        while not done:

            for t in range(rollout_len):

                mu, std = model.pi(s)
                dist = Normal(mu, std)
                a = dist.sample()
                a_mask, a_mask2, prob, prob2,alpha1, alpha2 = model.choose_action(s)
                # index_mask = torch.tensor(a_mask.copy(), dtype = torch.float)
                prob_am = prob.gather(0, a_mask).reshape((1,))
                prob_am2 = prob2.gather(0, a_mask2).reshape((1,))
                log_prob = dist.log_prob(a)


                s_prime, r, done, info = env.step((a_mask.numpy(), a_mask2.numpy(), a.numpy()))
                # s_prime = np.concatenate([s_prime["observation"], s_prime["achieved_goal"], s_prime["desired_goal"]])
                rewards += r
                rollout.append((s, a.numpy(), a_mask.numpy(), a_mask2.numpy(), r/10.0, s_prime, log_prob.detach().numpy(), prob_am.detach().numpy(), prob_am2.detach().numpy(), done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                if done:

                    break
            #     imgs.append(env.render("rgb_array"))
            # imageio.mimsave("t.mp4", imgs)
            model.train_net()
            torch.save({"epoch": n_epi, 'state_dict': model.state_dict(), 'rewards': rewards}, "save.pth")
        rewards_episode.append(rewards)
        print(rewards_episode)


        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()




