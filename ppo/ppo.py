import torch
import os
from ppo.actor_critic import Actor, Critic
import numpy as np

class PPO:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.clip = 0.2
        # create the network
        self.actor_network = Actor(args, agent_id)
        self.actor_network_old = Actor(args, agent_id)
        self.critic_network = Critic(args,agent_id)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # soft update
    def update_network(self):
        for target_param, param in zip(self.actor_network_old.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项

        o.append(transitions['o_%d' % self.agent_id])
        u.append(transitions['u_%d' % self.agent_id])
        o_next.append(transitions['o_next_%d' % self.agent_id])

        # calculate the target Q value function
        u_next = []
        self.update_network()
        with torch.no_grad():

            q_next = self.critic_network(o_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o)

        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变

        mu,sigma = self.actor_network(o[0])
        pi = torch.distributions.Normal(mu,sigma).sample()


        old_mu,old_sigma = self.actor_network_old(o[0])
        old_pi = torch.distributions.Normal(old_mu,old_sigma).sample()

        ratio = (pi/old_pi)
        tfadv = (target_q - q_value)
        clipped_ratio = torch.clamp(ratio, 1. - self.clip, 1. + self.clip)
        surr = ratio * tfadv
        clipped_surr = clipped_ratio * tfadv
        min_surr = torch.min(surr, clipped_surr)
        actor_loss = -torch.mean(min_surr)

        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        Actor_path = model_path + '/actor/'
        Critic_path = model_path + '/critic/'
        if not os.path.exists(Actor_path):
            os.makedirs(Actor_path)
        if not os.path.exists(Critic_path):
            os.makedirs(Critic_path)
        

        if len(os.listdir(model_path+'/actor/')) > self.args.max_file_saved:
            file_lst = os.listdir(model_path+'/actor/')
            file_lst.sort(key=lambda x: int(x.split('_')[0]))
            os.remove(model_path + '/actor/' + file_lst[0])

        if len(os.listdir(model_path+'/critic/')) > self.args.max_file_saved:
            file_lst = os.listdir(model_path+'/critic/')
            file_lst.sort(key=lambda x: int(x.split('_')[0]))
            os.remove(model_path + '/critic/' + file_lst[0])

        torch.save(self.actor_network.state_dict(), model_path + '/actor/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/critic/' + num + '_critic_params.pkl')

