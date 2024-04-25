from flask import Flask, jsonify, request
from torch import optim
import copy
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

gamma = 0.9
tau = 0.01
epsilon_start = 0.6
epsilon_decay = 0.9991
epsilon_min = 0.03
critic_learning_rate = 0.02
actor_learning_rate = 0.02
num_states = 8
num_output = 1
hidden_size = 64
batch_size = 128
num_actions = 2
max_memory_size = 50000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states,dtype=np.float64), np.array(actions,dtype=np.float64), np.array(rewards,dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states,dtype=np.float64)

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,output_size)

    def forward(self, state):
        #print(f"actor forward final shape:{state.shape}")
        x = F.relu(self.linear1(state.float()))
        x = F.relu(self.linear2(x))
        #x = torch.tanh(self.linear3(x))
        x = torch.mul(torch.tanh(self.linear3(x)), 3)
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,output_size)

    def forward(self, state, action):
        #print(f"critic state forward shape:{state.shape}")
        #print(f"critic action forward shape:{action.shape}")
        x = torch.cat([state, action], dim=2).float()
        #print(f"critic forward final shape:{x.shape}")
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DDPG:
    def __init__(self, buffer, ep_start):
        self.buffer = buffer
        self.epsilon = ep_start
        self.state = None

    # def _reset(self):
    #     self.state = None

    def select_action(self, actor_model):
        self.epsilon = max(self.epsilon*epsilon_decay,epsilon_min)
        if np.random.random() < self.epsilon:
            action = [random.uniform(-3, 3), random.uniform(-3, 3)]
        else:
            state = torch.tensor(self.state).float().unsqueeze(0).to(device)
            action = actor_model(state).cpu().detach().numpy()[0].tolist()
            print(action)
        return action

    def update_weights(self, actor_model, actor_target, critic_model, critic_target):
        batch = buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch
        states_t = torch.nn.functional.normalize(torch.tensor(states)).to(device)
        next_states_t = torch.nn.functional.normalize(torch.tensor(next_states)).to(device)
        actions_t = torch.nn.functional.normalize(torch.tensor(actions)).to(device)
        rewards_t = torch.tensor(rewards).to(device)
        done_mask = torch.tensor(dones).to(device)
        #
        q_values = critic_model(states_t,actions_t)
        next_actions = actor_target(next_states_t)
        q_next = critic_target(next_states_t,next_actions.detach())
        q_next = q_next.detach()
        q_targets = q_next*gamma*(1-done_mask) + rewards_t
        #
        critic_loss = nn.MSELoss()(q_values, q_targets)
        actor_loss = -critic_model(states_t, actor_model(states_t)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    def update_target_weights(self, actor_model, actor_target, critic_model, critic_target):
        for target_param, param in zip(actor_target.parameters(), actor_model.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

        for target_param, param in zip(critic_target.parameters(), critic_model.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
        print("target weights updated")

actor = Actor(num_states, hidden_size, num_actions).to(device)
actor_target = copy.deepcopy(actor).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic = Critic(num_states + num_actions, hidden_size, num_output).to(device)
critic_target = copy.deepcopy(critic).to(device)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate)

buffer = ExperienceReplay(max_memory_size)
agent = DDPG(buffer, epsilon_start)
episode_rewards = []
Experience = collections.namedtuple('Experience',field_names=['state', 'action', 'reward', 'done', 'new_state'])

@app.route('/check')
def check():
    return jsonify(200)

@app.route('/get_action', methods=["POST"])
def get_action():
    state = request.get_json()
    joint_sine, joint_cosine, point_sine, point_cosine, section_ang_speed, section_2_ang_speed, distance_x, distance_z = state['joint_sine'], state['joint_cosine'], state['point_sine'], state['point_cosine'], state['section_ang_speed'], state['section_2_ang_speed'], state['distance_x'], state['distance_z']
    agent.state = np.array([joint_sine, joint_cosine, point_sine, point_cosine, section_ang_speed, section_2_ang_speed, distance_x, distance_z],dtype=np.float32)
    action = agent.select_action(actor)
    return jsonify({"action_0":action[0], "action_1":action[1]})

@app.route('/get_experience', methods=["POST"])
def get_experience():
    experience = request.get_json()
    state = np.array(list(experience[0].values()), dtype=np.float64).reshape(1,-1)
    action = np.array(list(experience[1].values())).reshape(1,-1)
    reward = np.array(list(experience[2].values())).reshape(1,-1)
    terminate = np.array(list(experience[3].values())).reshape(1,-1)
    next_state = np.array(list(experience[4].values()), dtype=np.float64).reshape(1,-1)
    exp = Experience(state, action, reward, terminate, next_state)
    buffer.append(exp)
    print(len(buffer))
    return jsonify({"length":len(buffer)})

@app.route('/update_weights', methods=["GET"])
def update_weights():
    agent.update_weights(actor, actor_target, critic, critic_target)
    return jsonify(200)

@app.route('/update_target_weights', methods=["GET"])
def update_target_weights():
    agent.update_target_weights(actor, actor_target, critic, critic_target)
    return jsonify(200)

@app.route('/save_weights', methods=["GET"])
def save_weights():
    torch.save(actor.state_dict(), "actor.pth")
    torch.save(critic.state_dict(), "critic.pth")
    print("env solved, weights saved!")
    return jsonify(200)

if __name__ == "__main__":
    app.run()
