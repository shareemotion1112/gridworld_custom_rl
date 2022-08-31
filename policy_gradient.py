import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import env
import matplotlib.pylab as plt


class Policy_Gradient:
    def __init__(self):
        self.l1 = 100
        self.l2 = 150
        self.l3 = 100
        self.l4 = 4
        self.learning_rate = 1e-4
        self.gamma = 0.98
        self.data = []
        self.policy = None
        self.loss_fn = None
        self.optimizer = None
        self.state = None
        self.n_episodes = 100
        self.n_epochs = 1000

    def model(self):
        model = torch.nn.Sequential( 
            torch.nn.Linear(self.l1, self.l2), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.l2, self.l3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.l3, self.l4),
            torch.nn.Softmax(dim=1)
        )
        return model

    def put_data(self, data):
        self.data.append(data)

    def train(self):
        R = 0 
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = - torch.log(prob) * R
            self.optimizer.zero_grad()
            loss.backward() # gradient값이 더해짐
            self.optimizer.step() # theta값들이 업데이트
        self.data = []

    def run(self):
        game = env.GridWorldEnv()
        self.policy = self.model()
        self.loss_fn = torch.nn.MSELoss()        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)

        for epi in range(self.n_episodes):
            state = torch.from_numpy(game.get_grid_map()).float()
            score = 0.0
            epoch = 0
            while epoch < self.n_epochs:
                epoch += 1
                game.render()
                status = False

                while not status:
                    prob_action = self.policy(self.state.reshape(1, self.l1))
                    action = Categorical(prob_action).sample()

                    (new_state, reward, status) = game.step(action)

                    if reward == 0:
                        reward = -1 # 찾는 횟수가 증가하면 패널티를 주기 위함

                    self.policy.put_data((reward, prob_action))

                    if new_state == state:
                        next

                    new_state = torch.FloatTensor(new_state)
                    state = new_state
                    score += reward
            self.policy.train()




