import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import env
import matplotlib.pylab as plt


class Policy_Gradient:
    def __init__(self, n_epi = 1, n_iteration_per_one_game = 10000):
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
        self.n_episodes = n_epi
        self.n_epochs = n_iteration_per_one_game
        self.game = None

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
            # 리턴 계산이 어려워서 게임을 한번 끝까지 보내고 끝에서부터 리턴을 계산하여 파라미터 업데이트
            # 이렇게 해도 되고, sampling 해서 평균으로 진행해도 됨
            R = r + self.gamma * R
            loss = - torch.log(prob) * R
            self.optimizer.zero_grad()
            loss.backward() # gradient값이 더해짐
            self.optimizer.step() # theta값들이 업데이트
        self.data = []

    def run(self):
        self.game = env.GridWorldEnv()
        self.policy = self.model()
        self.loss_fn = torch.nn.MSELoss()        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)

        for epi in range(self.n_episodes):
            state = torch.from_numpy(self.game.get_grid_map()).float()
            score = 0.0           
            
            status = False

            while not status:
                prob_action = self.policy(state.reshape(1, self.l1))
                action = Categorical(prob_action).sample().numpy()[0]

                (new_state, reward, status) = self.game.step(action)

                if reward == 0:
                    reward = -1 # 찾는 횟수가 증가하면 패널티를 주기 위함

                self.put_data((reward, prob_action))

                if new_state == state:
                    next

                new_state = torch.FloatTensor(new_state)
                state = new_state
                score += reward
                print("=", end="")
            self.policy.train() 

    def test(self):
        status = False
        self.game.reset()
        self.game.render()

        state = torch.from_numpy(self.game.get_grid_map()).float()
        while not status:            
            prob_action = self.policy(state.reshape(1, self.l1))
            action = Categorical(prob_action).sample().numpy()[0]

            (new_state, reward, status) = self.game.step(action)

            if new_state == state:
                next

            new_state = torch.FloatTensor(new_state)
            state = new_state
            self.game.render()



