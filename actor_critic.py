# The difference between Vanilla Policy Gradient (VPG) with a baseline as value function and Advantage Actor-Critic (A2C)
# A2C can learn during an episode which can lead to faster refinements in policy than with VPG.
# A2C can learn in continuing environments, whilst VPG cannot.
# A2C relies on initially biased value estimates, so can take more tuning to find hyperparameters for the agent that allows for stable learning. 
# Whilst VPG typically has higher variance and can require more samples to achieve the same degree of learning.

import torch
import env


class Actor_Critic():
    def __init__(self, n_epi = 1, n_iteration_per_one_game = 10000, learning_rate = 1e-4, gamma = 0.98):
        self.l1 = 100
        self.l2 = 150
        self.l3 = 100
        self.l4 = 4
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.data_list = []
        self.loss_list = []        
        self.state = None
        self.n_episodes = n_epi
        self.n_epochs = n_iteration_per_one_game
        self.game = None
        self.actor_model = None
        self.critic_model = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    def actor(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.l1, self.l2), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.l2, self.l3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.l3, self.l4),
            torch.nn.Softmax(dim=1)
        )
        return model

    def critic(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.l1, self.l2), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.l2, self.l3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.l3, 1)
        )
        return model

    def put_loss(self, loss):
        self.loss_list.append(loss)


    def train(self):
        loss = torch.cat(self.loss_list).sum()
        loss = loss/len(self.loss_list)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.loss_list = []

    def run(self):
        self.game = env.GridWorldEnv()
        self.actor_model = self.actor()
        self.critic_model = self.critic()
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr = self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr = self.learning_rate)

        # 한 에피소드에서 
        # 초기 actor와 critic으로 게임을 끝까지 (or 원하는 스텝까지) 진행하여 loss들을 모은다
        # 한 에피소드가 끝나면 모은 loss들을 평균내서 파라미터들을 업데이트

        for epi in range(self.n_episodes):
            state = torch.from_numpy(self.game.get_grid_map()).float()
            score = 0.0           
            
            status = False

            while not status:
                probability_of_action = self.actor_model(state.reshape(1, self.l1))
                value = self.critic_model(state.reshape(1, self.l1))

                action = torch.distributions.Categorical(probability_of_action).sample().numpy()[0]

                (new_state, reward, status) = self.game.step(action)

                next_value = self.critic_model(torch.Tensor(new_state.reshape(1, self.l1)))

                if reward == 0:
                    reward = -1 # 찾는 횟수가 증가하면 패널티를 주기 위함

                td_error = reward + self.gamma * next_value - value

                loss = -torch.log(probability_of_action.squeeze(0)[action]) * td_error.item() + td_error ** 2
                
                self.put_loss(loss.unsqueeze(0))
                # print(self.loss_list)

                if new_state == state:
                    next

                new_state = torch.FloatTensor(new_state)
                state = new_state
                score += reward
                print("=", end="")

            self.train() 


    def test(self):
        status = False
        self.game.reset()
        self.game.render()

        state = torch.from_numpy(self.game.get_grid_map()).float()
        while not status:            
            probability_of_action = self.actor_model(state.reshape(1, self.l1))
            action = torch.distributions.Categorical(probability_of_action).sample().numpy()[0]

            (new_state, reward, status) = self.game.step(action)

            if new_state == state:
                next

            new_state = torch.FloatTensor(new_state)
            state = new_state
            self.game.render()



