
# This code was written by referring to the Pangyo Lab YouTube video.
# (https://www.youtube.com/watch?v=l1CZQWBkdcY&t=184s)


from turtle import done
from sklearn.metrics import det_curve
import torch
import env


class PPO():
    def __init__(self, n_epi = 1, n_iteration_per_one_game = 10000, learning_rate = 1e-4, gamma = 0.98):
        self.l1 = 100
        self.l2 = 150
        self.l3 = 100
        self.l4 = 4
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.data_list = []        
        self.state = None
        self.n_episodes = n_epi
        self.n_epochs = n_iteration_per_one_game
        self.game = None
        self.actor_model = None
        self.critic_model = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        # new hyper-parameter
        self.lmbda = 0.95
        self.epsilon = 0.1
        self.period_for_collecting_data = 200
        self.period_for_timeDelta = 3
        self.epsilon_for_clip = 0.1


    # actor와 critic의 input은 flatten된 1차원 텐서임
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
    
    def put_data(self, data):
        self.data_list.append(data)
    
    def train(self):
        s_matrix, a_matrix, r_matrix, s_prime_matrix, done_mask, probability_of_actions = self.create_batch()

        for i in range(self.period_for_timeDelta):
            td_target_matrix = r_matrix + self.gamma * self.critic_model(s_prime_matrix) * done_mask 
            # td_target = r + gamma * V
            # δ_t= r_t + γV * s_{t+1} − V(s_t)
            # A_t = δ_t + γλ * δ_{t+1} + ... + (γλ)^{T−t+1} * δ_{T− 1}
            #     = δ_t + γλA_{t+1}
            delta_matrix = td_target_matrix - self.critic_model(s_matrix)
            delta_matrix = delta_matrix.detach().numpy()

            advantage_matrix = self.generate_advantage_function_by_GAE(delta_matrix)

            probability_of_action = self.actor_model(s_matrix)
            probabilitys_of_choosed_action = torch.gather(probability_of_action, 1, a_matrix.type(torch.int64))

            # old policy와 policy의 비율로 업데이트 필요한 ratio 계산
            ratio = torch.exp( torch.log(probabilitys_of_choosed_action) - torch.log(probability_of_actions ) )

            surrogate_function1 = ratio * advantage_matrix
            surrogate_function2 = torch.clamp(ratio, 1 - self.epsilon_for_clip, 1 + self.epsilon_for_clip) * advantage_matrix


            # L^CLIP = Expeaction[ min[ ratio * A , clip(ratio, 1 - e,  1 + e) * A ]  ]
            loss = -torch.min(surrogate_function1, surrogate_function2) + \
                    torch.nn.functional.smooth_l1_loss(self.critic_model(s_matrix), td_target_matrix.detach())
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


    def generate_advantage_function_by_GAE(self, deltas):
        advantage_matrix = []
        advantage = 0

        for delta in deltas[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta[0]
            advantage_matrix.append([advantage])
        advantage_matrix.reverse()

        return torch.tensor(advantage_matrix, dtype=torch.float32)

    def create_batch(self):
        s_matrix, a_matrix, r_matrix, s_prime_matrix, done_matrix, probability_of_action_list = [], [], [], [], [], []
        
        for element in self.data_list:
            s, a, r, s_prime, probability_of_action, done = element

            s_matrix.append(s.reshape(1, self.l1)[0].numpy())
            a_matrix.append([a])
            r_matrix.append([r])
            s_prime_matrix.append(s_prime.reshape(1, self.l1)[0].numpy())
            probability_of_action_list.append(probability_of_action)
            done_mask = 0 if done else 1
            done_matrix.append([done_mask])

        s_matrix = torch.tensor(s_matrix, dtype=torch.float32)
        a_matrix = torch.tensor(a_matrix, dtype=torch.float32)
        r_matrix = torch.tensor(r_matrix, dtype=torch.float32)
        s_prime_matrix = torch.tensor(s_prime_matrix, dtype=torch.float32)
        done_matrix = torch.tensor(done_matrix, dtype=torch.float32)
        probability_of_action_list = torch.tensor(probability_of_action_list, dtype=torch.float32)

        self.data_list = []

        return s_matrix, a_matrix, r_matrix, s_prime_matrix, done_matrix, probability_of_action_list

    def run(self):
        self.game = env.GridWorldEnv()
        self.actor_model = self.actor()
        self.critic_model = self.critic()
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr = self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr = self.learning_rate)

        # 한 에피소드에서 
        # 설정한 step 만큼 데이터를 모은다. 
        # 모아야 하는 데이터 probability of action for old policy (ratio 계산때문), reward, action, state
        # GAE (Generalized Advantage Estimation) 사용해서 advantage function을 계산, GAE를 계산할때 사용하는 hyperPara가 lambda
        # ㄴ 행렬을 활용하여 계산
        # 각 스텝별로 모은 데이터를 활용해서 설정한 epochs 만큼 actor와 critic을 업데이트 한다.

        for epi in range(self.n_episodes):
            state = torch.from_numpy(self.game.get_grid_map()).float()
            score = 0.0           
            
            status = False          

            while not status:                                
                t = 0
                while t < self.period_for_collecting_data:
                    probability_of_action = self.actor_model(state.reshape(1, self.l1))

                    action = torch.distributions.Categorical(probability_of_action).sample().numpy()[0]

                    (new_state, reward, status) = self.game.step(action)                   

                    if reward == 0:
                        reward = -1 # 찾는 횟수가 증가하면 패널티를 주기 위함
                    
                    # print(self.loss_list)

                    if new_state == state:
                        next

                    t += 1
                    new_state = torch.FloatTensor(new_state)

                    self.put_data( (state, action, reward / 100.0, new_state, probability_of_action.squeeze(0)[action].item(), status) )

                    state = new_state
                    score += reward
                self.train()

            if epi%10 == 0:
                print(f"{epi} of episodes is ended")

            


    def test(self):
        status = False
        self.game.reset()
        self.game.render()

        state = torch.from_numpy(self.game.get_grid_map()).float()
        while not status:            
            probability_of_action = self.actor_model(state.reshape(1, self.l1))
            action = torch.distributions.Categorical(probability_of_action).sample().numpy()[0]

            print(f"probability_of_action : {probability_of_action}")
            print(f"action : {action}")

            (new_state, reward, status) = self.game.step(action)

            if new_state == state:
                next

            new_state = torch.FloatTensor(new_state)
            state = new_state
            self.game.render()