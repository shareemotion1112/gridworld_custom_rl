import torch
import env
import random
import numpy as np

class DQN():
    def __init__(self):
        self.l1 = 100
        self.l2 = 150
        self.l3 = 100
        self.l4 = 4

    def model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.l1, self.l2), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.l2, self.l3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.l3, self.l4)
        )
        return model


    def run(self):

        model = self.model()

        loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        gamma = 0.9
        epsilon = 0.5

        [epoch, epochs] = [0, 10]
        losses = []

        game = env.GridWorldEnv()

        state = torch.from_numpy(game.get_grid_map()).float()
        print(f"initial map : \n {game.get_grid_map()}")

        while epoch < epochs:
            epoch += 1
            print(f"epoch : {epoch}")
            game.render()
            status = False

            while not status:        
                q_val = model(state.reshape(1, self.l1)).data.numpy()
                
                if (random.random() < epsilon / epoch):
                    action = game.get_random_action()
                else:
                    action = np.argmax(q_val)
                print(f"action : {game.show_action_direction(action)}")

                (new_state, reward, status) = game.step(action)
                print(new_state)

                if new_state == state:
                    next;

                new_state = torch.FloatTensor(new_state)

                # print(f"new map : \n {new_state}")
                
                # new_q = model(new_state.reshape(1, self.l1)).data.numpy()
                max_q = torch.max(model(new_state.reshape(1, self.l1)))

                Y =  reward + gamma * max_q

                # X = torch.FloatTensor([q_val.squeeze()[action]])
                X = torch.Tensor(model(state.reshape(1, self.l1))[0][action])

                # print(f"\t\t X : {X}, Y: {Y} \n")

                loss = loss_fn(X, Y)

                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                state = new_state
                game.render()
