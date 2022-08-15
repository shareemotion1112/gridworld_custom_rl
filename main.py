import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['OPENBLAS_NUM_THREADS'] = '8' 
import torch
import env
import random
import matplotlib.pylab as plt




l1 = 100
l2 = 150
l3 = 100
l4 = 4


model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2), 
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

gamma = 0.9
epsilon = 0.001


[epoch, epochs] = [0, 1000]
losses = []

from importlib import reload; reload(env)
game = env.GridWorldEnv()
state = torch.from_numpy(game.get_grid_map()).float()
print(f"initial map : \n {game.get_grid_map()}")

while epoch < epochs:
    epoch += 1
    print(f"epoch : {epoch}")
    game.render()
    status = False
    while not status:
        q_val = model(state.reshape(1, l1)).data.numpy()
        
        if (random.random() < epsilon):
            action = game.get_random_action()
        else:
            action = np.argmax(q_val)

        # print(f"action : {game.show_action_direction(action)}")

        (new_state, reward, status) = game.step(action)
        new_state = torch.FloatTensor(new_state)

        # print(f"new map : \n {new_state}")
        
        # new_q = model(new_state.reshape(1, l1)).data.numpy()
        max_q = torch.max(model(new_state.reshape(1, l1)))

        Y =  reward + gamma * max_q

        # X = torch.FloatTensor([q_val.squeeze()[action]])
        X = torch.Tensor(model(state.reshape(1, l1))[0][action])

        print(f"\t\t X : {X}, Y: {Y} \n")

        loss = loss_fn(X, Y)

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        state = new_state

        print(status)