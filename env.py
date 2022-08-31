# render / step / render를 구현한다. 
# action_space.sample() : 임의의 행동을 선택

# obersvation_space
# observation


import copy
import matplotlib.pyplot as plt
import numpy as np
import random
from env_interface import Env_interface



# MDP
# (S, A, P, R, gamma) 
# S : state space, grid world에서는 격자판 자체가 상태
# A : Action space, 상하좌우
# P : transition kernel
# R : bounded reward function
# gamma : discount factor, 0 ~ 1,   환경 구성에는 필요없음.

class GridWorldEnv(Env_interface):
    def __init__(self):
        self.total_n_block = 2        
        
        # action
        self.action_space = [0, 1, 2, 3]
        self.action_to_dict = {0:[0, -1], 1: [-1, 0], 2:[0, 1], 3:[1, 0]}

        # agent는 값을 3으로
        self.agent_color = 3
        self.goal_color = 10

        # create grid : initial map
        self.grid_fig_size = (512, 512)
        self.number_of_grid = (10, 10)
        self.grid_map = np.zeros(self.number_of_grid)
        self.grid_fig_map = np.zeros(self.grid_fig_size)
        self.scale = (int(self.grid_fig_size[0] / self.number_of_grid[0]), int(self.grid_fig_size[1] / self.number_of_grid[1]))
        self.observation_space = [i for i in range(self.total_n_block)] # 이게 필요한 이유가 뭐지??        

        (self.grid_map, self.grid_fig_map, self.start_pos, self.goal_pos) = self._create_initial_map()
        self.initial_grid_map = self.grid_map
        self.observation = self.grid_fig_map

        self.agent_pos = self.start_pos # current_state

    def get_state(self):
        return self.agent_pos

    def get_observation(self):
        return self.observation

    def get_grid_map(self):
        return self.grid_map

    def show_action_direction(self, action):
        return self.action_to_dict[action]

    def _create_initial_map(self):
        weight = [np.exp(-i*2) for i in range(self.total_n_block)]

        # create initial map
        for i in range(self.number_of_grid[0]):
            for j in range(self.number_of_grid[1]):
                value = random.choices(self.observation_space, k=1, weights=weight)[0]
                self.grid_map[i, j] = value
                self.grid_fig_map[i*self.scale[0]:(i+1)*self.scale[0], j*self.scale[1]:(j+1)*self.scale[1]] = value

        # define start_pos and goal_pos
        self.start_pos = list(map(lambda x : x[0], np.where(self.grid_map == 0)))
        self.goal_pos = list(map(lambda x : x[-1], np.where(self.grid_map == 0)))

        # print(f"\t\t start state : {self.start_pos}")
        self.grid_map[self.start_pos[0], self.start_pos[1]] = self.agent_color
        self.grid_fig_map[ self.start_pos[0] * self.scale[0]: (self.start_pos[0] + 1) * self.scale[0], self.start_pos[1] * self.scale[1]: (self.start_pos[1] + 1) * self.scale[1] ] = self.agent_color
        self.grid_map[self.goal_pos[0], self.goal_pos[1]] = self.goal_color
        self.grid_fig_map[ self.goal_pos[0] * self.scale[0]: (self.goal_pos[0] + 1) * self.scale[0], self.goal_pos[1] * self.scale[1]: (self.goal_pos[1] + 1) * self.scale[1] ] = self.goal_color

        # print(f"start at {self.start_pos} , goal at {self.goal_pos}")

        return (self.grid_map, self.grid_fig_map, self.start_pos, self.goal_pos)

    def _gridmap_to_fig(self, grid_map):
        grid_fig = np.zeros(self.grid_fig_size)
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                grid_fig[i*self.scale[0]:(i+1)*self.scale[0], j*self.scale[1]:(j+1)*self.scale[1]] = grid_map[i, j]
        return grid_fig

    def render(self):    
        fig = plt.clf()
        plt.imshow(self._gridmap_to_fig(self.grid_map))
        plt.pause(0.1)
    
    def step(self, action):
        next_state = (self.agent_pos[0] + self.action_to_dict[action][0], self.agent_pos[1] + self.action_to_dict[action][1])
    
        if np.min(next_state) < 0 or next_state[0] >= self.number_of_grid[0] or next_state[1] >= self.number_of_grid[1]:
            # print("agent get out of grid!")
            return (self.grid_map, 0, False) # 경계를 넘어간 경우 보상은 0으로

        # curr_color = self.grid_map[self.agent_pos[0], self.agent_pos[1]]
        next_color = self.grid_map[next_state[0], next_state[1]]

        # print(f"next color : {next_color}")

        if next_color == 0:      
            # print("next color is not blocked")      
            # print(f"agent state : {self.agent_pos}, next state : {next_state}, initial_grid_map : {self.initial_grid_map}")
            
            self.initial_grid_map[np.where(self.initial_grid_map == self.agent_color)] = 0
            self.grid_map = self.initial_grid_map
            self.grid_map[next_state[0], next_state[1]] = self.agent_color
            self.agent_pos = next_state
            return (self.grid_map, -10, False)
        if next_color == 1:
            # print("blocked")
            return (self.grid_map, -100, False)

        if next_color == self.goal_color:
            # print("reach the goal")
            return (self.grid_map, 1000, True) # 차례로 (관측치, 보상, 게임이 종료되었는지)를 의미

        # print(f"next color is None : {next_state}")        
        return (self.grid_map, 0, False)

    def reset(self):
        self.agent_pos = self.start_pos
        self.grid_map = self.initial_grid_map
        self.observation = self._gridmap_to_fig(self.initial_grid_map)
        return self.observation

    def get_random_action(self):        
        return random.choice(self.action_space)


