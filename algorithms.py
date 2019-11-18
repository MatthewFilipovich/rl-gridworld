import numpy as np
import random

class UpdateAlgorithm:
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.random.random(size=(*grid_size, num_actions))

    def action_selection(self, state):
        # eps-greedy action selection from q-table



class Sarsa(UpdateAlgorithm):
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions, goal_state):
        super().__init__(alpha, epsilon, gamma, grid_size, num_actions)
        self.q[goal_state[0], goal_state[1], :] = 0

    def update_table(self, reward, state, action, next_state, next_action):
        self.q[state[0], state[1], action] += self.alpha * (reward +
                                                            self.gamma * self.q[next_state[0], next_state[1], next_action] -
                                                            self.q[state[0], state[1], action])
