import numpy as np


class UpdateAlgorithm:
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions, goal_state):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.random.random(size=(*grid_size, num_actions))
        self.q[goal_state[0], goal_state[1], :] = 0

    def action_selection(self, state):
        # epsilon-greedy action selection from q-table
        actions = self.q[state[0], state[1]]
        probability = np.array([self.epsilon/len(actions) for _ in actions])
        probability[np.argmax(actions)] += 1 - self.epsilon
        action = np.random.choice(len(actions), p=probability)
        return action


class Sarsa(UpdateAlgorithm):
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions, goal_state):
        super().__init__(alpha, epsilon, gamma, grid_size, num_actions, goal_state)

    def update_table(self, state, action, reward, next_state, next_action):
        self.q[state[0], state[1], action] += self.alpha * (
                reward + self.gamma * self.q[next_state[0], next_state[1], next_action] -
                self.q[state[0], state[1], action])


class Qlearning(UpdateAlgorithm):
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions, goal_state):
        super().__init__(alpha, epsilon, gamma, grid_size, num_actions, goal_state)


class SarsaLambda(UpdateAlgorithm):
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions, goal_state):
        super().__init__(alpha, epsilon, gamma, grid_size, num_actions, goal_state)


class QLambda(UpdateAlgorithm):
    def __init__(self, alpha, epsilon, gamma, grid_size, num_actions, goal_state):
        super().__init__(alpha, epsilon, gamma, grid_size, num_actions, goal_state)

