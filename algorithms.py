import numpy as np


class UpdateAlgorithm:
    def __init__(self, alpha=0.05, epsilon=0.01, gamma=0.5, state_space_size=(10, 7), num_actions=4, goal_state=(7, 3)):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.random.random(size=(*state_space_size, num_actions))
        self.q[goal_state[0], goal_state[1], :] = 0

    def action_selection(self, state):
        # epsilon-greedy action selection from q-table
        actions = self.q[state[0], state[1]]
        probability = np.array([self.epsilon/len(actions) for _ in actions])
        probability[np.argmax(actions)] += 1 - self.epsilon
        action = np.random.choice(len(actions), p=probability)
        return action


class Sarsa(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_table(self, state, action, reward, next_state, next_action):
        self.q[state[0], state[1], action] += self.alpha * (
                reward + self.gamma * self.q[next_state[0], next_state[1], next_action] -
                self.q[state[0], state[1], action])


class QLearning(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SarsaLambda(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class QLambda(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

