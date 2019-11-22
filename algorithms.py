import numpy as np


class UpdateAlgorithm:
    def __init__(self, env=None, alpha=0.05, epsilon=0.01, gamma=0.5,
                 table_init='zeros'):
        if env is None:
            raise ValueError('Environment is invalid.')
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        if table_init == 'random':
            self.q = np.random.random(size=(*env.size, env.num_actions))
            self.q[env.goal_state[0], env.goal_state[1], :] = 0
        elif table_init == 'zeros':
            self.q = np.zeros(shape=(*env.size, env.num_actions))
        self.z = np.zeros(shape=(*env.size, env.num_actions))

    def action_selection(self, state):
        """Epsilon-greedy action selection from q-table."""
        actions = self.q[state[0], state[1], :]
        probability = np.array([self.epsilon / len(actions) for _ in actions])
        probability[np.argmax(actions)] += 1 - self.epsilon
        return np.random.choice(len(actions), p=probability)

    def test(self):
        state = self.env.reset()
        episode = []
        done = False
        while not done:
            action = np.argmax(self.q[state[0], state[1], :])
            state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
        print('\tTook {} moves to reach the goal'.format(len(episode)))
        return episode

    def train_episode(self, t):
        raise NotImplementedError('Cannot call training on parent class')

    def train(self, num_episodes, epsilon=None):
        training = []
        for i in range(1, num_episodes + 1):
            if epsilon is not None:
                self.epsilon = epsilon / i
            episode = self.train_episode(i)
            training.append(episode)


class Sarsa(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_table(self, state, action, reward, next_state, next_action):
        self.q[state[0], state[1], action] += self.alpha * (
            reward + self.gamma * self.q[next_state[0], next_state[1], next_action] - self.q[state[0], state[1], action]
        )

    def train_episode(self, t):
        state = self.env.reset()
        action = self.action_selection(state)
        done = False
        episode = []
        print('Episode {}:'.format(t))
        while not done:
            next_state, reward, done = self.env.step(action)
            next_action = self.action_selection(next_state)
            self.update_table(state, action, reward, next_state, next_action)
            episode.append((state, action, reward))
            state = next_state
            action = next_action
        print('\tTook {} moves to reach the goal'.format(len(episode)))
        return episode


class QLearning(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_table(self, state, action, reward, next_state):
        self.q[state[0], state[1], action] += self.alpha * (
                reward + self.gamma * max(self.q[next_state[0], next_state[1], :]) - self.q[state[0], state[1], action]
        )

    def train_episode(self, t):
        state = self.env.reset()
        done = False
        episode = []
        print('Episode {}:'.format(t))
        while not done:
            action = self.action_selection(state)
            next_state, reward, done = self.env.step(action)
            self.update_table(state, action, reward, next_state)
            episode.append((state, action, reward))
            state = next_state
        print('\tTook {} moves to reach the goal'.format(len(episode)))
        return episode


class SarsaLambda(UpdateAlgorithm):
    def __init__(self, trace_decay=0.2, **kwargs):
        super().__init__(**kwargs)
        self.lambd = trace_decay

    def train_episode(self, t):
        state = self.env.reset()
        action = self.action_selection(state)
        done = False
        episode = []
        print('Episode {}:'.format(t))
        # z = np.zeros(shape=self.q.shape)
        while not done:
            next_state, reward, done = self.env.step(action)
            next_action = self.action_selection(next_state)
            delta = (reward +
                     self.gamma *
                     self.q[next_state[0], next_state[1], next_action] -
                     self.q[state[0], state[1], action]
                     )
            self.z[state[0], state[1], action] += 1
            self.q += self.alpha * delta * self.z
            self.z = self.gamma * self.lambd * self.z
            episode.append((state, action, reward))
            state = next_state
            action = next_action
        print('\tTook {} moves to reach the goal'.format(len(episode)))
        return episode


class QLambda(UpdateAlgorithm):
    def __init__(self, trace_decay=0.2, **kwargs):
        super().__init__(**kwargs)
        self.lambd = trace_decay

    def train_episode(self, t):
        state = self.env.reset()
        action = self.action_selection(state)
        done = False
        episode = []
        print('Episode {}:'.format(t))
        while not done:
            next_state, reward, done = self.env.step(action)
            next_action = self.action_selection(next_state)
            a_star = np.argmax(self.q[next_state[0], next_state[1], :])
            delta = (reward +
                     self.gamma *
                     self.q[next_state[0], next_state[1], next_action] -
                     self.q[state[0], state[1], a_star]
                     )
            self.z[state[0], state[1], action] += 1
            self.q += self.alpha * delta * self.z
            if next_action == a_star:
                self.z *= self.gamma * self.lambd
            else:
                self.z[:, :, :] = 0
            episode.append((state, action, reward))
            state = next_state
            action = next_action
        print('\tTook {} moves to reach the goal'.format(len(episode)))
        return episode
