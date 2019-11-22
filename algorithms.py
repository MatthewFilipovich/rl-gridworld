import numpy as np


class UpdateAlgorithm:
    def __init__(self, env=None, alpha=0.05, epsilon=0.01, gamma=0.5, table_init='zeros'):
        assert env is not None
        self.env = env
        state_space_size = env.size
        num_actions = env.num_actions
        goal_state = env.goal_state
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        if table_init == 'random':
            self.q = np.random.random(size=(*state_space_size, num_actions))
            self.q[goal_state[0], goal_state[1], :] = 0
        elif table_init == 'zeros':
            self.q = np.zeros(shape=(*state_space_size, num_actions))
        self.z = np.zeros(shape=(*state_space_size, num_actions))


    def action_selection(self, state, epsilon):
        # epsilon-greedy action selection from q-table
        actions = self.q[state[0], state[1], :]
        probability = np.array([epsilon/len(actions) for _ in actions])
        probability[np.argmax(actions)] += 1 - epsilon
        action = np.random.choice(len(actions), p=probability)
        return action


class Sarsa(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_table(self, state, action, reward, next_state, next_action):
        self.q[state[0], state[1], action] += self.alpha * (
            reward + self.gamma * self.q[next_state[0], next_state[1], next_action] - self.q[state[0], state[1], action]
        )

    def train(self, num_episodes, epsilon):
        training = []
        for i in range(1, num_episodes + 1):
            self.epsilon = epsilon / i
            state = self.env.reset()
            action = self.action_selection(state, self.epsilon)
            done = False
            episode = []
            print('Episode {}:'.format(i))
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.action_selection(next_state, self.epsilon)  # why is action selected when in terminal state?
                self.update_table(state, action, reward, next_state, next_action)
                episode.append((state, action, reward))
                state = next_state
                action = next_action
            training.append(episode)
            print('\tTook {} moves to reach the goal'.format(len(episode)))
        return training


class QLearning(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_table(self, state, action, reward, next_state):
        self.q[state[0], state[1], action] += self.alpha * (
                reward + self.gamma * max(self.q[next_state[0], next_state[1], :]) - self.q[state[0], state[1], action]
        )

    def train(self, num_episodes, epsilon):
        training = []
        for i in range(1, num_episodes + 1):
            #self.epsilon = epsilon / i
            state = self.env.reset()
            done = False
            episode = []
            print('Episode {}:'.format(i))
            while not done:
                action = self.action_selection(state, self.epsilon)
                next_state, reward, done = self.env.step(action)
                self.update_table(state, action, reward, next_state)
                episode.append((state, action, reward))
                state = next_state
            training.append(episode)
            print('\tTook {} moves to reach the goal'.format(len(episode)))
        return training


class SarsaLambda(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, num_episodes):
        training = []
        for i in range(1, num_episodes + 1):
            state = self.env.reset()
            action = self.action_selection(state, self.epsilon)
            done = False
            episode = []
            print('Episode {}:'.format(i))
            z = np.zeros()
            while not done:
                next_state, reward, done = self.env.step(action)
                delta = reward
                for j in F(state,action):
                    delta -= w
                    z += 1
                if done:
                    w += self.alpha * delta * z
                    break
                next_action = self.action_selection(next_state, 0.0001)
                for j in F(next_state,next_action):
                    delta += self.gamma * w
                w += self.alpha * delta * z
                z = self.gamma * lambd * z
                state = next_state
                action = next_action


class QLambda(UpdateAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_table(self, state, action, reward, next_state):
        self.q[state[0], state[1], action] += self.alpha * (
                reward + self.gamma * max(self.q[next_state[0], next_state[1], :]) - self.q[state[0], state[1], action]
        )

    def train(self, num_episodes, epsilon, lam):
        training = []
        for i in range(1, num_episodes + 1):
            self.epsilon = epsilon / i
            state = self.env.reset()
            done = False
            episode = []
            action = self.action_selection(state, self.epsilon)
            print('Episode {}:'.format(i))
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.action_selection(next_state, self.epsilon)
                a_star = np.argmax(self.q[next_state[0]][next_state[1]])
                delta = (reward + self.gamma * self.q[next_state[0]][next_state[1]][next_action] 
                        - self.q[state[0]][state[1]][a_star])
                self.z[state[0]][state[1]][action] += 1
                for i in range(self.q.shape[0]):
                    for j in range(self.q.shape[1]):
                        for k in range(self.q.shape[2]):
                            self.q[i][j][k] += self.alpha*delta*self.z[i][j][k]
                            if next_action == a_star:
                                self.z[i][j][k] *= self.gamma * lam      
                            else:
                                self.z[i][j][k] = 0
                state = next_state
                action = next_action
                episode.append((state, action, reward))
            training.append(episode)
            print('\tTook {} moves to reach the goal'.format(len(episode)))
        return training
