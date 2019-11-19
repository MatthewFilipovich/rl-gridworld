from gridworld import GridWorld
from algorithms import Sarsa, Qlearning, SarsaLambda, QLambda

alg_names = ['Sarsa',
             'Q-learning',
             'Sarsa-lambda',
             'Q-lambda']
algorithm = alg_names[0]


def run_sarsa():
    env = GridWorld()
    num_episodes = 1000
    alpha = 0.05
    epsilon = 0.05
    gamma = 0.95
    grid_size = (env.width, env.height)
    num_actions = env.num_actions
    goal_state = env.goal_state

    rl = Sarsa(alpha, epsilon, gamma, grid_size, num_actions, goal_state)

    training = []
    for _ in range(num_episodes):
        state = env.reset()
        action = rl.action_selection(state)
        done = False
        episode = []
        while not done:
            next_state, reward, done = env.step(action)
            next_action = rl.action_selection(next_state)  # why is action selected when in terminal state?
            rl.update_table(state, action, reward, next_state, next_action)
            episode.append((state, action, reward))
            state = next_state
            action = next_action
        training.append(episode)

    return training


def run_qlearning():
    raise NotImplementedError


def run_sarsalambda():
    raise NotImplementedError


def run_qlambda():
    raise NotImplementedError


if __name__ == '__main__':
    if algorithm is 'Sarsa':
        run_sarsa()
    elif algorithm is 'Q-learning':
        run_qlearning()
    elif algorithm is 'Sarsa-lambda':
        run_sarsalambda()
    elif algorithm is 'Q-lambda':
        run_qlambda()
    else:
        print('No Algorithm Selected')
