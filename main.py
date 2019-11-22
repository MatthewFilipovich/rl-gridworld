from gridworld import GridWorld
from algorithms import Sarsa, QLearning, SarsaLambda, QLambda
import math

alg_names = ['Sarsa',
             'Q-learning',
             'Sarsa-lambda',
             'Q-lambda']

algorithm = alg_names[0]


def run_sarsa(env, num_episodes, alpha, epsilon, gamma):
    rl = Sarsa(env=env,
               alpha=alpha,
               epsilon=epsilon,
               gamma=gamma,
               table_init='zeros')

    return rl.train(num_episodes, epsilon)


def run_qlearning(env, num_episodes, alpha, epsilon, gamma):
    rl = QLearning(env=env,
                   alpha=alpha,
                   epsilon=epsilon,
                   gamma=gamma,
                   table_init='zeros')
    return rl.train(num_episodes, epsilon)


def run_sarsalambda():
    raise NotImplementedError


def run_qlambda(env, num_episodes, alpha, epsilon, gamma, lam):
    rl = QLambda(env=env,
                   alpha=alpha,
                   epsilon=epsilon,
                   gamma=gamma,
                   table_init='zeros')
    return rl.train(num_episodes, epsilon, lam)


def plot_results(training):
    epsiode_lengths = [len(ep) for ep in training]
    print(epsiode_lengths)   # just printing len of episodes currently


if __name__ == '__main__':
    env = GridWorld(print_board=False)
    num_episodes = 1000
    alpha = 0.05
    epsilon = 0.1
    gamma = 0.5
    lam = 0.5

    if algorithm is 'Sarsa':
        episodes = run_sarsa(env, num_episodes, alpha, epsilon, gamma)
        plot_results(episodes)
    elif algorithm is 'Q-learning':
        episodes = run_qlearning(env, num_episodes, alpha, epsilon, gamma)
        plot_results(episodes)
    elif algorithm is 'Sarsa-lambda':
        run_sarsalambda()
    elif algorithm is 'Q-lambda':
        episodes = run_qlambda(env, num_episodes, alpha, epsilon, gamma, lam)
        plot_results(episodes)
    else:
        print('No Algorithm Selected')
