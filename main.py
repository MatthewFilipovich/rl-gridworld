from gridworld import GridWorld
from algorithms import Sarsa, QLearning, SarsaLambda, QLambda
import math

ALG_NAMES = ['Sarsa',
             'Q-learning',
             'Sarsa-lambda',
             'Q-lambda']


def plot_results(training):
    epsiode_lengths = [len(ep) for ep in training]
    print(epsiode_lengths)  


if __name__ == '__main__':
    env = GridWorld(print_board=False)
    algorithm = ALG_NAMES[0]
    num_episodes = 500
    alpha = 0.5
    epsilon = 0.1
    gamma = 1
    lam = 0.5

    if algorithm is 'Sarsa':
        rl = Sarsa(env=env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                   table_init='zeros')
    elif algorithm is 'Q-learning':
        rl = QLearning(env=env, alpha=alpha, epsilon=epsilon,
                       gamma=gamma, table_init='zeros')
    elif algorithm is 'Sarsa-lambda':
        rl = SarsaLambda(trace_decay=lam, env=env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                         table_init='zeros')
    elif algorithm is 'Q-lambda':
        rl = QLambda(trace_decay=lam, env=env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                     table_init='zeros')
    else:
        raise ValueError('Invalid Algorithm Choice')
    rl.train(num_episodes)
    rl.test()
    print('done')
