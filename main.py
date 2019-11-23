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


def main(env, algorithm, num_episodes, alpha, epsilon, gamma, lambd, decaying_epsilon):
    if algorithm is 'Sarsa':
        rl = Sarsa(env=env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                   table_init='zeros')
    elif algorithm is 'Q-learning':
        rl = QLearning(env=env, alpha=alpha, epsilon=epsilon,
                       gamma=gamma, table_init='zeros')
    elif algorithm is 'Sarsa-lambda':
        rl = SarsaLambda(trace_decay=lambd, env=env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                         table_init='zeros')
    elif algorithm is 'Q-lambda':
        rl = QLambda(trace_decay=lambd, env=env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                     table_init='zeros')
    else:
        raise ValueError('Invalid Algorithm Choice')
    if decaying_epsilon:
        rl.train(num_episodes, epsilon=0.9)
    else:
        rl.train(num_episodes)
    rl.test()
    print('done')


if __name__ == '__main__':
    env = GridWorld(print_board=False)
    algorithm = ALG_NAMES[3]
    num_episodes = 1000
    alpha = 0.4
    epsilon = 0.1
    gamma = 1
    lambd = 0.5
    decaying_epsilon = True
    main(env, algorithm, num_episodes, alpha, epsilon, gamma, lambd, decaying_epsilon)

