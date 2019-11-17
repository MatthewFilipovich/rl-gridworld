from gridworld import GridWorld

env = GridWorld()
state = env.reset()
for _ in range(1000):
    # action = RL(state, reward) <---- RL Algorithm Implementation
    state, reward, done = env.step(action)
    if done:
        state = env.reset()
