from gridworld import GridWorld

env = GridWorld()
state = env.reset()
rl = RL()
for _ in range(1000):
    action = rl.action_selection(state, reward) 
    state, reward, done = env.step(action)
    if done:
        state = env.reset()
