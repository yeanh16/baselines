import gym

env = gym.make('BipedalWalker-v2')

env.reset()
terminated = False

while not terminated:
    action = env.action_space.sample()
    obs, rewards, terminated, info = env.step(action)
    env.render()
