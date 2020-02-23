import gym
import os
from time import sleep
from evolutionary_algorithm.ea.eamain import ConvolutionalNeuralNetwork, SimpleNeuralNetwork
from evolutionary_algorithm.ea.gym_wrapper import RamGymWrapper, MainGymWrapper
ENV_NAME = "SpaceInvaders-ramNoFrameskip-v4"
RAM = True
MODEL_USED = "SIMPLE"
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-20_04-35" + "-model.h5")


if RAM:
    env = RamGymWrapper.wrap(gym.make(ENV_NAME))
else:
    env = MainGymWrapper.wrap(gym.make(ENV_NAME))

if MODEL_USED == "SIMPLE":
    model = SimpleNeuralNetwork((4, 128), env.action_space.n, filepath=LOAD_WEIGHTS_PATH)
else:
    model = ConvolutionalNeuralNetwork((4, 84, 84), env.action_space.n, filepath=LOAD_WEIGHTS_PATH)

terminated = False
state = env.reset()
total_reward = 0

while not terminated:
    action = model.predict(state)
    state, reward, terminated,_ = env.step(action)
    total_reward += reward
    env.render()
    sleep(0.01)

print("Final reward: " + str(total_reward))
