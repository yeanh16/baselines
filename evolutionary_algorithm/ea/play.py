"""
Notable models
830 SIMPLE: SpaceInvaders-ramNoFrameskip-v4 (in one generation!)
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-23_04-44" + "-model.h5")


720 SIMPLE SpaceInvaders-ramNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-20_04-35" + "-model.h5")

1025 SIMPLE SpaceInvaders-ramNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-24_03-21" + "-model.h5")

695 CNN SpaceInvadersNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-24_03-21" + "-model.h5")

"""

import gym
import os
from time import sleep
from evolutionary_algorithm.ea.eamain import ConvolutionalNeuralNetwork, SimpleNeuralNetwork
from evolutionary_algorithm.ea.gym_wrapper import RamGymWrapper, MainGymWrapper
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.animation as animation
import numpy as np


ENV_NAME = "SpaceInvadersNoFrameskip-v4"
RAM = False
SHOW_RAM = False
MODEL_USED = "CNN"
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-24_20-52" + "-model.h5")


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
reward = 0

if SHOW_RAM:
    while not terminated:

        showstate = np.array(state)[3]
        showstate = np.pad(showstate, (0, 16), 'constant', constant_values=(0, 0))
        showstate = showstate.reshape((12,12))

        fig = pyplot.figure()

        action = model.predict(state)

        def animate(i):
            global state, reward, terminated, action, total_reward
            state, reward, terminated, _ = env.step(action)
            env.render()
            total_reward += reward

            cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                 ['white', 'black'],
                                                                 256)

            img2 = pyplot.imshow(showstate, interpolation='nearest',
                                 cmap=cmap2,
                                 origin='lower')

            # pyplot.colorbar(img2, cmap=cmap2)


        ani = animation.FuncAnimation(fig, animate, interval=1)
        pyplot.show()

else:
    while not terminated:
        action = model.predict(state)
        state, reward, terminated, _ = env.step(action)
        env.render()
        #sleep(0.001)
        total_reward += reward

print("Final reward: " + str(total_reward))
