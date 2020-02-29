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

875 SIMPLE(512, 255, 128, 128) SpaceInvadersNoFrameskip-v4 stable model no ram
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-26_10-09" + "-model.h5")

Builder: It looks like the optimum number of layers is 3 when there are 127 nodes
SpaceInvadersFrameskip-v4
[3, 127, -0.07942180342399471, 0.3723077140562673, 548465349, 127, 0.5642988202272742, -0.01218497502158078, 1983418018, 127, 0.6018992809267186, 0.7353135712463283, 486090374]
reward: 860.0

SpaceInvaders-ramNoFrameskip-v4
[3, 127, -0.00802600960001551, 0.37193787474935996, 1863185804, 127, 0.6205937796994904, -0.10831157420869357, 1879081302, 127, 0.9130001984849104, 0.6419841067735436, 1780189885]
reward: 940.0 NON STABLE


"""

import gym
import os
from time import sleep
from evolutionary_algorithm.ea.eamain import ConvolutionalNeuralNetwork, SimpleNeuralNetwork
from evolutionary_algorithm.ea.eabuilder import BuildNeuralNetwork
from evolutionary_algorithm.ea.gym_wrapper import RamGymWrapper, MainGymWrapper
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.animation as animation
import numpy as np


ENV_NAME = "SpaceInvadersNoFrameskip-v4"
if "-ram" in ENV_NAME:
    RAM = True
    SHOW_RAM = False
    INPUT_SHAPE = (4, 128)
else:
    RAM = False
    SHOW_RAM = False
    INPUT_SHAPE = (4, 84, 84)

MODEL_USED = "CNN" # SIMPLE / CNN / BUILDER
BUILDER_MODEL_TYPE = "CNN" # SIMPLE OR CNN (only matters for BUILDER models)
CHROMOSOME = [4, 127, 0.16762467326820363, -0.30084199831130487, 602418869, 127, 0.579589063575082, 0.08030824452678353, 471575520, 127, 0.9061998971422971, -0.04686307526807765, 86548449, 127, 0.13744530377945674, 0.27779087866482044, 1170568538]
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-29_10-08" + "-model.h5")


if RAM:
    env = RamGymWrapper.wrap(gym.make(ENV_NAME))
else:
    env = MainGymWrapper.wrap(gym.make(ENV_NAME))

if MODEL_USED == "SIMPLE":
    model = SimpleNeuralNetwork(INPUT_SHAPE, env.action_space.n, filepath=LOAD_WEIGHTS_PATH)
elif MODEL_USED == "BUILDER":
    model = BuildNeuralNetwork(INPUT_SHAPE, env.action_space.n, CHROMOSOME, model_type=BUILDER_MODEL_TYPE)
elif MODEL_USED == "CNN":
    model = ConvolutionalNeuralNetwork(INPUT_SHAPE, env.action_space.n, filepath=LOAD_WEIGHTS_PATH)

terminated = False
state = env.reset()
total_reward = 0
reward = 0

if SHOW_RAM:
    while not terminated:
        fig = pyplot.figure()
        action = model.predict(state)

        def animate(i):
            global state, reward, terminated, action, total_reward
            state, reward, terminated, _ = env.step(action)
            env.render()
            total_reward += reward

            showstate = np.array(state)[3]
            showstate = np.pad(showstate, (0, 16), 'constant', constant_values=(0, 0))
            showstate = showstate.reshape((12, 12))

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
