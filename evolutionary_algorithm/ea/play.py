"""
Notable models
830 SIMPLE: SpaceInvaders-ramNoFrameskip-v4 (in one generation!)
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-23_04-44" + "-model.h5")

720 SIMPLE SpaceInvaders-ramNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-20_04-35" + "-model.h5")

1025 SIMPLE SpaceInvaders-ramNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-24_03-21" + "-model.h5")

820 (/5) SIMPLE (128, 64, 32) SpaceInvaders-ramNoFrameskip-v4 - SEEMS TO BE STABLE WITH THIS ARCHITECTURE!!!!
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-29_16-29" + "-model.h5")

800 as above (used in log 50 popultion)
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-29_17-19" + "-model.h5")

900 as above (8 (11 oops) pop)
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-29_23-03" + "-model.h5")

910 as above FOUND USING RANDOM SEARCH in 11469.35759575601s
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-29_23-01" + "-model.h5")


695 CNN SpaceInvadersNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-24_03-21" + "-model.h5")

925 CNN SpaceInvadersNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-29_10-40" + "-model.h5")


875 SIMPLE(512, 255, 128, 128) SpaceInvadersNoFrameskip-v4 stable model no ram
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-26_10-09" + "-model.h5")



Builder: It looks like the optimum number of layers is 3 when there are 127 nodes
SpaceInvadersFrameskip-v4
[3, 127, -0.07942180342399471, 0.3723077140562673, 548465349, 127, 0.5642988202272742, -0.01218497502158078, 1983418018, 127, 0.6018992809267186, 0.7353135712463283, 486090374]
reward: 860.0

SpaceInvaders-ramNoFrameskip-v4
[3, 127, -0.00802600960001551, 0.37193787474935996, 1863185804, 127, 0.6205937796994904, -0.10831157420869357, 1879081302, 127, 0.9130001984849104, 0.6419841067735436, 1780189885]
reward: 940.0 NON STABLE

2770 Frostbite-ramNoFrameskip-v4
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-03-01_20-50" + "-model.h5")

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


ENV_NAME = "Frostbite-ramNoFrameskip-v4"
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-03-01_20-50" + "-model.h5")
RANDOM_AGENT = False
MODEL_USED = "SIMPLE" # SIMPLE / CNN / BUILDER
BUILDER_MODEL_TYPE = "CNN" # SIMPLE OR CNN (only matters for BUILDER models)
GENOME = [4, 127, -0.494175080060691, -0.6090374638458818, 2143143846, 127, -0.9312494249699099, -0.6761432418135249, 1850620844, 127, -0.18520416191825528, -0.49851696479435637, 50157076, 127, 0.5798854878952928, 0.9357137092989538, 549960925]

if "-ram" in ENV_NAME:
    RAM = True
    SHOW_RAM = False
    INPUT_SHAPE = (4, 128)
else:
    RAM = False
    SHOW_RAM = False
    INPUT_SHAPE = (4, 84, 84)



if RAM:
    env = RamGymWrapper.wrap(gym.make(ENV_NAME))
else:
    env = MainGymWrapper.wrap(gym.make(ENV_NAME))

if not RANDOM_AGENT:
    if MODEL_USED == "SIMPLE":
        model = SimpleNeuralNetwork(INPUT_SHAPE, env.action_space.n, filepath=LOAD_WEIGHTS_PATH)
    elif MODEL_USED == "BUILDER":
        model = BuildNeuralNetwork(INPUT_SHAPE, env.action_space.n, GENOME, model_type=BUILDER_MODEL_TYPE)
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
    while True:
        total_reward = 0
        while not terminated:
            if RANDOM_AGENT:
                action = env.action_space.sample()
            else:
                action = model.predict(state)
            state, reward, terminated, _ = env.step(action)
            env.render()
            sleep(0.05)
            total_reward += reward
        print(total_reward)
        state = env.reset()
        terminated = False

print("Final reward: " + str(total_reward))
