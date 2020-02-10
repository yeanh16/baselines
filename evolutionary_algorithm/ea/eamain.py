from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from keras.initializers import RandomUniform
from evolutionary_algorithm.ea.gym_wrapper import MainGymWrapper
import numpy as np
import random
from time import sleep

import gym

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)
EPSILON = 0.0 #exploration/random move rate

class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, action_space):
        self.number_of_actions = action_space
        self.weight_initialiser = RandomUniform(minval=-0.1, maxval=0.1)
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,
                              kernel_size=8,
                              strides=(4, 4),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first",
                              kernel_initializer=self.weight_initialiser))
        self.model.add(Conv2D(filters=64,
                              kernel_size=4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first",
                              kernel_initializer=self.weight_initialiser))
        self.model.add(Conv2D(filters=64,
                              kernel_size=3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first",
                              kernel_initializer=self.weight_initialiser))
        self.model.add(Flatten())
        self.model.add(Dense(512,
                             activation="relu",
                             kernel_initializer=self.weight_initialiser))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()

    def _predict(self, state):
        #print("state: " + str(state))
        if np.random.rand() < EPSILON:
            print("random action taken")
            return random.randrange(self.number_of_actions)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        #print("output layer: " + str(q_values))
        #print("predict " + str(np.argmax(q_values[0])))
        return np.argmax(q_values[0])


def __main__():
    env = MainGymWrapper.wrap(gym.make("SpaceInvadersNoFrameskip-v4"))
    pop = []
    for i in range(0, 100):
        pop.append(ConvolutionalNeuralNetwork(INPUT_SHAPE,env.action_space.n))

    pop_fitness = []
    for model in pop:
        state = env.reset()
        terminated = False
        episode_reward = 0
        #model = ConvolutionalNeuralNetwork(INPUT_SHAPE, env.action_space.n)
        while not terminated:
            action = model._predict(state)
            state, reward, terminated, info = env.step(action)
            #env.render()
            episode_reward += reward
            #sleep(0.01)
        print("Episode reward: " + str(episode_reward))
        pop_fitness.append(episode_reward)

    print(str(pop_fitness))

__main__()