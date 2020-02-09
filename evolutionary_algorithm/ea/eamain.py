from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from baselines.GA.gym_wrapper import MainGymWrapper

import gym

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)

class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,
                              kernel_size=8,
                              strides=(4, 4),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(filters=64,
                              kernel_size=4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(filters=64,
                              kernel_size=3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()


def __main__():
    env = MainGymWrapper.wrap(gym.make("PongNoFrameskip-v4"))
    env.reset()
    terminated = False
    while not terminated:
        action = model.predict()
        obs, reward, terminated, info = env.step(action)
        env.render()



__main__()