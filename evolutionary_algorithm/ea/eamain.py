import csv

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.initializers import RandomUniform
from evolutionary_algorithm.ea.gym_wrapper import MainGymWrapper
import numpy as np
import random
import os
import datetime
import copy

#import multiprocessing as mp

from time import sleep

import gym

#pool = mp.Pool(mp.cpu_count())
##Make sure selection rate and elite ratio produce integers from population size
POPULATION_SIZE = 100
SELECTION_RATE = 0.1
MUTATION_POWER = 0.02
ELITE_RATIO = 0.02
NUMBEROFGENERATIONS = 1000
FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)
EPSILON = 0.0 #exploration/random move rate
ENV_NAME = "SpaceInvadersNoFrameskip-v4"
os.makedirs(os.path.dirname(__file__) + "/models/" + ENV_NAME, exist_ok=True)
MODEL_FILEPATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + "-model.h5")
os.makedirs(os.path.dirname(__file__) + "/logs/" + ENV_NAME, exist_ok=True)
LOGPATH = str(os.path.dirname(__file__) + "/logs/" + ENV_NAME + "/" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))

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
        #self.model.summary()

    def predict(self, state):
        #print("state: " + str(state))
        if np.random.rand() < EPSILON:
            print("random action taken")
            return random.randrange(self.number_of_actions)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        #print("output layer: " + str(q_values))
        #print("predict " + str(np.argmax(q_values[0])))
        return np.argmax(q_values[0])

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_model(self):
        self.model.save(MODEL_FILEPATH)
        #del self.model





class EA:
    def __init__(self, env_name, pop_size, input_shape, selection_rate):
        self.env_name = env_name
        self.pop_size = pop_size
        self.env = MainGymWrapper.wrap(gym.make(self.env_name))
        #for i in range(0, mp.cpu_count()):
        #    self.envs.append(MainGymWrapper.wrap(gym.make(self.env_name)))
        self.input_shape = input_shape
        self.pop = self._intialise_pop()
        self.selection_rate = selection_rate
        self.logger = Logger(LOGPATH)
        self.cumulative_frames = 0 #the total ammount of frames processed

    def _intialise_pop(self):
        pop = []
        for i in range(0, self.pop_size):
            pop.append(ConvolutionalNeuralNetwork(self.input_shape, self.env.action_space.n))
        return pop

    def train_evolutionary_algorithm(self):
        best_fitness = -9999
        for g in range(0, NUMBEROFGENERATIONS):
            print("Generation " + str(g))
            pop_fitness = []
            cumulative_fitness = 0
            min_gen_fitness = 9999
            max_gen_fitness = -9999
            cumulative_frames = 0
            # model = ConvolutionalNeuralNetwork(INPUT_SHAPE, env.action_space.n)
            count = 0
            # fitness_result_objects = [pool.apply_async(self._fitness_test, args=(model, env)) for model in self.pop for env in self.envs]
            for model in self.pop:
                count += 1
                _, episode_reward, frames = self._fitness_test(model)
                cumulative_frames += frames
                print("Chromosome " + str(count) + " reward: " + str(episode_reward))
                if episode_reward < min_gen_fitness:
                    min_gen_fitness = episode_reward
                if episode_reward > max_gen_fitness:
                    max_gen_fitness = episode_reward
                cumulative_fitness += episode_reward
                pop_fitness.append((model, episode_reward))
            av_gen_fitness = cumulative_fitness / self.pop_size
            pop_fitness.sort(key=lambda x: x[1])
            print("Generation min fitness: " + str(min_gen_fitness)
                  + " max_fitness: " + str(max_gen_fitness)
                  + " av_fitness: " + str(av_gen_fitness))
            self.logger.log_line_csv([min_gen_fitness] + [max_gen_fitness] + [av_gen_fitness] + [cumulative_frames])
            if pop_fitness[-1][1] > best_fitness:
                print("Generation max fitness increase, saving model...")
                best_fitness = pop_fitness[-1][1]
                pop_fitness[-1][0].save_model()  # save top model
            print("Population fitness: " + str(pop_fitness))

            # add elites
            new_pop = []
            selected = []
            number_of_elites = int(POPULATION_SIZE * ELITE_RATIO)
            for i in range(POPULATION_SIZE-number_of_elites, POPULATION_SIZE):
                selected.append(pop_fitness[i])
                new_pop.append(pop_fitness[i][0])

            # selection, fill up rest of selection list
            begin_index = int(POPULATION_SIZE -(POPULATION_SIZE*SELECTION_RATE))
            for i in range(begin_index, POPULATION_SIZE-number_of_elites):
                selected.append(pop_fitness[i])

            # mutation
            while len(new_pop) < POPULATION_SIZE:
                new_model = copy.deepcopy(selected[np.random.randint(0, int(POPULATION_SIZE*SELECTION_RATE))][0])
                new_model = self._mutation(new_model)
                new_pop.append(new_model)

            #next generation
            self.pop = new_pop


    # results = [r.get()[1] for r in fitness_result_objects]
    # pool.close
    # pool.join
    # print(fitness_result_objects)

    def _mutation(self, model):
        # add random gaussian noise to every weight
        print("weights before mutation: " + str(np.array(model.get_weights)))
        new_weights = [w + np.random.normal(0, MUTATION_POWER) for w in model.get_weights()]
        model.set_weights(new_weights)
        print("weights after mutation " + str(np.array(model.get_weights)))
        return model

    def _selection(self):
        pass

    def _crossover(self):
        pass

    def _fitness_test(self, model):
        state = self.env.reset()
        terminated = False
        episode_reward = 0
        frames_count = 0
        while not terminated:
            action = model.predict(state)
            state, reward, terminated, info = self.env.step(action)
            frames_count += 1
            # env.render()
            episode_reward += reward
            # sleep(0.01)
        #print(str(episode_reward))
        return model, episode_reward, frames_count



class Logger:
    def __init__(self, logpath):
        self.logpath = logpath
        ##write header
        if not os.path.exists(self.logpath):
            with open(self.logpath, "w"):
                pass
        scores_file = open(self.logpath, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(["POPULATION_SIZE", "SELECTION_RATE", "MUTATION_POWER", "ELITE_RATIO"])
            writer.writerow([POPULATION_SIZE, SELECTION_RATE, MUTATION_POWER, ELITE_RATIO])
            writer.writerow(["min", "max", "av", "frames"])
        ##/write header

    def log_line_csv(self, line):
        if not os.path.exists(self.logpath):
            with open(self.logpath, "w"):
                pass
        scores_file = open(self.logpath, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([line])





def __main__():
    #pass
    run = EA(ENV_NAME, POPULATION_SIZE, INPUT_SHAPE, SELECTION_RATE)
    run.train_evolutionary_algorithm()

__main__()

# test_model = Sequential()
# test_model.add(Dense(2, activation='relu', use_bias=False, kernel_initializer='ones', input_shape=(1,)))
# test_model.add(Dense(2, activation='relu', use_bias=False, kernel_initializer='ones'))
# test_model.add(Dense(2, use_bias=False, kernel_initializer='ones'))
#
# test_model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False))
# test_model.summary()
#
# test_weights = np.array(test_model.get_weights())
# for x in np.nditer(test_weights, op_flags=['readwrite'], flags=['refs_ok']):
#     x[...]
#
# weights = [w + 2 for w in test_model.get_weights()]
# print(weights)
#
# print("test weights before: \n" + str(test_weights))
#
# test_model.set_weights(weights)
#
# print("test weights: \n" + str(test_weights))
