import csv

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.initializers import RandomUniform, RandomNormal
from evolutionary_algorithm.ea.gym_wrapper import MainGymWrapper, RamGymWrapper
from pympler.tracker import SummaryTracker
from pympler import muppy, summary, refbrowser

import ray
import numpy as np
import random
import os
import time
import datetime
import copy
import gc
import gym

from time import sleep

gc.enable()

##Make sure selection rate and elite ratio produce integers from population size
POPULATION_SIZE = 8
INITIALISER_WEIGHTS_RANGE = 0.1
SELECTION_RATE = 0.5
MUTATE_CHANCE = 0.25  # mutate chance per weight in a model
MUTATION_POWER = 0.01
ELITE_RATIO = 0.25
FITNESS_RUNS = 1  # number of runs to find best fitness score for a chromosome
NUMBEROFGENERATIONS = 5000
MODEL_USED = "CNN"  # SIMPLE or CNN
NUM_WORKERS = 8
FRAMES_IN_OBSERVATION = 4  # if changed, need to also change this value in the gym_wrapper.py file
FRAME_SIZE = 84
EPSILON = 0.0  # exploration/random move rate
ENV_NAME = "SpaceInvadersNoFrameskip-v4"
USE_RAM = False
if "-ram" in ENV_NAME:
    USE_RAM = True
if not USE_RAM:
    INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)
else:
    INPUT_SHAPE = (FRAMES_IN_OBSERVATION, 128)
os.makedirs(os.path.dirname(__file__) + "/models/" + ENV_NAME, exist_ok=True)
MODEL_FILEPATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + str(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + "-model.h5")
os.makedirs(os.path.dirname(__file__) + "/logs/" + ENV_NAME, exist_ok=True)
LOGPATH = str(
    os.path.dirname(__file__) + "/logs/" + ENV_NAME + "/" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
USE_LOAD_WEIGHTS = False
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-13_04-09" + "-model.h5")


class BuildNeuralNetwork:
    """
    Builds a neural network based on the chromosome
    chromosome composition for SIMPLE networks:
    [L, (N, RL, RH, S)]
    L = Layers used
    The following are repeated for number of layers:
    N = Number of Nodes in layer
    RL = Weights range, low
    RH = Weights range, high
    S = Seed used for weight range initialisation

    chromosome composition for CNN networks:
    L == 4
    """

    def __init__(self, input_shape, action_space, chromosome, model_type=MODEL_USED):
        self.input_shape = input_shape
        self.action_space = action_space
        self.chromosome = chromosome
        if model_type == "SIMPLE":
            self.model = self._build_model()
        elif model_type == "CNN":
            self.model = self._build_model_CNN()

    def _build_model(self):
        model = Sequential()
        chromosome_index = 1
        for l in range(self.chromosome[0] + 1):
            if chromosome_index < len(self.chromosome):
                mn = min(self.chromosome[chromosome_index + 1], self.chromosome[chromosome_index + 2])
                mx = max(self.chromosome[chromosome_index + 1], self.chromosome[chromosome_index + 2])
                kernel_initializer = RandomUniform(minval=mn, maxval=mx, seed=self.chromosome[chromosome_index + 3])
            else:
                model.add(Flatten())
                model.add(Dense(self.action_space))
                continue

            if l == 0:
                model.add(Dense(int(self.chromosome[chromosome_index]),
                                input_shape=self.input_shape,
                                kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(int(self.chromosome[chromosome_index]),
                                activation="relu",
                                kernel_initializer=kernel_initializer))
            chromosome_index += 4
        model.compile(loss="mean_squared_error",
                      optimizer=RMSprop(lr=0.00025,
                                        rho=0.95,
                                        epsilon=0.01),
                      metrics=["accuracy"])

        return model

    def _build_model_CNN(self):
        model = Sequential()
        chromosome_index = 1
        for l in range(self.chromosome[0]):
            if chromosome_index < len(self.chromosome):
                mn = min(self.chromosome[chromosome_index + 1], self.chromosome[chromosome_index + 2])
                mx = max(self.chromosome[chromosome_index + 1], self.chromosome[chromosome_index + 2])
                kernel_initializer = RandomNormal(mean=mn, stddev=mx, seed=self.chromosome[chromosome_index + 3])

            if l == 0:
                model.add(Conv2D(filters=32,
                                 kernel_size=8,
                                 strides=(4, 4),
                                 padding="valid",
                                 activation="relu",
                                 input_shape=self.input_shape,
                                 data_format="channels_first",
                                 kernel_initializer=kernel_initializer))
            elif l == 1:
                model.add(Conv2D(filters=64,
                                 kernel_size=4,
                                 strides=(2, 2),
                                 padding="valid",
                                 activation="relu",
                                 input_shape=self.input_shape,
                                 data_format="channels_first",
                                 kernel_initializer=kernel_initializer))
            elif l == 2:
                model.add(Conv2D(filters=64,
                                 kernel_size=3,
                                 strides=(1, 1),
                                 padding="valid",
                                 activation="relu",
                                 input_shape=self.input_shape,
                                 data_format="channels_first",
                                 kernel_initializer=kernel_initializer))
            elif l == 3:
                model.add(Flatten())
                model.add(Dense(512,
                                activation="relu",
                                kernel_initializer=kernel_initializer))
                model.add(Dense(self.action_space))

            chromosome_index += 4
        model.compile(loss="mean_squared_error",
                      optimizer=RMSprop(lr=0.00025,
                                        rho=0.95,
                                        epsilon=0.01),
                      metrics=["accuracy"])

        return model

    def predict(self, state):
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])

    def get_weights(self):
        return self.model.get_weights()

    def save_model(self):
        self.model.save(MODEL_FILEPATH)
        self.model.summary()

    def load_model(self):
        self.model = load_model(LOAD_WEIGHTS_PATH)

    def get_model(self):
        return self.model


class EA:
    def __init__(self, env_name, pop_size, input_shape, selection_rate):
        self.env_name = env_name
        self.pop_size = pop_size
        if not USE_RAM:
            self.env = MainGymWrapper.wrap(gym.make(self.env_name))
        else:
            self.env = RamGymWrapper.wrap(gym.make(self.env_name))
        # self.nproc = 8
        # self.envs = [self._make_env(self.env_name, seed) for seed in range(self.nproc)]
        # self.envs = SubprocVecEnv(self.envs)
        # for i in range(0, mp.cpu_count()):
        #    self.envs.append(MainGymWrapper.wrap(gym.make(self.env_name)))
        self.input_shape = input_shape
        self.pop = self._initialise_pop()
        self.selection_rate = selection_rate
        self.logger = Logger(LOGPATH)
        self.cumulative_frames = 0  # the total ammount of frames processed

    def _make_env(self, env_id, seed):
        def _f():
            env = MainGymWrapper.wrap(gym.make(self.env_name))
            env.seed(seed)
            return env

        return _f

    def _initialise_pop(self):
        pop = []
        for j in range(self.pop_size):
            chromosome = []
            if MODEL_USED == "SIMPLE":
                num_layers = np.random.randint(low=2, high=6)
            elif MODEL_USED == "CNN":
                num_layers = 4
            chromosome.append(num_layers)
            for i in range(num_layers):
                chromosome.append(np.random.randint(low=127, high=128))
                chromosome.append(np.random.uniform(low=-1, high=1))
                chromosome.append(np.random.uniform(low=-1, high=1))
                chromosome.append(np.random.randint(low=0, high=2147483647))
            pop.append(chromosome)
        return pop

    def new_seeds(self, chromosome):
        chromosome_index = 0
        while chromosome_index + 4 <= len(chromosome):
            chromosome_index += 4
            chromosome[chromosome_index] = np.random.randint(low=0, high=2147483647)
        return chromosome

    def train_evolutionary_algorithm(self):
        # ray.init(
        #          memory=4000 * 1024 * 1024,
        #          object_store_memory=2000 * 1024 * 1024)
        # workers = [RolloutWorker.remote() for _ in range(NUM_WORKERS)]

        best_fitness = -9999
        best_fitness_last_counter = 0  # counter of generations since the best fitness was updated
        start_time = time.perf_counter()
        cumulative_frames = 0

        for g in range(0, NUMBEROFGENERATIONS):
            print("Generation " + str(g))
            if g == 0:
                pop_fitness = []
            else:
                pop_fitness = pop_fitness[int(-POPULATION_SIZE * ELITE_RATIO):]
            gen_cumulative_fitness = 0
            min_gen_fitness = 9999
            max_gen_fitness = -9999
            count = 0

            for model in self.pop:
                count += 1
                if count > POPULATION_SIZE * ELITE_RATIO or g == 0:
                    chromosome, episode_reward, frames = self._fitness_test(model)
                    cumulative_frames += frames
                    print("CHROMOSOME = " + str(chromosome) + "\nreward: " + str(episode_reward))
                    if episode_reward < min_gen_fitness:
                        min_gen_fitness = episode_reward
                    if episode_reward > max_gen_fitness:
                        max_gen_fitness = episode_reward
                    gen_cumulative_fitness += episode_reward
                    pop_fitness.append((chromosome, episode_reward))
                else:
                    episode_reward = pop_fitness[count - 1][1]
                    if episode_reward < min_gen_fitness:
                        min_gen_fitness = episode_reward
                    if episode_reward > max_gen_fitness:
                        max_gen_fitness = episode_reward
                    gen_cumulative_fitness += episode_reward
                    print("CHROMOSOME = " + str(chromosome) + "\nreward: " + str(pop_fitness[count - 1][1]))

            # # set up a worker for each model
            # model_counter = 0
            # gc.collect()
            # done = False
            # while model_counter < POPULATION_SIZE:
            #     for worker_num in range(NUM_WORKERS):
            #         if (model_counter < POPULATION_SIZE):
            #             model = BuildNeuralNetwork(input_shape=INPUT_SHAPE, action_space=self.env.action_space.n ,chromosome=self.pop[model_counter])
            #             model_counter += 1
            #             model_id = ray.put(model, weakref=True)
            #             worker = workers[worker_num]
            #             object_ids = worker.set_model.remote(model_id)
            #         else:
            #             continue
            #     # ray.get(object_ids) #no return here, just setting models
            #
            #     # rollout each worker and store results in object_ids
            #     object_ids = [worker.rollout.remote() for worker in workers]
            #     # [obj_id], object_ids = ray.wait(object_ids)
            #     results = ray.get(object_ids)
            #     for result in results:
            #         model = result[0]
            #         episode_reward = result[1]
            #         frames = result[2]
            #         print("Reward: " + str(episode_reward))
            #         cumulative_frames += frames
            #         if episode_reward < min_gen_fitness:
            #             min_gen_fitness = episode_reward
            #         if episode_reward > max_gen_fitness:
            #             max_gen_fitness = episode_reward
            #         gen_cumulative_fitness += episode_reward
            #         pop_fitness.append((model, episode_reward))

            av_gen_fitness = gen_cumulative_fitness / self.pop_size
            pop_fitness.sort(key=lambda x: x[1])
            print("Generation min fitness: " + str(min_gen_fitness)
                  + " max_fitness: " + str(max_gen_fitness)
                  + " av_fitness: " + str(av_gen_fitness)
                  + " time: " + str(time.perf_counter() - start_time))
            self.logger.log_line_csv([min_gen_fitness, max_gen_fitness, av_gen_fitness, cumulative_frames,
                                      (time.perf_counter() - start_time)])
            if pop_fitness[-1][1] > best_fitness:
                print("Generation max fitness increase, saving model...")
                best_fitness = pop_fitness[-1][1]
                top_model = pop_fitness[-1][0]
                #pop_fitness[-1][0].save_model()  # save top model
                print("Top chromosome: " + str(top_model))
                best_fitness_last_counter = 0
            else:
                best_fitness_last_counter += 1
            # print("Population fitness: " + str(pop_fitness))

            # add elites
            new_pop = []
            # selected = []
            number_of_elites = int(POPULATION_SIZE * ELITE_RATIO)
            for i in range(POPULATION_SIZE - number_of_elites, POPULATION_SIZE):
                # selected.append(pop_fitness[i])
                new_pop.append(pop_fitness[i][0])

            # # selection, fill up rest of selection list
            # begin_index = int(POPULATION_SIZE - (POPULATION_SIZE * SELECTION_RATE))
            # for i in range(begin_index, POPULATION_SIZE - number_of_elites):
            #     selected.append(pop_fitness[i])

            # mutation & selection
            while len(new_pop) < POPULATION_SIZE:
                parent1 = copy.deepcopy(self._selection(pop_fitness, type="tournament"))
                # parent2 = copy.deepcopy(self._selection(pop_fitness, type="tournament"))
                # new_model = copy.deepcopy(selected[np.random.randint(0, int(POPULATION_SIZE*SELECTION_RATE))][0])
                # self._crossover(parent1, parent2)
                self._mutation(parent1)
                # self._mutation(parent2)
                new_pop.append(parent1)
                # new_pop.append(parent2)

            # object_ids = []
            # def _new_model():
            #     new_model = copy.deepcopy(selected[np.random.randint(0, int(POPULATION_SIZE * SELECTION_RATE))][0])
            #     return new_model
            #
            # object_ids = [self._mutation.remote(ray.put(_new_model())) for i in range(POPULATION_SIZE - len(new_pop))]
            # results = ray.get(object_ids)
            # new_pop.append(results)

            # next generation
            self.pop = new_pop

            # all_objects = muppy.get_objects()
            # sum1 = summary.summarize(all_objects)
            # # Prints out a summary of the large objects
            # summary.print_(sum1)

    def _mutation(self, chromosome):
        """
        mutate ranges only
        """
        chromosome_index = 1
        while chromosome_index < len(chromosome):
            # mutate number of nodes
            # if np.random.choice([True, False], p=[MUTATE_CHANCE, 1 - MUTATE_CHANCE]):
            #    chromosome[chromosome_index] = max(1, chromosome[chromosome_index] + np.round(np.random.normal(0, chromosome[chromosome_index]*MUTATION_POWER)))
            # mutate ranges
            chromosome_index += 1
            if np.random.choice([True, False], p=[MUTATE_CHANCE, 1 - MUTATE_CHANCE]):
                chromosome[chromosome_index] = chromosome[chromosome_index] + np.random.normal(0, MUTATION_POWER)
            chromosome_index += 1
            if np.random.choice([True, False], p=[MUTATE_CHANCE, 1 - MUTATE_CHANCE]):
                chromosome[chromosome_index] = chromosome[chromosome_index] + np.random.normal(0, MUTATION_POWER)
            chromosome_index += 2
        return chromosome

    def _random_weight(self, base_value=0):
        return base_value + random.uniform(-MUTATION_POWER, MUTATION_POWER)

    def _selection(self, population_fitness, type="tournament"):
        """
        takes a list of (chromosome, fitness) and returns model
        """
        if type == "tournament":
            x = population_fitness[np.random.randint(0, len(population_fitness))]
            y = population_fitness[np.random.randint(0, len(population_fitness))]
            if x[1] > y[1]:
                return x[0]
            else:
                return y[0]
        pass

    def _crossover(self, chromosome_1, chromosome_2):
        pass

    def _fitness_test(self, chromosome):
        total = 0
        frames_count = 0
        max_score = 0
        best_chromosome = chromosome
        state = self.env.reset()
        for i in range(FITNESS_RUNS):
            if i is not 0:
                chromosome = self.new_seeds(chromosome)
            model = BuildNeuralNetwork(self.input_shape, self.env.action_space.n, chromosome)
            terminated = False
            episode_reward = 0
            while not terminated:
                action = model.predict(state)
                state, reward, terminated, _ = self.env.step(action)
                frames_count += 1
                # self.env.render()
                episode_reward += reward
                # sleep(0.01)
            # print(str(episode_reward))
            if episode_reward > max_score:
                max_score = episode_reward
                best_chromosome = chromosome
            state = self.env.reset()

        return best_chromosome, max_score, frames_count


# test_model = SimpleNeuralNetwork((4, 128), 6)

@ray.remote(memory=(500 * 1024 * 1024), object_store_memory=(100 * 1024 * 1024))
class RolloutWorker(object):
    def __init__(self):
        # Tell numpy to only use one core. If we don't do this, each actor may
        # try to use all of the cores and the resulting contention may result
        # in no speedup over the serial version. Note that if numpy is using
        # OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
        # probably need to do it from the command line (so it happens before
        # numpy is imported).
        os.environ["MKL_NUM_THREADS"] = "1"
        self.model = None
        if not USE_RAM:
            self.env = MainGymWrapper.wrap(gym.make(ENV_NAME))
        else:
            self.env = RamGymWrapper.wrap(gym.make(ENV_NAME))

    def rollout(self):
        """Evaluates  env and model until the env returns "Terminated".

        Returns:
            model: the model used
            episode_reward: A list of observations
            frames_count: number of frames used to train

        """
        if self.model is None:
            print("Model is not set for worker!")
            return EOFError
        state = self.env.reset()
        terminated = False
        episode_reward = 0
        frames_count = 0
        while not terminated:
            action = self.model.predict(state)
            state, reward, terminated, _ = self.env.step(action)
            frames_count += 1
            # self.env.render()
            episode_reward += reward
            # sleep(0.01)
        # print(str(episode_reward))
        return self.model, episode_reward, frames_count

    def set_model(self, model):
        # del self.model
        self.model = model
        # pass


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
            writer.writerow(["POPULATION_SIZE", "SELECTION_RATE", "MUTATION_CHANCE", "MUTATION_POWER", "ELITE_RATIO",
                             "INITIALISER_WEIGHT_RANGE", "MODEL_USED", "FITNESS_RUNS"])
            writer.writerow(
                [POPULATION_SIZE, SELECTION_RATE, MUTATE_CHANCE, MUTATION_POWER, ELITE_RATIO, INITIALISER_WEIGHTS_RANGE,
                 MODEL_USED, FITNESS_RUNS, "EABUILDER"])
            writer.writerow(["min", "max", "av", "frames", "time"])
        ##/write header

    def log_line_csv(self, line):
        if not os.path.exists(self.logpath):
            with open(self.logpath, "w"):
                pass
        scores_file = open(self.logpath, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(line)


def __main__():
    # pass
    if __name__ == "__main__":
        run = EA(ENV_NAME, POPULATION_SIZE, INPUT_SHAPE, SELECTION_RATE)
        run.train_evolutionary_algorithm()


__main__()

# chromosome = [2, 2, 0, 1, 1, 2, 0, 1, 1]
# action_space = 6
# input_shape = (4, 4)

# a = BuildNeuralNetwork(input_shape, action_space, chromosome)

# test_model = Sequential()
# test_model.add(Dense(2, activation='relu', use_bias=False, kernel_initializer='ones', input_shape=(4,4,3,)))
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
# np.ndarray.tolist(test_weights)
#
#
# def mutate(w):
#     if np.random.uniform(0, 1) < 0.5:
#         return w + 2
#     return w
#
# weights = [mutate(w) for w in test_model.get_weights()]
# #print(weights)
#
# print("test weights before: \n" + str(test_weights))
#
# for idx, value in np.ndenumerate(test_weights):
#     print(idx, value)
#
# test_model.set_weights(test_weights)
#
# print("test weights: \n" + str(test_model.get_weights()))
