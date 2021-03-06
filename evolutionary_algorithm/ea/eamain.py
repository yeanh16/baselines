import csv

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.initializers import RandomUniform
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
INITIALISER_WEIGHTS_RANGE = 5
SELECTION_RATE = 0.5
MUTATE_CHANCE = 0.1  # mutate chance per weight vector for a node in a model
MUTATION_POWER = 0.02
ELITE_RATIO = 0.25
NUMBEROFGENERATIONS = 5000
MODEL_USED = "SIMPLE"
NUM_WORKERS = 8
FRAMES_IN_OBSERVATION = 4 #if changed, need to also change this value in the gym_wrapper.py file
USE_RAM = True
FRAME_SIZE = 84
if not USE_RAM:
    INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)
else:
    INPUT_SHAPE = (FRAMES_IN_OBSERVATION, 128)
EPSILON = 0.0  # exploration/random move rate
ENV_NAME = "SpaceInvaders-ramNoFrameskip-v4"
os.makedirs(os.path.dirname(__file__) + "/models/" + ENV_NAME, exist_ok=True)
MODEL_FILEPATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + str(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + "-model.h5")
os.makedirs(os.path.dirname(__file__) + "/logs/" + ENV_NAME, exist_ok=True)
LOGPATH = str(
    os.path.dirname(__file__) + "/logs/" + ENV_NAME + "/" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
USE_LOAD_WEIGHTS = False
LOAD_WEIGHTS_PATH = str(os.path.dirname(__file__) + "/models/" + ENV_NAME + "/" + "2020-02-13_04-09" + "-model.h5")


class BaseNeuralNetwork:
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space


class ConvolutionalNeuralNetwork():
    def __init__(self, input_shape, action_space, filepath=None):
        # BaseNeuralNetwork.__init__(self, input_shape, action_space)
        # super(BaseNeuralNetwork, self).__init__()
        self.number_of_actions = action_space
        self.weight_initialiser = RandomUniform(minval=-INITIALISER_WEIGHTS_RANGE, maxval=INITIALISER_WEIGHTS_RANGE)
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
        if filepath:
            print("Loading model...")
            self.model.model.load_weights(filepath)
        # self.model.summary()

    def predict(self, state):
        # print("state: " + str(state))
        if np.random.rand() < EPSILON:
            print("random action taken")
            return random.randrange(self.number_of_actions)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        # print("output layer: " + str(q_values))
        # print("predict " + str(np.argmax(q_values[0])))
        return np.argmax(q_values[0])

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def load_weights(self, filepath):
        self.model = self.model.load_weights(filepath)

    def save_model(self):
        self.model.save_weights(MODEL_FILEPATH)
        # del self.model


class SimpleNeuralNetwork():
    def __init__(self, input_shape, action_space, filepath=None):
        # BaseNeuralNetwork.__init__(self, input_shape, action_space)
        # super(BaseNeuralNetwork, self).__init__()
        self.number_of_actions = action_space
        self.weight_initialiser = RandomUniform(minval=-INITIALISER_WEIGHTS_RANGE, maxval=INITIALISER_WEIGHTS_RANGE)
        self.model = Sequential()
        self.model.add(Dense(128,
                             activation="relu",
                             input_shape=input_shape,
                             kernel_initializer=self.weight_initialiser))
        self.model.add(Dense(128,
                             activation="relu",
                             kernel_initializer=self.weight_initialiser))
        self.model.add(Dense(128,
                             activation="relu",
                             kernel_initializer=self.weight_initialiser))
        self.model.add(Dense(128,
                             activation="relu",
                             kernel_initializer=self.weight_initialiser))
        self.model.add(Flatten())
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        if filepath:
            print("Loading model...")
            self.model.model.load_weights(filepath)
        # self.model.summary() 61,190 weights

    def predict(self, state):
        # print("state: " + str(state))
        if np.random.rand() < EPSILON:
            print("random action taken")
            return random.randrange(self.number_of_actions)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        # print("output layer: " + str(q_values))
        # print("predict " + str(np.argmax(q_values[0])))
        return np.argmax(q_values[0])

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def load_weights(self, filepath):
        self.model = self.model.load_weights(filepath)

    def save_model(self):
        self.model.save_weights(MODEL_FILEPATH)
        # del self.model


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
        self.pop = self._intialise_pop()
        self.selection_rate = selection_rate
        self.logger = Logger(LOGPATH)
        self.cumulative_frames = 0  # the total ammount of frames processed

    def _make_env(self, env_id, seed):
        def _f():
            env = MainGymWrapper.wrap(gym.make(self.env_name))
            env.seed(seed)
            return env

        return _f

    def _intialise_pop(self):
        if MODEL_USED == "SIMPLE":
            if not USE_LOAD_WEIGHTS:
                pop = []
                for i in range(0, self.pop_size):
                    pop.append(SimpleNeuralNetwork(self.input_shape, self.env.action_space.n))
                return pop
            else:
                pop = []
                model = SimpleNeuralNetwork(self.input_shape, self.env.action_space.n, LOAD_WEIGHTS_PATH)
                pop.append(model)
                # the rest of the population will be mutations of this model
                # mutation
                while len(pop) < POPULATION_SIZE:
                    new_model = copy.deepcopy(model)
                    new_model = self._mutation(new_model)
                    pop.append(new_model)
                return pop
        else:
            if not USE_LOAD_WEIGHTS:
                pop = []
                for i in range(0, self.pop_size):
                    pop.append(ConvolutionalNeuralNetwork(self.input_shape, self.env.action_space.n))
                return pop
            else:
                pop = []
                model = ConvolutionalNeuralNetwork(self.input_shape, self.env.action_space.n, LOAD_WEIGHTS_PATH)
                pop.append(model)
                # the rest of the population will be mutations of this model
                # mutation
                while len(pop) < POPULATION_SIZE:
                    new_model = copy.deepcopy(model)
                    new_model = self._mutation(new_model)
                    pop.append(new_model)
                return pop

    def train_evolutionary_algorithm(self):
        # self.envs.reset()
        # for t in range(10000):
        #     ut = np.stack([self.envs.action_space.sample() for _ in range(self.nproc)])
        #     xtp1, rt, done, info = self.envs.step(ut)
        #     self.envs.render()
        #     print(done)
        ray.init(num_gpus=1,
                 memory=3200 * 1024 * 1024,
                 object_store_memory=2000 * 1024 * 1024)
        workers = [RolloutWorker.remote() for _ in range(NUM_WORKERS)]
        best_fitness = -9999
        start_time = time.perf_counter()
        cumulative_frames = 0
        for g in range(0, NUMBEROFGENERATIONS):
            print("Generation " + str(g))
            pop_fitness = []
            gen_cumulative_fitness = 0
            min_gen_fitness = 9999
            max_gen_fitness = -9999
            # model = ConvolutionalNeuralNetwork(INPUT_SHAPE, env.action_space.n)
            count = 0

            # for model in self.pop:
            #     count += 1
            #     _, episode_reward, frames = self._fitness_test(model)
            #     cumulative_frames += frames
            #     print("Chromosome " + str(count) + " reward: " + str(episode_reward))
            #     if episode_reward < min_gen_fitness:
            #         min_gen_fitness = episode_reward
            #     if episode_reward > max_gen_fitness:
            #         max_gen_fitness = episode_reward
            #     gen_cumulative_fitness += episode_reward
            #     pop_fitness.append((model, episode_reward))

            # set up a worker for each model
            model_counter = 0
            gc.collect()
            done = False
            while model_counter < POPULATION_SIZE:
                for worker_num in range(NUM_WORKERS):
                    if (model_counter < POPULATION_SIZE):
                        model = self.pop[model_counter]
                        model_counter += 1
                        model_id = ray.put(model, weakref=True)
                        worker = workers[worker_num]
                        object_ids = worker.set_model.remote(model_id)
                    else:
                        continue
                # ray.get(object_ids) #no return here, just setting models

                # rollout each worker and store results in object_ids
                object_ids = [worker.rollout.remote() for worker in workers]
                # [obj_id], object_ids = ray.wait(object_ids)
                results = ray.get(object_ids)
                for result in results:
                    model = result[0]
                    episode_reward = result[1]
                    frames = result[2]
                    print("Reward: " + str(episode_reward))
                    cumulative_frames += frames
                    if episode_reward < min_gen_fitness:
                        min_gen_fitness = episode_reward
                    if episode_reward > max_gen_fitness:
                        max_gen_fitness = episode_reward
                    gen_cumulative_fitness += episode_reward
                    pop_fitness.append((model, episode_reward))

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
                pop_fitness[-1][0].save_model()  # save top model
            # print("Population fitness: " + str(pop_fitness))

            # add elites
            new_pop = []
            selected = []
            number_of_elites = int(POPULATION_SIZE * ELITE_RATIO)
            for i in range(POPULATION_SIZE - number_of_elites, POPULATION_SIZE):
                selected.append(pop_fitness[i])
                new_pop.append(pop_fitness[i][0])

            # selection, fill up rest of selection list
            begin_index = int(POPULATION_SIZE - (POPULATION_SIZE * SELECTION_RATE))
            for i in range(begin_index, POPULATION_SIZE - number_of_elites):
                selected.append(pop_fitness[i])

            # mutation
            while len(new_pop) < POPULATION_SIZE:
                new_model = copy.deepcopy(selected[np.random.randint(0, int(POPULATION_SIZE*SELECTION_RATE))][0])
                new_model = self._mutation(new_model)
                new_pop.append(new_model)

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

            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            # Prints out a summary of the large objects
            summary.print_(sum1)



    def _mutation(self, model=None):
        if model is None:
            print("BUG!")
            return

        # add random gaussian noise to every weight
        # print("weights before mutation: " + str(np.array(model.get_weights)))
        def mutate(w):
            result = w
            if np.random.uniform(0, 1) < MUTATE_CHANCE:
                result = [i + np.random.normal((0, MUTATION_POWER)) for i in w]
            return result

        new_weights = [w + np.random.normal(0, MUTATION_POWER) for w in model.get_weights()]
        model.set_weights(new_weights)
        # print("weights after mutation " + str(np.array(model.get_weights)))
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
        # print(str(episode_reward))
        return model, episode_reward, frames_count


#test_model = SimpleNeuralNetwork((4, 128), 6)

@ray.remote(memory = (300 * 1024 * 1024), object_store_memory=(100*1024*1024))
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
            state, reward, terminated, info = self.env.step(action)
            frames_count += 1
            # self.env.render()
            episode_reward += reward
            # sleep(0.01)
        # print(str(episode_reward))
        return self.model, episode_reward, frames_count

    def set_model(self, model):
        #del self.model
        self.model = model
        #pass


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
                             "INITIALISER_WEIGHT_RANGE", "MODEL_USED"])
            writer.writerow(
                [POPULATION_SIZE, SELECTION_RATE, MUTATE_CHANCE, MUTATION_POWER, ELITE_RATIO, INITIALISER_WEIGHTS_RANGE,
                 MODEL_USED])
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
