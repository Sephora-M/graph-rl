import lspi

import numpy as np

NUM_BASIS = 5
DEGREE = 3
DISCOUNT = .8
EXPLORE = 0
NUM_SAMPLES = 3000
LEN_SAMPLE = 100
MAX_ITERATIONS = 100000
MAX_STEPS = 100


class LearningMazeDomain():

    def __init__(self, domain, num_sample=NUM_SAMPLES, length_sample=LEN_SAMPLE):

        self.domain = domain

        self.sampling_policy = lspi.Policy(lspi.basis_functions.FakeBasis(4), DISCOUNT, 1)

        self.samples = []

        for i in xrange(num_sample):
            sample = self.domain.generate_samples(length_sample, self.sampling_policy)
            self.samples.extend(sample)

        self.random_policy_cumulative_rewards = np.sum([sample.reward for
                                                        sample in self.samples])

        self.solver = lspi.solvers.LSTDQSolver()

    def learn_proto_values_basis(self, num_basis=NUM_BASIS,  walk_length=30, num_walks=10, discount=DISCOUNT,
                                 explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES,
                                 initial_policy=None, rpi_epochs=1, run_simulation=False):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.ProtoValueBasis(
                self.domain.learn_graph(walk_length, num_walks, self.sampling_policy), 4, num_basis), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_polynomial_basis(self, degree=DEGREE, discount=DISCOUNT,
                               explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES,
                               initial_policy=None, run_simulation=False):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.OneDimensionalPolynomialBasis(degree, 4), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_node2vec_basis(self, dimension=NUM_BASIS, walk_length=30, num_walks=10, window_size=10,
                             p=1, q=1, epochs=1, discount=DISCOUNT, explore=EXPLORE, max_iterations=MAX_ITERATIONS,
                             max_steps=NUM_SAMPLES, initial_policy=None, edgelist ='node2vec/graph/grid6.edgelist',
                             run_simulation=False):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.Node2vecBasis(
                edgelist, num_actions=4, transition_probabilities=self.domain.transition_probabilities,
                dimension=dimension, walk_length=walk_length, num_walks=num_walks, window_size=window_size,
                p=p, q=q, epochs=epochs), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_graphwave_basis(self, graph_edgelist, dimension, walk_length=30, num_walks=10, time_pts_range=[0, 25],
                              taus='auto', max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, nb_filters=1,
                              initial_policy=None, discount=DISCOUNT, explore=EXPLORE, run_simulation=False):

        # graph = self.domain.learn_graph(sample_length=walk_length, num_samples=num_walks,
        #                                 sampling_policy=self.sampling_policy)
        #
        # self.domain.write_edgelist(graph_edgelist, graph)

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.GraphWaveBasis(graph_edgelist, num_actions=4,
                                                                             dimension=dimension,
                                                                             time_pts_range=time_pts_range, taus=taus,
                                                                             nb_filters=nb_filters), discount, explore)
        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_struc2vec_basis(self, dimension=30, walk_length=100, num_walks=50, window_size=10, epochs=1,
                              edgelist='node2vec/graph/tworooms.edgelist', max_iterations=MAX_ITERATIONS, discount=DISCOUNT,
                               explore=EXPLORE, max_steps=NUM_SAMPLES, initial_policy=None, run_simulation=False):

        if initial_policy is None:
            initial_policy = lspi.Policy(lspi.basis_functions.Struc2vecBasis(graph_edgelist=edgelist, num_actions=4,
                                                                             dimension=dimension,
                                                                             walk_length=walk_length,
                                                                             num_walks=num_walks,
                                                                             window_size=window_size, epochs=epochs)
                                         , discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances