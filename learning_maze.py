from lspi import policy,basis_functions, solvers, lspi

import numpy as np

NUM_BASIS = 5
DEGREE = 3
DISCOUNT = .8
EXPLORE = 0
NUM_SAMPLES = 200
LEN_SAMPLE = 500
MAX_ITERATIONS = 100000
MAX_STEPS = 500


class LearningMazeDomain():

    def __init__(self, domain, num_sample=NUM_SAMPLES, length_sample=LEN_SAMPLE):

        self.domain = domain

        self.sampling_policy = policy.Policy(basis_functions.FakeBasis(4), DISCOUNT, 1)

        self.num_samples = num_sample
        self.length_samples = length_sample
        self.samples = []
        self.lspi_samples = []
        self.walks = []

        self.random_policy_cumulative_rewards = np.sum([sample.reward for
                                                        sample in self.samples])

        self.solver = solvers.LSTDQSolver()

    def compute_samples(self, reset_samples=True):
        if reset_samples:
            self.samples = []
            self.lspi_samples = []
        for i in range(self.num_samples):
            sample, walk, terminated = self.domain.generate_samples(self.length_samples, self.sampling_policy)
            self.samples.extend(sample)
            self.walks.append(walk)
            # if terminated and len(self.lspi_samples) <= NUM_SAMPLES:
            self.lspi_samples.extend(sample)

    def learn_proto_values_basis(self, num_basis=NUM_BASIS,  walk_length=30, num_walks=10, discount=DISCOUNT,
                                 explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES,
                                 initial_policy=None, rpi_epochs=1, run_simulation=False):

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.ProtoValueBasis(
                self.domain.learn_graph(self.samples), 4, num_basis), discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
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
            initial_policy = policy.Policy(basis_functions.OneDimensionalPolynomialBasis(degree, 4), discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
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
            initial_policy = policy.Policy(basis_functions.Node2vecBasis(
                edgelist, num_actions=4, transition_probabilities=self.domain.transition_probabilities,
                dimension=dimension, walks=self.walks, walk_length=walk_length, num_walks=num_walks, window_size=window_size,
                p=p, q=q, epochs=epochs), discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
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
            initial_policy = policy.Policy(basis_functions.GraphWaveBasis(graph_edgelist, num_actions=4,
                                                                             dimension=dimension,
                                                                             time_pts_range=time_pts_range, taus=taus,
                                                                             nb_filters=nb_filters), discount, explore)
        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
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
            initial_policy = policy.Policy(basis_functions.Struc2vecBasis(graph_edgelist=edgelist, num_actions=4,
                                                                             dimension=dimension,
                                                                             walk_length=walk_length,
                                                                             num_walks=num_walks,
                                                                             window_size=window_size, epochs=epochs)
                                         , discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
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

    def learn_gcn_basis(self, graph_edgelist, dimension, walk_length=30, num_walks=10, time_pts_range=[0, 25],
                              taus='auto', max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, nb_filters=1,
                              initial_policy=None, discount=DISCOUNT, explore=EXPLORE, run_simulation=False, model_str='gcn_vae',):

        # graph = self.domain.learn_graph(sample_length=walk_length, num_samples=num_walks,
        #                                 sampling_policy=self.sampling_policy)
        #
        # self.domain.write_edgelist(graph_edgelist, graph)

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.GCNBasis(graph_edgelist, num_actions=4,
                                                                             dimension=dimension, model_str='gcn_vae',), discount, explore)
        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
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