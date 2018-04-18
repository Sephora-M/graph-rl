import numpy as np
import scipy.optimize as optimization
from learning_maze import LearningMazeDomain
import lspi
import matplotlib.pyplot as plt


def f(basis, weights):
    return np.dot(basis, weights)


def func(params, xdata, ydata):
    return ydata - np.matmul(xdata, params)


def mse(params, xdata, ydata):
    return sum(np.square(ydata - np.matmul(xdata, params))) / len(xdata)


def least_squares(basis, values, weights):
    params, _ = optimization.leastsq(func, weights, args=(basis, values))

    error = mse(params, basis, values)

    return params, error


def example_grid_maze(plotV=True):
    height = 10
    width = 10
    reward_location = 9
    initial_state = None  # np.array([25])
    obstacles_location = [14, 13, 24, 23, 29, 28, 39, 38]  # range(height*width)
    walls_location = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
    obstacles_transition_probability = .2
    maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location, initial_state, obstacles_transition_probability,
                              num_sample=2000)

    def value_iteration(G, finish_state, obstacles, walls):
        V = [0] * G.N
        R = [0] * G.N
        R[finish_state] = 100
        gamma = 0.9
        success_prob = [1] * G.N
        for i in obstacles:
            success_prob[i] = obstacles_transition_probability
        for i in walls:
            success_prob[i] = .0
        epsilon = .0001
        diff = 100
        iterations = 0
        while diff > epsilon:
            iterations = iterations + 1
            diff = 0
            for s in xrange(G.N):
                if s == finish_state:
                    max_a = success_prob[s] * R[s]
                else:
                    max_a = float('-inf')
                    for s_prime in G.W.getcol(s).nonzero()[0]:
                        new_v = success_prob[s] * (R[s] + gamma * V[s_prime])
                        if new_v > max_a:
                            max_a = new_v
                diff = diff + abs(V[s] - max_a)
                V[s] = max_a
        print "number of iterations in Value Iteration:"
        print iterations
        return V

    V = value_iteration(maze.domain.graph, reward_location, obstacles_location, walls_location)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('graphs/simpleMaze_trueV.pdf')
        plt.close()

    return maze, V


def compute_ProtoValueBasis(maze, num_basis=30, weighted_graph=False, lap_type='combinatorial'):

    if weighted_graph:
        graph = maze.domain.weighted_graph
    else:
        graph = maze.domain.graph

    basis = lspi.basis_functions.ProtoValueBasis(graph, 4, num_basis, lap_type)

    all_basis = []

    for state in range(maze.domain.graph.N):
        all_basis.append(basis.graph.U[state, 1:basis.num_laplacian_eigenvectors + 1])

    return all_basis


def compute_node2VecBasis(maze, dimension=30, walk_length=30, num_walks=10, window_size=10, p=1, q=1, epochs=1):
    basis = lspi.basis_functions.Node2vecBasis('node2vec/graph/grid.edgelist', num_actions=4,
                                               transition_probabilities=maze.domain.transition_probabilities,
                                               dimension=dimension,walk_length=walk_length, num_walks=num_walks,
                                               window_size=window_size, p=p, q=q, epochs=epochs)

    all_basis = []

    for state in range(maze.domain.graph.N):
        all_basis.append(basis.model[str(state)])

    return all_basis


def plot_values(graph, basis, params, save=False, file_name='approx_v.pdf'):
    if graph is None:
        raise ValueError('graph cannot be None')

    if graph.N != len(basis):
        raise ValueError('graph.N and len(basis) must be equal')
    approx_values = np.matmul(basis, params)

    fig, ax = plt.subplots(1, 1)
    graph.plot_signal(approx_values, vertex_size=60, ax=ax)

    if not save:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

# dimensions=[3,5,10,15,20,25,30,35,40,45,50]
# pv_errors=[]
# n2v_errors=[]
# for dim in dimensions:
#     pvfs = opt.compute_ProtoValueBasis(maze,num_basis=dim)
#     n2v = opt.compute_node2VecBasis(maze,dimension=dim, walk_length=50, epochs=3)
#     pv_params, pv_error = opt.least_squares(pvfs, V, np.random.random(dim))
#     n2v_params, n2v_error = opt.least_squares(n2v, V, np.random.random(dim))
#     pv_errors.append(pv_error)
#     n2v_errors.append(n2v_error)
#     plot_values(maze.domain.graph, pvfs, pv_params, True, 'graphs/'+str(dim)+'pvfs.pdf')
#     plot_values(maze.domain.graph, n2v, n2v_params, True, 'graphs/'+str(dim)+'n2v.pdf')

# walk_lengths=[3,5,10,15,20,25,30,35,40,45,50]
# walk_length=30
# nums_walks=[6,7,8,9,10]
# pv_errors=[]
# n2v_errors=[]
# dim=5
# p=2
# q=0.1
# for num_walks in nums_walks:
#     n2v = opt.compute_node2VecBasis(maze,dimension=dim, walk_length=walk_length,num_walks=num_walks, p=p, q=q, epochs=3)
#     n2v_params, n2v_error = opt.least_squares(n2v, V, np.random.random(dim))
#     n2v_errors.append(n2v_error)
#     opt.plot_values(maze.domain.graph, n2v, n2v_params, True, 'graphs/numwalks'+str(num_walks)+'n2v.pdf')

# pv_errors=[]
# n2v_errors=[]
# dim=5
# for i in xrange(20):
#     pvfs = opt.compute_ProtoValueBasis(maze,num_basis=dim,weighted_graph=True, lap_type='normalized')
#     n2v = opt.compute_node2VecBasis(maze,dimension=dim, walk_length=30, epochs=3)
#     pv_params, pv_error = opt.least_squares(pvfs, V, np.random.random(dim))
#     n2v_params, n2v_error = opt.least_squares(n2v, V, np.random.random(dim))
#     pv_errors.append(pv_error)
#     n2v_errors.append(n2v_error)
#     opt.plot_values(maze.domain.graph, n2v, n2v_params, True, 'graphs/' + str(i) + 'n2v.pdf')