import numpy as np
import scipy.optimize as optimization
from learning_maze import LearningMazeDomain
import lspi
import matplotlib.pyplot as plt

dimensions = [4, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
plot_se_dims = [30, 50, 80]
wl = 100
nw = 100
K = 20


def main(folder='plots/two_room/'):
    maze, V = tworooms(True, computeV=True, num_sample=1)
    fig, ax = plt.subplots(1, 1)
    maze.domain.graph.plot_signal(maze.domain.transition_probabilities, vertex_size=60, ax=ax)
    plt.savefig(folder + 'transition_prob')
    plt.close()

    pvfs_mean_errors = []
    n2v_mean_errors = []
    gw_mean_errors = []
    s2v_mean_errors = []

    pvfs_std_errors = []
    n2v_std_errors = []
    gw_std_errors = []
    s2v_std_errors = []

    for d in dimensions:
        d_gw = d
        if d == 99:
            d_gw = 98

        pvfs_errors = []
        n2v_errors = []
        gw_errors = []
        s2v_errors = []

        for k in range(K):
            pvfs_basis = compute_ProtoValueBasis(maze, num_basis=d, walk_length=wl, num_walks=nw)
            n2v_basis = compute_node2VecBasis(maze, dimension=d_gw, walk_length=wl, num_walks=nw, window_size=10,
                                              edgelist='node2vec/graph/tworooms_all_nodes.edgelist')
            gw_basis = compute_grapheWaveBasis(maze, num_basis=d_gw, walk_length=wl, num_walks=nw,
                                               graph_edgelist='node2vec/graph/tworooms_all_nodes.edgelist')
            s2v_basis = compute_struc2VecBasis(maze, dimension=d_gw, walk_length=wl, num_walks=nw, window_size=10,
                                               edgelist='node2vec/graph/tworooms_all_nodes.edgelist')
            pvf_params, pvf_error = least_squares(pvfs_basis, V, np.random.uniform(-1.0, 1.0, size=(d,)))
            n2v_params, n2v_error = least_squares(n2v_basis, V, np.random.uniform(-1.0, 1.0, size=(d_gw,)))
            gw_params, gw_error = least_squares(gw_basis, V, np.random.uniform(-1.0, 1.0, size=(d_gw,)))
            s2v_params, s2v_error = least_squares(s2v_basis, V, np.random.uniform(-1.0, 1.0, size=(d_gw,)))

            pvfs_errors.append(pvf_error)
            n2v_errors.append(n2v_error)
            gw_errors.append(gw_error)
            s2v_errors.append(s2v_error)

        pvfs_mean_errors.append(np.mean(pvfs_errors))
        n2v_mean_errors.append(np.mean(n2v_errors))
        gw_mean_errors.append(np.mean(gw_errors))
        s2v_mean_errors.append(np.mean(s2v_errors))

        pvfs_std_errors.append(np.std(pvfs_errors))
        n2v_std_errors.append(np.std(n2v_errors))
        gw_std_errors.append(np.std(gw_errors))
        s2v_std_errors.append(np.std(s2v_errors))

        if d in plot_se_dims:
            pvf_se = np.square(V - np.matmul(pvfs_basis, pvf_params))
            n2v_se = np.square(V - np.matmul(n2v_basis, n2v_params))
            gw_se = np.square(V - np.matmul(gw_basis, gw_params))
            s2v_se = np.square(V - np.matmul(s2v_basis, s2v_params))

            fig, ax = plt.subplots(1, 1)
            maze.domain.graph.plot_signal(pvf_se, vertex_size=60, ax=ax)
            plt.savefig(folder + 'dim'+str(d)+'/pvf_SE')
            plt.close()

            fig, ax = plt.subplots(1, 1)
            maze.domain.graph.plot_signal(n2v_se, vertex_size=60, ax=ax)
            plt.savefig(folder + 'dim'+str(d)+'/n2v_SE')
            plt.close()

            fig, ax = plt.subplots(1, 1)
            maze.domain.graph.plot_signal(gw_se, vertex_size=60, ax=ax)
            plt.savefig(folder + 'dim'+str(d)+'/gw_SE')
            plt.close()

            fig, ax = plt.subplots(1, 1)
            maze.domain.graph.plot_signal(s2v_se, vertex_size=60, ax=ax)
            plt.savefig(folder + 'dim'+str(d)+'/s2v_SE')
            plt.close()

            plot_values(maze.domain.graph, pvfs_basis, pvf_params, save=True, file_name=folder + 'dim'+str(d)+'/pvf_approx_v.pdf')
            plot_values(maze.domain.graph, n2v_basis, n2v_params, save=True, file_name=folder + 'dim'+str(d)+'/n2v_approx_v.pdf')
            plot_values(maze.domain.graph, gw_basis, gw_params, save=True, file_name=folder + 'dim'+str(d)+'/gw_approx_v.pdf')
            plot_values(maze.domain.graph, s2v_basis, s2v_params, save=True, file_name=folder + 'dim'+str(d)+'/s2v_approx_v.pdf')

    return pvfs_mean_errors, n2v_mean_errors, gw_mean_errors, s2v_mean_errors, pvfs_std_errors, n2v_std_errors, gw_std_errors, s2v_std_errors


def plot_errors(folder='plots/two_rooms/'):
    # plot approx values on a grid
    # plot the MSE on the grid
    # try ploting the actions
    # try plugging in Q(sa) to each aalgo
    pvfs_mean_errors, n2v_mean_errors, gw_mean_errors, s2v_mean_errors, pvfs_std_errors, n2v_std_errors, gw_std_errors, s2v_std_errors = main(folder)
    # plt.plot(dimensions, pvfs_errors, color='green', label='pvf')
    # plt.plot(dimensions, n2v_errors, color='blue', label='n2v')
    # plt.plot(dimensions, gw_errors, color='magenta', label='gw')
    # plt.plot(dimensions, s2v_errors, color='cyan', label='v2v')

    plt.errorbar(dimensions, n2v_mean_errors, yerr=n2v_std_errors, fmt='b', ecolor='blue', label='n2v')
    plt.errorbar(dimensions, pvfs_mean_errors, yerr=pvfs_std_errors, fmt='g', ecolor='green', label='pvf')
    plt.errorbar(dimensions, s2v_mean_errors, yerr=s2v_std_errors, fmt='c', ecolor='cyan', label='s2v')
    plt.errorbar(dimensions, gw_mean_errors, yerr=gw_std_errors, fmt='m', ecolor='magenta', label='gw')
    plt.legend()
    plt.suptitle('Mean Square Error')
    plt.savefig(folder + 'all_MSE.pdf')
    plt.close()


def plot_se(maze, V, d, folder='plots/obstacles_room/'):
    fig, ax = plt.subplots(1, 1)
    maze.domain.graph.plot_signal(maze.domain.transition_probabilities, vertex_size=60, ax=ax)
    plt.savefig(folder + 'transition_prob')
    plt.close()

    pvfs_basis = compute_ProtoValueBasis(maze, num_basis=d, walk_length=wl, num_walks=nw)
    n2v_basis = compute_node2VecBasis(maze, dimension=d, walk_length=wl, num_walks=nw, window_size=10,
                                      edgelist='node2vec/graph/tworooms.edgelist')
    gw_basis = compute_grapheWaveBasis(maze, num_basis=d, walk_length=wl, num_walks=nw,
                                       graph_edgelist='node2vec/graph/tworooms.edgelist')
    s2v_basis = compute_struc2VecBasis(maze, dimension=d, walk_length=wl, num_walks=nw, window_size=10,
                                       edgelist='node2vec/graph/tworooms.edgelist')

    pvf_params, _ = least_squares(pvfs_basis, V, np.random.uniform(-1.0, 1.0, size=(d,)))
    n2v_params, _ = least_squares(n2v_basis, V, np.random.uniform(-1.0, 1.0, size=(d,)))
    gw_params, _ = least_squares(gw_basis, V, np.random.uniform(-1.0, 1.0, size=(d,)))
    s2v_params, _ = least_squares(s2v_basis, V, np.random.uniform(-1.0, 1.0, size=(d,)))

    pvf_se = np.square(V - np.matmul(pvfs_basis, pvf_params))
    n2v_se = np.square(V - np.matmul(n2v_basis, n2v_params))
    gw_se = np.square(V - np.matmul(gw_basis, gw_params))
    s2v_se = np.square(V - np.matmul(s2v_basis, s2v_params))

    fig, ax = plt.subplots(1, 1)
    maze.domain.graph.plot_signal(pvf_se, vertex_size=60, ax=ax)
    plt.savefig(folder + 'pvf_SE')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    maze.domain.graph.plot_signal(n2v_se, vertex_size=60, ax=ax)
    plt.savefig(folder + 'n2v_SE')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    maze.domain.graph.plot_signal(gw_se, vertex_size=60, ax=ax)
    plt.savefig(folder + 'gw_SE')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    maze.domain.graph.plot_signal(s2v_se, vertex_size=60, ax=ax)
    plt.savefig(folder + 's2v_SE')
    plt.close()

    plot_values(maze.domain.graph, pvfs_basis, pvf_params, save=True, file_name=folder + 'pvf_approx_v.pdf')
    plot_values(maze.domain.graph, n2v_basis, n2v_params, save=True, file_name=folder + 'n2v_approx_v.pdf')
    plot_values(maze.domain.graph, gw_basis, gw_params, save=True, file_name=folder + 'gw_approx_v.pdf')
    plot_values(maze.domain.graph, s2v_basis, s2v_params, save=True, file_name=folder + 's2v_approx_v.pdf')


def func(params, xdata, ydata):
    return ydata - np.matmul(xdata, params)


def mse(params, xdata, ydata):
    return sum(np.square(ydata - np.matmul(xdata, params))) / len(xdata)


def least_squares(basis, values, weights):
    params, _ = optimization.leastsq(func, weights, args=(basis, values))

    error = mse(params, basis, values)

    return params, error


def value_iteration(G, finish_state, obstacles, walls, obstacles_transition_probability):
    V = [0] * G.N
    R = [0] * G.N
    R[finish_state] = 100
    gamma = 0.8
    success_prob = [.9] * G.N
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


def example_grid_maze(plotV=True):
    height = 10
    width = 10
    reward_location = 9
    initial_state = None  # np.array([25])
    obstacles_location = [14, 13, 24, 23, 29, 28, 39, 38]  # range(height*width)
    walls_location = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
    obstacles_transition_probability = .2
    domain = lspi.domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=2000)
    V = value_iteration(maze.domain.graph, reward_location, obstacles_location, walls_location, obstacles_transition_probability)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('graphs/simpleMaze_trueV.pdf')
        plt.close()

    return maze, V


def low_stretch_tree_maze(plotV=True, num_sample=100, computeV=False):
    reward_location = [15]
    obstacles_location = []
    obstacles_transition_probability = .2
    domain = lspi.domains.SymmetricMazeDomain(rewards_locations=reward_location,
                                              obstacles_location=obstacles_location)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample)

    V = None
    if computeV:
        V = value_iteration(maze.domain.graph, reward_location[0], obstacles_location, [],
                            obstacles_transition_probability)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('graphs/lowStretchTree_trueV.pdf')
        plt.close()

    return maze, V


def tworooms(plotV=True, num_sample=100, computeV=False):
    height = 10
    width = 10
    reward_location = 18
    initial_state = None  # np.array([25])
    obstacles_location = []  # range(height*width)
    walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90,
                      9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      41, 42, 43, 44, 46, 47, 48, 49]
    obstacles_transition_probability = .2
    domain = lspi.domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample)

    V = None
    if computeV:
        V = value_iteration(maze.domain.graph, reward_location, obstacles_location, walls_location,
                        obstacles_transition_probability)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('graphs/twoRooms_trueV.pdf')
        plt.close()

    return maze, V


def oneroom(plotV=True, num_sample=100, computeV=False):
    height = 10
    width = 10
    reward_location = 9
    initial_state = None  # np.array([25])
    obstacles_location = []  # range(height*width)
    walls_location = []
    obstacles_transition_probability = .2
    domain = lspi.domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample)

    V = None
    if computeV:
        V = value_iteration(maze.domain.graph, reward_location, obstacles_location, walls_location,
                        obstacles_transition_probability)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('plots/one_room/trueV.pdf')
        plt.close()

    return maze, V


def obstacles_room(plotV=True, num_sample=100, computeV=False):
    height = 10
    width = 10
    reward_location = 18
    initial_state = None  # np.array([25])
    obstacles_location = [12, 13, 22, 23,
                          35, 36, 45, 46,
                          62, 63, 72, 73,
                          67, 77]
    walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90,
                      9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    obstacles_transition_probability = .2
    domain = lspi.domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample)

    V = None
    if computeV:
        V = value_iteration(maze.domain.graph, reward_location, obstacles_location, walls_location,
                            obstacles_transition_probability)

    if plotV:
        fig, ax = plt.subplots(1, 1)
        maze.domain.graph.plot_signal(np.array(V), vertex_size=60, ax=ax)
        plt.savefig('graphs/obstacleRoom_trueV.pdf')
        plt.close()

    return maze, V


def compute_ProtoValueBasis(maze, num_basis=30, walk_length=100, num_walks=50, weighted_graph=False,
                            lap_type='combinatorial'):
    if weighted_graph:
        graph = maze.domain.weighted_graph
    else:
        graph = maze.domain.learn_graph(sample_length=walk_length, num_samples=num_walks,
                                        sampling_policy=maze.sampling_policy)

    basis = lspi.basis_functions.ProtoValueBasis(graph, 4, num_basis, lap_type)

    all_basis = []

    for state in range(graph.N):
        # if maze.domain.transition_probabilities[state] == 0.:
        #     all_basis.append([0] * num_basis)
        # else:
        all_basis.append(basis.graph.U[state, 1:basis.num_laplacian_eigenvectors + 1])

    return all_basis


def compute_grapheWaveBasis(maze, num_basis=30, walk_length=100, num_walks=50,
                            graph_edgelist='learned_tworooms.edgelist', time_pts_range=[0, 100], taus='auto'):
    # graph = maze.domain.learn_graph(sample_length=walk_length, num_samples=num_walks, sampling_policy=maze.sampling_policy)
    #
    # maze.domain.write_edgelist(graph_edgelist, graph)

    basis = lspi.basis_functions.GraphWaveBasis(graph_edgelist=graph_edgelist, num_actions=4, dimension=num_basis,
                                                time_pts_range=time_pts_range, taus=taus)
    graph = maze.domain.graph
    all_basis = []
    for state in range(graph.N):
        # if maze.domain.transition_probabilities[state] == 0.:
        #     all_basis.append([0] * num_basis)
        # else:
        try:
            all_basis.append(basis.structural_emb[state])
        except IndexError:
            all_basis.append([0] * num_basis)

    return all_basis


def compute_struc2VecBasis(maze, dimension=30, walk_length=100, num_walks=50, window_size=10, p=1, q=1, epochs=1,
                           edgelist='node2vec/graph/tworooms.edgelist'):
    basis = lspi.basis_functions.Struc2vecBasis(graph_edgelist=edgelist, num_actions=4,
                                                dimension=dimension, walk_length=walk_length, num_walks=num_walks,
                                                window_size=window_size, epochs=epochs)

    all_basis = []

    for state in range(maze.domain.graph.N):
        # if maze.domain.transition_probabilities[state] == 0.:
        #     all_basis.append([0] * dimension)
        # else:
        try:
            all_basis.append(basis.model[str(state)])
        except KeyError:
            all_basis.append([0] * dimension)

    return all_basis


def compute_node2VecBasis(maze, dimension=30, walk_length=100, num_walks=50, window_size=10, p=1, q=1, epochs=1,
                          edgelist='node2vec/graph/tworooms.edgelist'):
    basis = lspi.basis_functions.Node2vecBasis(graph_edgelist=edgelist, num_actions=4,
                                               transition_probabilities=maze.domain.transition_probabilities,
                                               dimension=dimension, walk_length=walk_length, num_walks=num_walks,
                                               window_size=window_size, p=p, q=q, epochs=epochs)

    all_basis = []

    for state in range(maze.domain.graph.N):
        # if maze.domain.transition_probabilities[state] == 0.:
        #     all_basis.append([0] * dimension)
        # else:
        try:
            all_basis.append(basis.model[str(state)])
        except KeyError:
            all_basis.append([0] * dimension)

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


if __name__ == "__main__":
    plot_errors()
