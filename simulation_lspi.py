import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from optimise import tworooms, obstacles_room, low_stretch_tree_maze, oneroom, threerooms

GRID_SIZES = [100]
grid_size = 10
WINDOW_SIZES = [10]
window_size = 10
discount = 0.8
MAX_STEPS = 100
K = 2
p = 1
Q = [4]
wl = 100
NUM_WALKS = [100]
LSPI_EPOCHS=[1,2,3,4,5,6,7,8,9,10]
nw = 256

dimension = 30
def run_experiment(environment_name, environment, edge_lisit, num_states, reward_location, walls_location, DIMENSION,
                   gcn=False, n2v=True, pvf=False, gw=False, s2v=False):
    print('>>>>>>>>>>>>>>> ENVIRONMENT: ' + environment_name)

    for q in Q:
        n2v_mean_steps_to_goal = []
        pvf_mean_steps_to_goal = []
        s2v_mean_steps_to_goal = []
        gw_mean_steps_to_goal = []
        gcn_mean_steps_to_goal = []

        n2v_mean_cumul_reward = []
        pvf_mean_cumul_reward = []
        s2v_mean_cumul_reward = []
        gw_mean_cumul_reward = []
        gcn_mean_cumul_reward = []

        n2v_std_steps_to_goal = []
        pvf_std_steps_to_goal = []
        s2v_std_steps_to_goal = []
        gw_std_steps_to_goal = []
        gcn_std_steps_to_goal = []

        n2v_std_cumul_reward = []
        pvf_std_cumul_reward = []
        s2v_std_cumul_reward = []
        gw_std_cumul_reward = []
        gcn_std_cumul_reward = []

        for lspi_epochs in LSPI_EPOCHS:
            maze, _ = environment(False, nw, wl, discount)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>> dimension basis function : ' + str(dimension))
            print('>>>>>>>>>>>>>>>>>>>>>>>>>> number of random walks of lenth '+ str(wl) +' : ' + str(nw))
            print('>>>>>>>>>>>>>>>>>>>>>>>>>> number of epochs of lspi '+ str(lspi_epochs))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>> discount factor : ' + str(discount))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>> q : ' + str(q))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>> p : ' + str(p))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>> number of walks : ' + str(nw))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>> number of samples : ' + str(num_samples))

            n2v_all_results = {}
            pvf_all_results = {}
            gw_all_results = {}
            gcn_all_results = {}
            s2v_all_results = {}

            d_n2v_mean_steps_to_goal = []
            d_n2v_mean_cumul_reward = []

            d_s2v_mean_steps_to_goal = []
            d_s2v_mean_cumul_reward = []

            d_pvf_mean_steps_to_goal = []
            d_pvf_mean_cumul_reward = []

            d_gw_mean_steps_to_goal = []
            d_gw_mean_cumul_reward = []

            d_gcn_mean_steps_to_goal = []
            d_gcn_mean_cumul_reward = []

            for k in range(K):
                maze.compute_samples(reset_policy=True)

                if gcn:
                    gcn_num_steps, gcn_learned_policy, gcn_samples, gcn_distances = maze.learn_gcn_basis(
                        graph_edgelist=edge_lisit, dimension=dimension,
                        walk_length=wl)

                    gcn_all_steps_to_goal, gcn_all_samples, gcn_all_cumulative_rewards, gcn_all_mean_steps_to_goal, gcn_all_mean_cumulative_rewards = simulate(
                        num_states, reward_location, walls_location, maze, gcn_learned_policy, max_steps=MAX_STEPS)
                    gcn_all_results[k] = {'steps_to_goal': gcn_all_steps_to_goal, 'samples': gcn_all_samples,
                                         'cumul_rewards': gcn_all_cumulative_rewards,
                                         'learning_distances': gcn_distances}

                if pvf:
                    pvf_num_steps, pvf_learned_policy, pvf_samples, pvf_distances = maze.learn_proto_values_basis(
                        num_basis=dimension, walk_length=wl, num_walks=nw, explore=0)

                    pvf_all_steps_to_goal, pvf_all_samples, pvf_all_cumulative_rewards, pvf_all_mean_steps_to_goal, pvf_all_mean_cumulative_rewards = simulate(
                        num_states, reward_location, walls_location, maze, pvf_learned_policy, max_steps=MAX_STEPS)
                    pvf_all_results[k] = {'steps_to_goal': pvf_all_steps_to_goal, 'samples': pvf_all_samples,
                                          'cumul_rewards': pvf_all_cumulative_rewards,
                                          'learning_distances': pvf_distances}

                if n2v:
                    n2v_num_steps, n2v_learned_policy, n2v_samples, n2v_distances = maze.learn_node2vec_basis(
                        dimension=dimension, walk_length=wl,
                        num_walks=nw, window_size=window_size, p=p, q=q,
                        epochs=1, explore=1, edgelist=edge_lisit,lspi_epochs=lspi_epochs)
                    n2v_all_steps_to_goal, n2v_all_samples, n2v_all_cumulative_rewards, n2v_all_mean_steps_to_goal, n2v_all_mean_cumulative_rewards = simulate(
                        num_states, reward_location, walls_location, maze, n2v_learned_policy, max_steps=MAX_STEPS)

                    n2v_all_results[k] = {'steps_to_goal': n2v_all_steps_to_goal, 'samples': n2v_all_samples,
                                          'cumul_rewards': n2v_all_cumulative_rewards,
                                          'learning_distances': n2v_distances}

                if gw:
                    gw_num_steps, gw_learned_policy, gw_samples, gw_distances = maze.learn_graphwave_basis(
                        graph_edgelist=edge_lisit, dimension=dimension, walk_length=wl,
                        num_walks=nw, explore=0, discount=discount, time_pts_range=[0, 100], taus='auto',
                        nb_filters=1)

                    gw_all_steps_to_goal, gw_all_samples, gw_all_cumulative_rewards, gw_all_mean_steps_to_goal, gw_all_mean_cumulative_rewards = simulate(
                        num_states, reward_location, walls_location, maze, gw_learned_policy, max_steps=MAX_STEPS)
                    gw_all_results[k] = {'steps_to_goal': gw_all_steps_to_goal, 'samples': gw_all_samples,
                                          'cumul_rewards': gw_all_cumulative_rewards,
                                          'learning_distances': gw_distances}

                if s2v:
                    s2v_num_steps, s2v_learned_policy, s2v_samples, s2v_distances = maze.learn_struc2vec_basis(
                        dimension=dimension, walk_length=wl, num_walks=nw, window_size=window_size, epochs=1,
                        explore=0, discount=discount,
                        edgelist=edge_lisit)

                    s2v_all_steps_to_goal, s2v_all_samples, s2v_all_cumulative_rewards, s2v_all_mean_steps_to_goal, s2v_all_mean_cumulative_rewards = simulate(
                        num_states, reward_location, walls_location, maze, s2v_learned_policy, max_steps=MAX_STEPS)

                    s2v_all_results[k] = {'steps_to_goal': s2v_all_steps_to_goal, 'samples': s2v_all_samples,
                                          'cumul_rewards': s2v_all_cumulative_rewards,
                                          'learning_distances': s2v_distances}

                if n2v:
                    d_n2v_mean_cumul_reward.append(n2v_all_mean_cumulative_rewards)
                    print("n2v mean cumul reward " + str(n2v_all_mean_cumulative_rewards))
                    d_n2v_mean_steps_to_goal.append(n2v_all_mean_steps_to_goal)
                    print("n2v mean steps to goal " + str(n2v_all_mean_steps_to_goal))
                if pvf:
                    d_pvf_mean_cumul_reward.append(pvf_all_mean_cumulative_rewards)
                    d_pvf_mean_steps_to_goal.append(pvf_all_mean_steps_to_goal)
                    print("pvf mean cumul reward " + str(pvf_all_mean_cumulative_rewards))
                    print("pvf mean steps to goal " + str(pvf_all_mean_steps_to_goal))
                if s2v:
                    d_s2v_mean_cumul_reward.append(s2v_all_mean_cumulative_rewards)
                    d_s2v_mean_steps_to_goal.append(s2v_all_mean_steps_to_goal)
                if gw:
                    d_gw_mean_cumul_reward.append(gw_all_mean_cumulative_rewards)
                    d_gw_mean_steps_to_goal.append(gw_all_mean_steps_to_goal)
                if gcn:
                    d_gcn_mean_cumul_reward.append(gcn_all_mean_cumulative_rewards)
                    d_gcn_mean_steps_to_goal.append(gcn_all_mean_steps_to_goal)

            if n2v:
                n2v_mean_steps_to_goal.append(np.mean(d_n2v_mean_steps_to_goal))
                n2v_mean_cumul_reward.append(np.mean(d_n2v_mean_cumul_reward))
                n2v_std_steps_to_goal.append(np.std(d_n2v_mean_steps_to_goal))
                n2v_std_cumul_reward.append(np.std(d_n2v_mean_cumul_reward))

            if pvf:
                pvf_mean_steps_to_goal.append(np.mean(d_pvf_mean_steps_to_goal))
                pvf_mean_cumul_reward.append(np.mean(d_pvf_mean_cumul_reward))
                pvf_std_steps_to_goal.append(np.std(d_pvf_mean_steps_to_goal))
                pvf_std_cumul_reward.append(np.std(d_pvf_mean_cumul_reward))

            if s2v:
                s2v_mean_steps_to_goal.append(np.mean(d_s2v_mean_steps_to_goal))
                s2v_mean_cumul_reward.append(np.mean(d_s2v_mean_cumul_reward))
                s2v_std_steps_to_goal.append(np.std(d_s2v_mean_steps_to_goal))
                s2v_std_cumul_reward.append(np.std(d_s2v_mean_cumul_reward))

            if gw:
                gw_mean_steps_to_goal.append(np.mean(d_gw_mean_steps_to_goal))
                gw_mean_cumul_reward.append(np.mean(d_gw_mean_cumul_reward))
                gw_std_steps_to_goal.append(np.std(d_gw_mean_steps_to_goal))
                gw_std_cumul_reward.append(np.std(d_gw_mean_cumul_reward))

            if gcn:
                gcn_mean_steps_to_goal.append(np.mean(d_gcn_mean_steps_to_goal))
                gcn_mean_cumul_reward.append(np.mean(d_gcn_mean_cumul_reward))
                gcn_std_steps_to_goal.append(np.std(d_gcn_mean_steps_to_goal))
                gcn_std_cumul_reward.append(np.std(d_gcn_mean_cumul_reward))


            # plot_results(n2v_all_results, pvf_all_results, grid_size, reward_location, walls_location, dimension, discount, nw)

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = axs[0]
        if n2v:
            ax.errorbar(LSPI_EPOCHS, n2v_mean_steps_to_goal, yerr=n2v_std_steps_to_goal, fmt='b', ecolor='blue', label='n2v')
        if pvf:
            ax.errorbar(LSPI_EPOCHS, pvf_mean_steps_to_goal, yerr=pvf_std_steps_to_goal, fmt='g', ecolor='green', label='pvf')
        if s2v:
            ax.errorbar(LSPI_EPOCHS, s2v_mean_steps_to_goal, yerr=s2v_std_steps_to_goal, fmt='c', ecolor='cyan', label='s2v')
        if gcn:
            ax.errorbar(LSPI_EPOCHS, gcn_mean_steps_to_goal, yerr=gcn_std_steps_to_goal, fmt='m', ecolor='magenta', label='gcn')
        if gw:
            ax.errorbar(LSPI_EPOCHS, gw_mean_steps_to_goal, yerr=gw_std_steps_to_goal, fmt='k', ecolor='black', label='gw')

        ax.legend()
        ax.set_title('average number of steps')

        ax = axs[1]
        if n2v:
            ax.errorbar(LSPI_EPOCHS, n2v_mean_cumul_reward, yerr=n2v_std_cumul_reward, fmt='b', ecolor='blue', label='n2v')
        if pvf:
            ax.errorbar(LSPI_EPOCHS, pvf_mean_cumul_reward, yerr=pvf_std_cumul_reward, fmt='g', ecolor='green', label='pvf')
        if s2v:
            ax.errorbar(LSPI_EPOCHS, s2v_mean_cumul_reward, yerr=s2v_std_cumul_reward, fmt='c', ecolor='cyan', label='s2v')
        if gcn:
            ax.errorbar(LSPI_EPOCHS, gcn_mean_cumul_reward, yerr=gcn_std_cumul_reward, fmt='m', ecolor='magenta', label='gcn')
        if gw:
            ax.errorbar(LSPI_EPOCHS, gw_mean_cumul_reward, yerr=gw_std_cumul_reward, fmt='k', ecolor='black', label='gw')

        ax.set_title('average cumulative reward')
        ax.legend()
        figure_name = 'plots/' + environment_name + '/n2v_' + str(dimension) + 'dimension_'+ str(discount) + 'discount_'\
                      + str(nw) + 'walks_' + str(wl) + 'walk_length_penalty_lspirep_0.9decay.pdf'
        plt.savefig(figure_name)
        print("Saved figure %s " % figure_name)

        # UNCOMMENT the lines below to write the results in pickle files
        # print('Writing pickles files...')
        # if n2v:
        #     n2v_pickle = open(
        #         'pickles/n2v_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(num_samples) + 'samples', 'wb')
        #     pickle.dump({'mean': n2v_mean_steps_to_goal, 'std': n2v_std_steps_to_goal}, n2v_pickle)
        #     n2v_pickle.close()
        # if pvf:
        #     pvf_pickle = open(
        #         'pickles/pvf_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(num_samples) + 'samples', 'wb')
        #     pickle.dump({'mean': pvf_mean_steps_to_goal, 'std': pvf_std_steps_to_goal}, pvf_pickle)
        #     pvf_pickle.close()
        # if s2v:
        #     s2v_pickle = open(
        #         'pickles/s2v_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(num_samples) + 'samples', 'wb')
        #     pickle.dump({'mean': s2v_mean_steps_to_goal, 'std': s2v_std_steps_to_goal}, s2v_pickle)
        #     s2v_pickle.close()
        # if gcn:
        #     gcn_pickle = open(
        #         'pickles/gcn_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(num_samples) + 'samples', 'wb')
        #     pickle.dump({'mean': gcn_mean_steps_to_goal, 'std': gcn_std_steps_to_goal}, gcn_pickle)
        #     gcn_pickle.close()
        # if gw:
        #     gw_pickle = open(
        #         'pickles/gw_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(num_samples) + 'samples', 'wb')
        #
        #     pickle.dump({'mean': gw_mean_steps_to_goal, 'std': gw_std_steps_to_goal}, gw_pickle)
        #     gw_pickle.close()


def example():
    grid_size = 10
    dimension = 5
    discount = .8
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> Simulation grid of size : ' + str(grid_size) + 'x' + str(grid_size))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> dimension basis function : ' + str(dimension))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> discount factor : ' + str(discount))
    height = width = grid_size
    num_states = grid_size * grid_size
    reward_location = 18
    obstacles_location = []
    walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90,
                      9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      41, 42, 43, 44, 46, 47, 48, 49]
    maze = tworooms()
    # maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location,num_sample=num_samples)

    n2v_all_results = {}
    pvf_all_results = {}

    for k in xrange(10):
        n2v_num_steps, n2v_learned_policy, n2v_samples, n2v_distances = maze.learn_node2vec_basis(dimension=dimension,
                                                                                                  walk_length=30,
                                                                                                  num_walks=10,
                                                                                                  window_size=10, p=1,
                                                                                                  q=1,
                                                                                                  epochs=3, explore=0,
                                                                                                  discount=discount,
                                                                                                  max_steps=100,
                                                                                                  max_iterations=200,
                                                                                                  edgelist='node2vec/graph/tworooms.edgelist')
        n2v_all_steps_to_goal, n2v_all_samples, n2v_all_cumulative_rewards, n2v_all_mean_steps_to_goal, n2v_all_mean_cumulative_rewards = simulate(
            num_states, reward_location,
            walls_location, maze, n2v_learned_policy)

        n2v_all_results[k] = {'steps_to_goal': n2v_all_steps_to_goal, 'samples': n2v_all_samples,
                              'cumul_rewards': n2v_all_cumulative_rewards, 'learning_distances': n2v_distances}

        pvf_num_steps, pvf_learned_policy, pvf_samples, pvf_distances = maze.learn_proto_values_basis(
            num_basis=dimension, explore=0, discount=discount, max_steps=100, max_iterations=200)

        pvf_all_steps_to_goal, pvf_all_samples, pvf_all_cumulative_rewards, pvf_all_mean_steps_to_goal, pvf_all_mean_cumulative_rewards = simulate(
            num_states, reward_location,
            walls_location, maze, pvf_learned_policy)
        pvf_all_results[k] = {'steps_to_goal': pvf_all_steps_to_goal, 'samples': pvf_all_samples,
                              'cumul_rewards': pvf_all_cumulative_rewards, 'learning_distances': pvf_distances}

    return n2v_all_results, pvf_all_results, grid_size, reward_location, walls_location, dimension, discount


def simulate(num_states, reward_location, walls_location, maze, learned_policy, max_steps=100):
    print(learned_policy.explore)
    learned_policy.explore = 0.
    all_steps_to_goal = {}
    all_samples = {}
    all_cumulative_rewards = {}
    mean_steps_to_goal = 0.
    mean_cumulative_rewards = 0.
    num_starting_states = 0
    for state in range(num_states):
        if state not in reward_location and state not in walls_location:
            num_starting_states += 1
            steps_to_goal = 0
            maze.domain.reset(np.array([state]))
            absorb = False
            samples = []
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(maze.domain.current_state())
                sample = maze.domain.apply_action(action)
                absorb = sample.absorb
                steps_to_goal += 1
                samples.append(sample)
            all_steps_to_goal[state] = steps_to_goal
            all_samples[state] = samples
            all_cumulative_rewards[state] = np.sum([s.reward for s in samples])
            mean_cumulative_rewards += all_cumulative_rewards[state]
            mean_steps_to_goal += steps_to_goal

    mean_cumulative_rewards /= num_starting_states
    mean_steps_to_goal /= num_starting_states

    return all_steps_to_goal, all_samples, all_cumulative_rewards, mean_steps_to_goal, mean_cumulative_rewards


def plot_results(n2v_all_results, pvf_all_results, grid_size, reward_location, walls_location, dimension, discount,
                 num_samples):
    n2v_mean_cumulative_rewards = []
    n2v_std_cumulative_rewards = []
    pvf_mean_cumulative_rewards = []
    pvf_std_cumulative_rewards = []

    n2v_mean_steps_to_goal = []
    n2v_std_steps_to_goal = []
    pvf_mean_steps_to_goal = []
    pvf_std_steps_to_goal = []

    non_walls = []

    for init_state in range(grid_size * grid_size):
        if init_state != reward_location and init_state not in walls_location:
            non_walls.append(init_state)
            n2v_cumulative_rewards = []
            pvf_cumulative_rewards = []
            n2v_steps_to_goal = []
            pvf_steps_to_goal = []
            for k in range(10):
                n2v_cumulative_rewards.append(n2v_all_results[k]['cumul_rewards'][init_state])
                pvf_cumulative_rewards.append(pvf_all_results[k]['cumul_rewards'][init_state])
                n2v_steps_to_goal.append(n2v_all_results[k]['steps_to_goal'][init_state])
                pvf_steps_to_goal.append(pvf_all_results[k]['steps_to_goal'][init_state])
            pvf_mean_cumulative_rewards.append(np.mean(pvf_cumulative_rewards))
            pvf_std_cumulative_rewards.append(np.std(pvf_cumulative_rewards))
            n2v_mean_cumulative_rewards.append(np.mean(n2v_cumulative_rewards))
            n2v_std_cumulative_rewards.append(np.std(n2v_cumulative_rewards))
            pvf_mean_steps_to_goal.append(np.mean(pvf_steps_to_goal))
            pvf_std_steps_to_goal.append(np.std(pvf_steps_to_goal))
            n2v_mean_steps_to_goal.append(np.mean(n2v_steps_to_goal))
            n2v_std_steps_to_goal.append(np.std(n2v_steps_to_goal))

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0, 0]
    ax.errorbar(non_walls, n2v_mean_steps_to_goal, yerr=n2v_std_steps_to_goal, fmt='bo', ecolor='blue')
    ax.set_title('n2v: number of steps')

    # With 4 subplots, reduce the number of axis ticks to avoid crowding.

    ax = axs[0, 1]
    ax.errorbar(non_walls, pvf_mean_steps_to_goal, yerr=pvf_std_steps_to_goal, fmt='ro', ecolor='red')
    ax.set_title('pvf: number of steps')

    ax = axs[1, 0]
    ax.errorbar(non_walls, n2v_mean_cumulative_rewards,
                yerr=n2v_std_cumulative_rewards, fmt='bo', ecolor='blue')
    ax.set_title('n2v: cumulative reward')

    ax = axs[1, 1]
    ax.errorbar(non_walls, pvf_mean_cumulative_rewards,
                yerr=pvf_std_cumulative_rewards, fmt='ro', ecolor='red')
    ax.set_title('pvf: cumulative reward')
    fig.suptitle('Grid size = ' + str(grid_size) + ', Dimension = ' + str(dimension) + ', Discount =' + str(discount))
    plt.savefig('plots/' + str(grid_size) + 'grid_' + str(dimension) + 'dimension_' + str(discount) + 'discount_' + str(
        num_samples) + 'samples.pdf')

def main():
    if len(sys.argv) < 2:
        raise ValueError('Please enter the environment')

    env = sys.argv[1]
    DIMENSION = [30] #[10, 20, 30, 40, 50, 60, 70]

    if env == 'tworooms':
        environment = tworooms
        edgelist = 'node2vec/graph/tworooms_withwalls.edgelist'
        num_states = 100
        reward_location = [18]
        walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 20, 30, 40, 50, 60, 70, 80, 90,
                          9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                          41, 42, 43, 44, 46, 47, 48, 49]

    elif env == 'obstacles_room':
        environment = obstacles_room
        edgelist = 'node2vec/graph/oneroom_all_nodes.edgelist'
        reward_location = [18]
        num_states = 100
        walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 20, 30, 40, 50, 60, 70, 80, 90,
                          9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    elif env == 'low_stretch_tree_maze':
        environment = low_stretch_tree_maze
        edgelist = 'node2vec/graph/lowStretchTree.edgelist'
        reward_location = [15]
        num_states = 64
        walls_location = []
        DIMENSION = [3, 5, 10, 20, 30, 40, 50, 60, 63]
    elif env == 'threerooms':
        num_states = 5000
        environment = threerooms
        edgelist = 'node2vec/graph/threerooms.edgelist'
        reward_location = [198]
        walls_location = []
        walls_location.extend(range(100))
        walls_location.extend(range(4900, 5000))
        walls_location.extend(range(0, 5000, 100))
        walls_location.extend(range(99, 5000, 100))
        walls_location.extend(range(1600, 1670))
        walls_location.extend(range(1680, 1700))
        walls_location.extend(range(3200, 3220))
        walls_location.extend(range(3230, 3300))
        DIMENSION = [100]
    elif env == 'oneroom':
        environment = oneroom
        edgelist = 'node2vec/graph/grid10.edgelist'
        reward_location = [9]
        walls_location = []
        num_states = 100
    else:
        raise ValueError('Environment not recognized')

    run_experiment(env, environment, edgelist, num_states, reward_location, walls_location, [10, 50, 100, 200, 300, 500, 100])


if __name__ == "__main__":
    main()
