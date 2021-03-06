import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from optimise import tworooms, obstacles_room, low_stretch_tree_maze, oneroom

num_samples = 200

DISCOUNT = [0.8]
GRID_SIZES = [10]
grid_size = 10
WINDOW_SIZES = [10]
window_size = 10
discount = 0.8
K = 5
p = 0.5
Q = [2]
wl = 100
nw = 200


def run_experiment(environment_name, environment, edge_lisit, num_states, reward_location, walls_location, DIMENSION,
                   gcn=True):
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

        for dimension in DIMENSION:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>> dimension basis function : ' + str(dimension))
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
                maze, _ = environment(False, num_samples)

                if gcn:
                    gcn_num_steps, gcn_learned_policy, gcn_samples, gcn_distances = maze.learn_node2vec_basis(
                        dimension=dimension, walk_length=wl,
                        num_walks=int(nw/num_states), window_size=window_size, p=1, q=1,
                        epochs=1, explore=0, discount=discount,
                        max_steps=100, max_iterations=200,
                        edgelist=edge_lisit)

                    gcn_all_steps_to_goal, gcn_all_samples, gcn_all_cumulative_rewards, gcn_all_mean_steps_to_goal, gcn_all_mean_cumulative_rewards = simulate(
                        num_states, reward_location, walls_location, maze, gcn_learned_policy)
                    gcn_all_results[k] = {'steps_to_goal': gcn_all_steps_to_goal, 'samples': gcn_all_samples,
                                         'cumul_rewards': gcn_all_cumulative_rewards,
                                         'learning_distances': gcn_distances}

                pvf_num_steps, pvf_learned_policy, pvf_samples, pvf_distances = maze.learn_node2vec_basis(
                    dimension=dimension, walk_length=wl,
                    num_walks=int(nw/num_states), window_size=window_size, p=2, q=0.5,
                    epochs=1, explore=0, discount=discount,
                    max_steps=100, max_iterations=200,
                    edgelist=edge_lisit)

                pvf_all_steps_to_goal, pvf_all_samples, pvf_all_cumulative_rewards, pvf_all_mean_steps_to_goal, pvf_all_mean_cumulative_rewards = simulate(
                    num_states, reward_location, walls_location, maze, pvf_learned_policy)
                pvf_all_results[k] = {'steps_to_goal': pvf_all_steps_to_goal, 'samples': pvf_all_samples,
                                      'cumul_rewards': pvf_all_cumulative_rewards,
                                      'learning_distances': pvf_distances}

                n2v_num_steps, n2v_learned_policy, n2v_samples, n2v_distances = maze.learn_node2vec_basis(
                    dimension=dimension, walk_length=wl,
                    num_walks=int(nw/num_states), window_size=window_size, p=1, q=4,
                    epochs=1, explore=0, discount=discount,
                    max_steps=100, max_iterations=200,
                    edgelist=edge_lisit)
                n2v_all_steps_to_goal, n2v_all_samples, n2v_all_cumulative_rewards, n2v_all_mean_steps_to_goal, n2v_all_mean_cumulative_rewards = simulate(
                    num_states, reward_location, walls_location, maze, n2v_learned_policy)

                n2v_all_results[k] = {'steps_to_goal': n2v_all_steps_to_goal, 'samples': n2v_all_samples,
                                      'cumul_rewards': n2v_all_cumulative_rewards,
                                      'learning_distances': n2v_distances}

                gw_num_steps, gw_learned_policy, gw_samples, gw_distances = maze.learn_node2vec_basis(
                    dimension=dimension, walk_length=wl,
                    num_walks=int(nw/num_states), window_size=window_size, p=0.25, q=0.25,
                    epochs=1, explore=0, discount=discount,
                    max_steps=100, max_iterations=200,
                    edgelist=edge_lisit)

                gw_all_steps_to_goal, gw_all_samples, gw_all_cumulative_rewards, gw_all_mean_steps_to_goal, gw_all_mean_cumulative_rewards = simulate(
                    num_states, reward_location, walls_location, maze, gw_learned_policy)
                gw_all_results[k] = {'steps_to_goal': gw_all_steps_to_goal, 'samples': gw_all_samples,
                                      'cumul_rewards': gw_all_cumulative_rewards,
                                      'learning_distances': gw_distances}

                s2v_num_steps, s2v_learned_policy, s2v_samples, s2v_distances = maze.learn_node2vec_basis(
                    dimension=dimension, walk_length=wl,
                    num_walks=int(nw/num_states), window_size=window_size, p=4, q=1,
                    epochs=1, explore=0, discount=discount,
                    max_steps=100, max_iterations=200,
                    edgelist=edge_lisit)

                s2v_all_steps_to_goal, s2v_all_samples, s2v_all_cumulative_rewards, s2v_all_mean_steps_to_goal, s2v_all_mean_cumulative_rewards = simulate(
                    num_states, reward_location, walls_location, maze, s2v_learned_policy)

                s2v_all_results[k] = {'steps_to_goal': s2v_all_steps_to_goal, 'samples': s2v_all_samples,
                                      'cumul_rewards': s2v_all_cumulative_rewards,
                                      'learning_distances': s2v_distances}

                d_n2v_mean_cumul_reward.append(n2v_all_mean_cumulative_rewards)
                d_n2v_mean_steps_to_goal.append(n2v_all_mean_steps_to_goal)

                d_pvf_mean_cumul_reward.append(pvf_all_mean_cumulative_rewards)
                d_pvf_mean_steps_to_goal.append(pvf_all_mean_steps_to_goal)

                d_s2v_mean_cumul_reward.append(s2v_all_mean_cumulative_rewards)
                d_s2v_mean_steps_to_goal.append(s2v_all_mean_steps_to_goal)

                d_gw_mean_cumul_reward.append(gw_all_mean_cumulative_rewards)
                d_gw_mean_steps_to_goal.append(gw_all_mean_steps_to_goal)
                if gcn:
                    d_gcn_mean_cumul_reward.append(gcn_all_mean_cumulative_rewards)
                    d_gcn_mean_steps_to_goal.append(gcn_all_mean_steps_to_goal)

            n2v_mean_steps_to_goal.append(np.mean(d_n2v_mean_steps_to_goal))
            n2v_mean_cumul_reward.append(np.mean(d_n2v_mean_cumul_reward))

            pvf_mean_steps_to_goal.append(np.mean(d_pvf_mean_steps_to_goal))
            pvf_mean_cumul_reward.append(np.mean(d_pvf_mean_cumul_reward))

            n2v_std_steps_to_goal.append(np.std(d_n2v_mean_steps_to_goal))
            n2v_std_cumul_reward.append(np.std(d_n2v_mean_cumul_reward))

            pvf_std_steps_to_goal.append(np.std(d_pvf_mean_steps_to_goal))
            pvf_std_cumul_reward.append(np.std(d_pvf_mean_cumul_reward))

            s2v_mean_steps_to_goal.append(np.mean(d_s2v_mean_steps_to_goal))
            s2v_mean_cumul_reward.append(np.mean(d_s2v_mean_cumul_reward))

            gw_mean_steps_to_goal.append(np.mean(d_gw_mean_steps_to_goal))
            gw_mean_cumul_reward.append(np.mean(d_gw_mean_cumul_reward))
            if gcn:
                gcn_mean_steps_to_goal.append(np.mean(d_gcn_mean_steps_to_goal))
                gcn_mean_cumul_reward.append(np.mean(d_gcn_mean_cumul_reward))

            s2v_std_steps_to_goal.append(np.std(d_s2v_mean_steps_to_goal))
            s2v_std_cumul_reward.append(np.std(d_s2v_mean_cumul_reward))

            gw_std_steps_to_goal.append(np.std(d_gw_mean_steps_to_goal))
            gw_std_cumul_reward.append(np.std(d_gw_mean_cumul_reward))
            if gcn:
                gcn_std_steps_to_goal.append(np.std(d_gcn_mean_steps_to_goal))
                gcn_std_cumul_reward.append(np.std(d_gcn_mean_cumul_reward))

            # plot_results(n2v_all_results, pvf_all_results, grid_size, reward_location, walls_location, dimension, discount, num_samples)

        # UNCOMMENT the lines below to right the results in pickle files
        n2v_pickle = open(
            'pickles/n2v_p1q4_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(
                num_samples) + 'samples', 'wb')
        pvf_pickle = open(
            'pickles/n2v_p2q0.5_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(
                num_samples) + 'samples', 'wb')
        s2v_pickle = open(
            'pickles/n2v_p4q1_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(
                num_samples) + 'samples', 'wb')
        if gcn:
            gcn_pickle = open(
                'pickles/n2v_p1q1_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(
                    num_samples) + 'samples', 'wb')
        gw_pickle = open(
            'pickles/n2v_p0.25q0.25_' + environment_name + '_' + str(DISCOUNT) + 'discount_' + str(
                num_samples) + 'samples', 'wb')

        print('Writing pickles files...')
        pickle.dump({'mean': n2v_mean_steps_to_goal, 'std': n2v_std_steps_to_goal}, n2v_pickle)
        pickle.dump({'mean': pvf_mean_steps_to_goal, 'std': pvf_std_steps_to_goal}, pvf_pickle)
        pickle.dump({'mean': s2v_mean_steps_to_goal, 'std': s2v_std_steps_to_goal}, s2v_pickle)
        pickle.dump({'mean': gw_mean_steps_to_goal, 'std': gw_std_steps_to_goal}, gw_pickle)
        if gcn:
            pickle.dump({'mean': gcn_mean_steps_to_goal, 'std': gcn_std_steps_to_goal}, gcn_pickle)

        n2v_pickle.close()
        pvf_pickle.close()
        s2v_pickle.close()
        if gcn:
            gcn_pickle.close()
        gw_pickle.close()

        
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
        ax = axs[0]
        ax.errorbar(DIMENSION, n2v_mean_steps_to_goal, yerr=n2v_std_steps_to_goal, fmt='b', ecolor='blue', label='p=1 q=4')
        ax.errorbar(DIMENSION, pvf_mean_steps_to_goal, yerr=pvf_std_steps_to_goal, fmt='g', ecolor='green', label='p=2 q=0.5')
        ax.errorbar(DIMENSION, s2v_mean_steps_to_goal, yerr=s2v_std_steps_to_goal, fmt='c', ecolor='cyan', label='p=4 q=1')
        if gcn:
            ax.errorbar(DIMENSION, gcn_mean_steps_to_goal, yerr=gcn_std_steps_to_goal, fmt='m', ecolor='magenta', label='p=1 q=1')
        ax.errorbar(DIMENSION, gw_mean_steps_to_goal, yerr=gw_std_steps_to_goal, fmt='k', ecolor='black', label='p=0.25 q=0.25')

        ax.legend()
        ax.set_title('average number of steps')

        # ax = axs[1]
        # ax.errorbar(DIMENSION, n2v_mean_cumul_reward, yerr=n2v_std_cumul_reward, fmt='b', ecolor='blue', label='n2v')
        # ax.errorbar(DIMENSION, pvf_mean_cumul_reward, yerr=pvf_std_cumul_reward, fmt='g', ecolor='green', label='pvf')
        # ax.errorbar(DIMENSION, s2v_mean_cumul_reward, yerr=s2v_std_cumul_reward, fmt='c', ecolor='cyan', label='s2v')
        # if gcn:
        #     ax.errorbar(DIMENSION, gcn_mean_cumul_reward, yerr=gcn_std_cumul_reward, fmt='m', ecolor='magenta', label='gcn')
        # ax.errorbar(DIMENSION, gw_mean_cumul_reward, yerr=gw_std_cumul_reward, fmt='k', ecolor='black', label='gw')
        #
        # ax.set_title('average cumulative reward')
        # ax.legend()
        figure_name = 'plots/' + environment_name + '/n2v_' + str(num_samples) + 'samples_' + str(nw) + 'num_walks' \
                      + str(wl) + 'walk_length.pdf'
        plt.savefig(figure_name)
        print("Saved figure %s " % figure_name)




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

    return n2v_all_results, pvf_all_results, grid_size, reward_location, walls_location, dimension, discount, num_samples


def simulate(num_states, reward_location, walls_location, maze, learned_policy, max_steps=100):
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


def plot_result(all_results, grid_size, reward_location, walls_location, dimension, discount, num_samples, window_size):
    mean_cumulative_rewards = []
    std_cumulative_rewards = []

    mean_steps_to_goal = []
    std_steps_to_goal = []

    non_walls = []

    for init_state in range(grid_size * grid_size):
        if init_state != reward_location and init_state not in walls_location:
            non_walls.append(init_state)
            pvf_cumulative_rewards = []
            pvf_steps_to_goal = []
            for k in range(10):
                pvf_cumulative_rewards.append(all_results[k]['cumul_rewards'][init_state])
                pvf_steps_to_goal.append(all_results[k]['steps_to_goal'][init_state])
            mean_cumulative_rewards.append(np.mean(pvf_cumulative_rewards))
            std_cumulative_rewards.append(np.std(pvf_cumulative_rewards))
            mean_steps_to_goal.append(np.mean(pvf_steps_to_goal))
            std_steps_to_goal.append(np.std(pvf_steps_to_goal))

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax = axs[0]
    ax.errorbar(non_walls, mean_steps_to_goal, yerr=std_steps_to_goal, fmt='ro', ecolor='red')
    ax.set_title('n2v: number of steps')

    ax = axs[1]
    ax.errorbar(non_walls, mean_cumulative_rewards,
                yerr=std_cumulative_rewards, fmt='ro', ecolor='red')
    ax.set_title('n2v: cumulative reward')
    fig.suptitle('Grid size = ' + str(grid_size) + ', Dimension = ' + str(dimension) + ', Discount =' + str(discount))
    plt.savefig(
        '12sept/' + str(grid_size) + 'grid_' + str(dimension) + 'dimension_' + str(discount) + 'discount_' + str(
            num_samples) + 'samples' + str(window_size) + 'ws.pdf')


def main():
    if len(sys.argv) < 2:
        raise ValueError('Please enter the environment')

    env = sys.argv[1]
    DIMENSION = [3, 5, 10, 20, 30, 40, 50, 60, 70]

    if env == 'tworooms':
        environment = tworooms
        edgelist = 'node2vec/graph/tworooms_all_nodes.edgelist'
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
    elif env == 'oneroom':
        environment = oneroom
        edgelist = 'node2vec/graph/grid10.edgelist'
        reward_location = [9]
        walls_location = []
        num_states = 100
    else:
        raise ValueError('Environment not recognized')

    run_experiment(env, environment, edgelist, num_states, reward_location, walls_location, DIMENSION)


if __name__ == "__main__":
    main()
