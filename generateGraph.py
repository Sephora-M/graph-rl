import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
from scipy.sparse import lil_matrix

def value_iteration(G, finish_state, obstacles, walls):
    V = [0]*G.N
    R = [0]*G.N
    R[finish_state] = 100
    gamma = 0.9
    success_prob = [1]*G.N
    for i in obstacles:
        success_prob[i] = .9
    for i in walls:
        success_prob[i] = .0
    epsilon = .0001
    diff = 100
    iterations = 0
    while diff > epsilon:
        iterations = iterations+1
        diff = 0
        for s in xrange(G.N):
            if s == finish_state:
                max_a = success_prob[s]*R[s]
            else:
                max_a = float('-inf')
                for s_prime in G.W.getcol(s).nonzero()[0]:
                    new_v = success_prob[s]*(R[s] + gamma*V[s_prime])
                    if new_v > max_a:
                        max_a = new_v
            diff = diff + abs(V[s]-max_a)
            V[s] = max_a
    print "number of iterations:"
    print iterations
    return V

def example_fullyConnected():
    G = graphs.FullConnected(N=20)
    G.set_coordinates(kind='spring', seed=42)
    fig, axes = plt.subplots(1, 2)
    _ = axes[0].spy(G.W, markersize=5)
    G.plot(ax=axes[1])
    plt.show()


N1 = 10
N2 = 10
G = graphs.Grid2d(N1=N1, N2=N2)
fig, axes = plt.subplots(1, 2)
_ = axes[0].spy(G.W)
G.plot(ax=axes[1])
plt.show()

# OBSTACLES = [40, 41, 42, 43, 44, 55, 56, 57, 58 ,59]
# OBSTACLES = np.random.randint(150, size=20)
# WALLS = [76, 77, 78, 79, 80, 81, 82, 83, 112, 113, 114, 115, 116, 117, 118, 119]

OBSTACLES = []
WALLS = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
GOAL = 9

nodes = np.ones(height*width)

for i in OBSTACLES:
    nodes[i] = 2.

for i in WALLS:
    nodes[i] = 0.

nodes[GOAL] = 3.

fig, ax = plt.subplots(1, 1)
G.plot_signal(nodes, vertex_size=60, ax=ax)
plt.show()

n = range(26)
n[15] = 17
n[16:] = range(20,30)
sub_G = G.subgraph(n)

N = N1 * N2
x = np.kron(np.ones((N1, 1)), (np.arange(N2) / float(N2)).reshape(N2, 1))
y = np.kron(np.ones((N2, 1)), np.arange(N1) / float(N1)).reshape(N, 1)
y = np.sort(y, axis=0)[::-1]
coords = np.concatenate((x, y), axis=1)
sub_coords= coords[n,:]
sub_G.set_coordinates(sub_coords)

# fig, axes = plt.subplots(1, 2)
# _ = axes[0].spy(sub_G.W)
# sub_G.plot(ax=axes[1])

G = sub_G
fig, axes = plt.subplots(1, 2)
_ = axes[0].spy(G.W)
G.plot(ax=axes[1])

#G.compute_laplacian('normalized')
G.compute_laplacian('combinatorial')
G.compute_fourier_basis(recompute=True)

fig, axes = plt.subplots(5, 5, figsize=(20, 12))
i = 0
for j, axesj in enumerate(axes):
    for s, ax in enumerate(axesj):
        G.plot_signal(G.U[:, i+1], vertex_size=30, ax=ax)
        _ = ax.set_title('Eigenvector {}'.format(i+2))
        ax.set_axis_off()
        i = i+1

fig.tight_layout()
plt.show()


V = value_iteration(G, GOAL, OBSTACLES, WALLS)
V = np.array(V)
fig, ax = plt.subplots(1, 1)
G.plot_signal(V, vertex_size=60, ax=ax)
plt.show()

# --------------------------------------------------

from learning_maze import LearningMazeDomain
import matplotlib.pyplot as plt
import numpy as np

height = 10
width = 10
reward_location = 9
initial_state = None  # np.array([25])
obstacles_location = [14, 13, 24, 23, 29, 28, 39, 38]  # range(height*width)
walls_location = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
obstacles_transition_probability = .2
maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location, num_sample=10000)
maze.random_policy_cumulative_rewards
num_steps, learned_policy, samples, distances = maze.learn_node2vec_basis(dimension=5, walk_length=30, num_walks=10, window_size=10, p=1, q=1, epochs=3, explore=0, discount=.99, max_steps=1000, max_iterations=500)

num_steps, learned_policy, samples, distances = maze.learn_proto_values_basis(num_basis=30, explore=0, max_steps=500, max_iterations=1000)

G = maze.domain.graph


#----------------------------------------------------
from learning_maze import LearningMazeDomain
import matplotlib.pyplot as plt
import numpy as np
height = 6
width = 6
reward_location = 4
initial_state = None  # np.array([25])
obstacles_location = []  # range(height*width)
walls_location = []
obstacles_transition_probability = .2
maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location, num_sample=2000)
maze.random_policy_cumulative_rewards
num_steps, learned_policy, samples, distances = maze.learn_node2vec_basis(dimension=30, walk_length=30, num_walks=10, window_size=10, p=1, q=1, epochs=3, explore=0, discount=.99, max_steps=1000, max_iterations=500,edgelist ='node2vec/graph/grid6.edgelist')



for state in range(height*width):
    if state != reward_location and state not in walls_location:
        steps_to_goal = 0
        maze.domain.reset(np.array([state]))
        absorb = False
        samples = []
        max_steps = 1000
        print maze.domain._state
        while (not absorb) and (steps_to_goal < max_steps):
            action = learned_policy.select_action(maze.domain.current_state())
            sample = maze.domain.apply_action(action)
            absorb = sample.absorb
            steps_to_goal += 1
            samples.append(sample)
        print steps_to_goal

for state in [0,18,23]:
    steps_to_goal = 0
    maze.domain.reset(np.array([state]))
    absorb = False
    samples = []
    max_steps = 1000
    print maze.domain._state
    while (not absorb) and (steps_to_goal < max_steps):
        action = learned_policy.select_action(maze.domain.current_state())
        sample = maze.domain.apply_action(action)
        absorb = sample.absorb
        steps_to_goal += 1
        samples.append(sample)
    print steps_to_goal
    print samples

# --------------------------------------------------

def select_action(domain, policy, V):

    max_v = float('-inf')
    selected_action = -1
    for action in range(4):
        next_location = domain.next_location(domain.current_state(), action)
        if V[next_location] > max_v:
            max_v = V[next_location]
            selected_action = action

    return selected_action

steps_to_goal = 0
maze.domain.reset(np.array([25]))
absorb = False
samples = []
max_steps=10000
maze.domain._state
while (not absorb) and (steps_to_goal < max_steps):
    action = learned_policy.select_action(maze.domain.current_state())
    sample = maze.domain.apply_action(action)
    absorb = sample.absorb
    steps_to_goal += 1
    samples.append(sample)


# --------------------------------------------------

steps_to_goal = 0
#maze.domain.reset()
maze.domain.reset(np.array([25]))
absorb = False
samples = []
max_steps=10000
maze.domain._state
while (not absorb) and (steps_to_goal < max_steps):
    action = learned_policy.select_action(maze.domain.current_state())
    sample = maze.domain.apply_action(action)
    absorb = sample.absorb
    steps_to_goal += 1
    samples.append(sample)

steps_to_goal
samples[0:10]
# --------------------------------------------------

from learning_maze import LearningMazeDomain
import matplotlib.pyplot as plt
import numpy as np

height = 10
width = 10
reward_location = 9
initial_state = None  # np.array([25])
obstacles_location = []  #range(height*width)
walls_location = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location, initial_state, .9, num_sample=2000)
maze.random_policy_cumulative_rewards
num_steps, learned_policy, samples, distances = maze.learn_node2vec_basis(
    dimension=30, walk_length=80, num_walks=10, window_size=10, p=1, q=1, epochs=6, explore=0,
    max_steps=1000, max_iterations=100)
#num_steps, learned_policy, samples, distances = maze.learn_proto_values_basis(num_basis=30, explore=0, max_steps=1000, max_iterations=1000)
#num_steps, learned_policy, samples, distances = maze.learn_polynomial_basis(degree=3, explore=0, max_steps=1000, max_iterations=100)

G = maze.domain.graph

V = np.zeros(height*width)

for state in range(height*width):
    if state != reward_location:
        V[state] = learned_policy.calc_v_value(np.array([state]))


fig, ax = plt.subplots(1, 1)
G.plot_signal(V, vertex_size=60, ax=ax)
plt.show()


V = value_iteration(G, reward_location, obstacles_location, walls_location)
V = np.array(V)


traj = np.zeros(height*width)
for sample in maze.samples:
    traj[sample.state[0]] += 1

fig, ax = plt.subplots(1, 1)
G.plot_signal(traj, vertex_size=60, ax=ax)
plt.show()

from pygsp import graphs, filters, plotting
path = 'node2vec/graph/grid9.edgelist'
file = open(path,"a+")
N1 = N2 = 9
G = graphs.Grid2d(N1=N1, N2=N2)
X, Y, _ = G.get_edge_list()
for x, y in zip(X,Y):
    file.write("%d %d\n" % (x,y))

file.close()
