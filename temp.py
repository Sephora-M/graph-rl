import numpy as np
import scipy.optimize as optimization
from learning_maze import LearningMazeDomain
from lspi import domains, basis_functions
import matplotlib.pyplot as plt
import pickle
from smoothness import compute_ProtoValueBasis,compute_node2VecBasis,least_squares,threerooms

dimensions = [30, 50, 100, 500]
plot_se_dims = []
wl = 100
nw = 500
K = 3
DISCOUNT = 0.9
environment_name = 'threerooms'

d = dimensions[0]
maze, V = threerooms(False, computeV=True, num_sample=1)

pvfs_mean_errors = []
# pvfsW_mean_errors = []
n2v_mean_errors = []

pvfs_std_errors = []
# pvfsW_std_errors = []
n2v_std_errors = []

pvfs_errors = []
# pvfsW_errors = []
n2v_errors = []

for d in dimensions:
    pvfs_basis = compute_ProtoValueBasis(maze, num_basis=d, walk_length=wl, num_walks=nw)
    pvf_params, pvf_error = least_squares(pvfs_basis, V, np.random.uniform(-1.0, 1.0, size=(d,)))
    pvfs_errors.append(pvf_error)
n2v_basis = compute_node2VecBasis(maze, dimension=d, walk_length=wl, num_walks=nw, window_size=10,
                                  edgelist='node2vec/graph/threerooms.edgelist')





from optimise_n2v import compute_node2VecBasis
from optimise import tworooms
import csv
from gammanode2vec import gammanode2vec

wl = 20
nw = 300
discount=.9
maze, _ = tworooms(False, nw, wl, discount)
maze.compute_samples(reset_policy=True)
d = 30
ws = 20

discoutedn2v_model = gammanode2vec.DiscountedNode2Vec(maze.domain.num_states, d, ws, maze.walks, discount)

discountedn2v_basis, training_info = discoutedn2v_model.train_discounted_n2v(learning_rate=0.5, num_epochs=20)


n2v_basis = compute_node2VecBasis(maze, dimension=d, walk_length=wl, num_walks=nw, window_size=ws,
                                              edgelist='node2vec/graph/threerooms.edgelist', epochs=50)
pvfs_basis = compute_ProtoValueBasis(maze, num_basis=d, walk_length=wl, num_walks=nw)

with open('discounted300.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in discountedn2v_basis.values():
        tsv_writer.writerow(row)

with open('meta_discounted300.tsv', 'wt') as out_file:
    # tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in discountedn2v_basis.keys():
        out_file.write(row + '\n')

