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




from lspi import basis_functions
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

basis = basis_functions.DiscountedNode2vecBasis(
    'node2vec/graph/tworooms_withwalls.edgelist', num_actions=4,
    transition_probabilities=maze.domain.transition_probabilities, discount=discount,
                dimension=d, walks=maze.walks, walk_length=wl, num_walks=nw, window_size=ws,
                p=2, q=0.5, epochs=0, learning_rate=1)

discoutedn2v_model = gammanode2vec.DiscountedNode2Vec(maze.domain.num_states, d, ws, maze.walks, discount)

discountedn2v_basis, training_info = discoutedn2v_model.train_discounted_n2v(learning_rate=0.5, num_epochs=20)


with open('discounted_sigmoid20.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in discountedn2v_basis.values():
        tsv_writer.writerow(row)

with open('meta_discounted_sigmoid20.tsv', 'wt') as out_file:
    # tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in discountedn2v_basis.keys():
        out_file.write(row + '\n')


n2v_basis = compute_node2VecBasis(maze, dimension=d, walk_length=wl, num_walks=nw, window_size=ws,
                                              edgelist='node2vec/graph/threerooms.edgelist', epochs=50)
pvfs_basis = compute_ProtoValueBasis(maze, num_basis=d, walk_length=wl, num_walks=nw)

import numpy as np
from numpy.linalg import inv
from pygsp import graphs

T = graphs.Grid2d(N1=10, N2=10)

walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90,
                      9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      41, 42, 43, 44, 46, 47, 48, 49]

for wall in walls_location:
    T.W[wall, :] = 0.
    T.W[:, wall] = 0.

W = T.W.toarray()

for w in range(len(W)):
    sum = W[w].sum()
    for i in range(len(W[w])):
        if sum != 0:
            W[w,i] /= sum

I=np.identity(W.shape[1])
phi=inv(I-discount*W)

with open('SR.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in phi:
        tsv_writer.writerow(row)

with open('SR_meta.tsv', 'wt') as out_file:
    # tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in range(100):
        out_file.write(str(row) + '\n')

def feature_computer(state,p,gamma):
  if state=="No":
    I=np.identity(p.shape[1])
    phi=inv(I-gamma*p)
  else:
    I=np.identity(p.shape[1])
    phi=inv(I-gamma*p)[state,:]
  return phi

