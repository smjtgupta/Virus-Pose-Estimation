import numpy as np

# assume have 10 sets, each with 8 data points.
# want to generate 1/2 pairs belong to same set.

sample_factor = 100
num_sets = 10  # k
imgs_per_mat = 8 # p
num_imgs = num_sets*imgs_per_mat

num_rand_pairs = num_sets*num_sets * sample_factor
num_same_pairs = int((num_sets - 2)*num_rand_pairs*1.0/num_sets)
# first create pairs uniformly. Only 1/k will belong to same set
rand_pairs = np.random.randint(num_imgs,size=(num_rand_pairs,2))

# gen k-2 that are from same set.
same_pairs = np.random.randint(imgs_per_mat,size=(num_same_pairs,2))


all_pairs = np.concatenate((same_pairs,rand_pairs),axis=0)

groups = all_pairs / imgs_per_mat

ratio = (groups[:,0] == groups[:,1]).sum() * 1.0 / groups.shape[0]
print(ratio)

