'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
import cv2
import os
import h5py
import numpy as np
import numpy.matlib as matlib
import scipy.io as sio
#from ImageDataGenerator import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AtrousConvolution2D, UpSampling2D, Deconvolution2D
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, Reshape, Permute, Cropping2D, merge, Embedding
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.objectives import categorical_crossentropy
from keras.engine.training import weighted_objective
import tensorflow as tf
from functools import partial
from itertools import product
from keras.utils.np_utils import convert_kernel
import code
np.random.seed(1337)
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

# for custom scaling
from keras import backend as K
from keras.engine.topology import Layer

import numpy as np
import warnings
from glob import glob
from keras.layers import merge, Input
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.optimizers import SGD, RMSprop

matfiles = glob('../maps/*.mat')
checkpoint_file = 'model.hdf5'
kmeans_pickle = 'kmeans.pickle'
np.random.seed(1337)
img_size = 256
num_pairs = 20000     #>1000
nb_epoch = 1

def load_data(matfiles):
# loads data into arrays
# outputs imgs: h,w,imgs_per_mat,num_sets
#            q: imgs_per_mat, 4
# assume matfiles of 2048 images
    img_size = 256
    imgs_per_mat = 2048
    num_sets = len(matfiles)
    imgs = np.zeros((img_size,img_size,imgs_per_mat,num_sets))
    for i in range(len(matfiles)):
        temp = sio.loadmat(matfiles[i])
        #for j in range(2048):
        imgs[:,:,:,i] = temp['noisy_projections']
    
    # imgs (256,256,2048,22), quats (2048,4,22)
    return imgs, quats

def gen_data(imgs,quats,num_pairs):

    img_size,imgs_per_mat,num_sets = imgs.shape[1:4]
    sample_factor = int( num_pairs  * 1.0/ (num_sets * ( 2 * num_sets - 2)) )
    num_rand_pairs = num_sets * num_sets * sample_factor  # set this to zero if want only generate matching object pairs
    num_same_pairs = num_pairs - num_rand_pairs   # num_same_pairs = int((num_sets - 2)*num_rand_pairs*1.0/num_sets)
    
    # first create pairs uniformly. Only 1/k will belong to same set
    rand_pairs_set = np.random.randint(num_sets,size=(num_rand_pairs,2))
    rand_pairs_index = np.random.randint(imgs_per_mat,size=(num_rand_pairs,2))
    same_pairs_set = np.random.randint(num_sets,size=(num_same_pairs))
    same_pairs_index = np.random.randint(imgs_per_mat,size=(num_same_pairs,2))

    img_pairs = np.zeros((num_pairs,2,img_size,img_size))
    img_rot = np.zeros((num_pairs,4))
    img_label = np.zeros((num_pairs))

    for i in range(num_rand_pairs):
        a,b = rand_pairs_index[i]
        c,d = rand_pairs_set[i]
        img_pairs[i,0] = cv2.resize(imgs[:,:,a,c],(img_size,img_size))
        img_pairs[i,1] = cv2.resize(imgs[:,:,b,d],(img_size,img_size))
        img_label[i] = 1 * ( c == d )
        img_rot[i] = img_label[i] * quatmultiply(quats[b,:,d], quatinv(quats[a,:,c]) )

    for i in range(num_same_pairs):
        a,b = same_pairs_index[i]
        c = same_pairs_set[i]
        img_pairs[num_rand_pairs + i,0] = cv2.resize(imgs[:,:,a,c],(img_size,img_size))
        img_pairs[num_rand_pairs + i,1] = cv2.resize(imgs[:,:,b,c],(img_size,img_size))
        img_label[num_rand_pairs + i] = 1
        img_rot[num_rand_pairs + i] = img_label[i] * quatmultiply(quats[b,:,c], quatinv(quats[a,:,c]) )

    return img_pairs, img_rot, img_label

def expand_dims(layers):
    return K.expand_dims(layers)

def quatinv(q):
    return np.array([q[0],-q[1],-q[2],-q[3] ] )

def quatmultiply(q,r):
    q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3];
    r0 = r[0]; r1 = r[1]; r2 = r[2]; r3 = r[3];
    t0=(r0*q0-r1*q1-r2*q2-r3*q3)
    t1=(r0*q1+r1*q0-r2*q3+r3*q2)
    t2=(r0*q2+r1*q3+r2*q0-r3*q1)
    t3=(r0*q3-r1*q2+r2*q1+r3*q0)
    return np.array([t0, t1, t2, t3])

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    # layer 1
    seq.add(Convolution2D(16, 3, 3, input_shape = input_dim, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 2
    seq.add(Convolution2D(32, 3, 3, input_shape = input_dim, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(32, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 3
    seq.add(Convolution2D(64, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(64, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 4
    seq.add(Convolution2D(128, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(128, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(128, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 5
    seq.add(Convolution2D(256, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(256, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(256, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 6
    seq.add(Convolution2D(512, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(512, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(512, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # final features
    seq.add(Flatten(name='flatten'))
    seq.summary()
    return seq

def create_feature_matching_network(flat_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape = flat_dim, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.summary()
    return seq

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

if __name__ == '__main__':
	# the data, shuffled and split between train and test sets
    imgs = load_data(matfiles)
    input_dim = (img_size,img_size,1)
    flat_dim = (img_size*img_size*(512)/(64*64),) #change the number
		
	# network definition
    base_network = create_base_network(input_dim)

    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)

	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    feature_matching_network = create_feature_matching_network(flat_dim)

    feat_a = feature_matching_network(processed_a)
    feat_b = feature_matching_network(processed_b)

	classification = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_a, feat_b])
	
    model = Model(input=[input_a, input_b], output=[classification])
	model.summary()

	# train
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss')
    rms = RMSprop()
    model.compile(loss=[contrastive_loss], optimizer=rms)
	
    lr=1e-3
    train_err = np.zeros(40,dtype=np.float32)
    for i in range(2): # num times to drop learning rate
        print('Learning rate: {0}'.format(lr))
        K.set_value(model.optimizer.lr, lr)
        for j in range(20): # num times to generate data ~8M images
            [img_pairs, img_rot, img_label] = gen_data(imgs, quats, num_pairs)
            input_top = np.expand_dims(img_pairs[:,0], axis=3)
            input_bot = np.expand_dims(img_pairs[:,1], axis=3)
            model.fit([input_top, input_bot], [img_label], validation_split = 0.1, batch_size=128, nb_epoch=nb_epoch, callbacks=model_checkpoint)
	        # compute final accuracy on training and test sets
            retval = model.predict([input_top, input_bot])
            #print(len(retval))
            pred = retval
	        #print pred
	        #rot = retval[1]
            tr_acc = compute_accuracy(pred, img_label)
            train_err[i*10+j] = 1 - tr_acc
            print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

