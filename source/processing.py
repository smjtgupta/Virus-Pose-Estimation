from __future__ import absolute_import
import cv2
import os
import h5py
import numpy as np
import numpy.matlib as matlib
#from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AtrousConvolution2D, UpSampling2D, Deconvolution2D
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

# for vgg
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

from keras.layers import merge, Input
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
#from keras.magenet_utils import decode_predictions, preprocess_input
import scipy.io as sio

# assume matfiles of 2048 images
def load_data(matfiles):
    num_pairs = 2048
    imgs_per_mat = 2048
    img_size = 256
    imgs = np.zeros((img_size,img_size,0))
    quats = np.zeros((4,0))

    for i in range(len(matfiles)):
        temp = sio.loadmat(matfiles[0])
        imgs = np.concatenate((imgs,temp['noisy_projections']),axis=2)
        quats = np.concatenate((quats,temp['q']),axis = 1)
  
    imgs = np.expand_dims(imgs,axis = 3) 
    num_imgs = imgs.shape[2]
    pairs = np.random.randint(num_imgs,size=(num_pairs,2))
    img_pairs = np.zeros((num_pairs,img_size,img_size,2))
    img_rot = np.zeros((num_pairs,4))
    img_label = np.zeros((num_pairs,1))

    for i in range(num_pairs):
        a = pairs[i,0]
        b = pairs[i,1]
        img_pairs[i] = np.concatenate((imgs[:,:,a],imgs[:,:,b]),axis=2)
        img_label[i] = 1 * ( (a/imgs_per_mat) == (b/imgs_per_mat) )
        img_rot[i] = img_label[i] * quatmultiply(quats[:,b], quatinv(quats[:,a]) )

    return img_pairs, img_rot, img_label


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
