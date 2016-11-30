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
from keras.utils.np_utils import convert_kernel, to_categorical
import code
# for vgg
import warnings
import pickle
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
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.cluster import KMeans

matfiles = glob('../maps/*.mat')
checkpoint_file = 'model.hdf5'
kmeans_pickle = 'kmeans.pickle'
np.random.seed(1337)
img_size = 128
num_pairs=10000     #>1000
num_outputs = 4
num_clusters = 32 #set value here

batch_size = 256
nb_epoch = 1

def load_data(matfiles):
# loads data into arrays
# outputs imgs: h,w,imgs_per_mat,num_sets
#            q: imgs_per_mat, 4
# assume matfiles of 2048 images
    full_img_size = 256
    imgs_per_mat = 2048
    num_sets = len(matfiles)
    imgs = np.zeros((full_img_size,full_img_size,imgs_per_mat,num_sets))
    quats = np.zeros((imgs_per_mat,4,num_sets))

    for i in range(len(matfiles)):
        temp = sio.loadmat(matfiles[i])
        #for j in range(2048):
        imgs[:,:,:,i] = temp['noisy_projections']
        quats[:,:,i] = temp['q']
    
    # imgs (256,256,2048,22), quats (2048,4,22)
    return imgs, quats

def gen_data(imgs,quats,num_pairs):

    unused,imgs_per_mat,num_sets = imgs.shape[1:4]
    sample_factor = int( num_pairs  * 1.0/ (num_sets * ( 2 * num_sets - 2)) )
    #num_rand_pairs = num_sets * num_sets * sample_factor  # set this to zero if want only generate matching object pairs
    num_rand_pairs = 0 
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


def create_clusters(imgs,quats,num_clusters=32):
    # will only generate rotations for one set, gives 1M rots
    unused,imgs_per_mat,num_sets = imgs.shape[1:4]
    num_pairs_per_set = imgs_per_mat * imgs_per_mat
    num_pairs = num_pairs_per_set * num_sets
    img_rot = np.zeros((num_pairs_per_set,4))

    counter = 0
    for i in range(1):
        for j in range(imgs_per_mat):
            for k in range(imgs_per_mat):
                #index = k + num_pairs_per_set * j + num_pairs_per_set**2*i
                img_rot[counter] = quatmultiply(quats[j,:,i], quatinv(quats[k,:,i]) )
                counter += 1
   
    print('Creating kmeans clusters') 
    kmeans = KMeans(n_clusters=num_clusters,n_init=8,n_jobs=8,random_state=1337)
    kmeans.fit(img_rot)
    with open(kmeans_pickle,'wb') as pickleFile:
        pickle.dump(kmeans,pickleFile)
    print('Saved kmeans into pickle: {0}'.format(kmeans_pickle))
    return kmeans

def compute_clusters(img_rot):
    with open(kmeans_pickle,'r') as pickleFile:
        kmeans = pickle.load(pickleFile)
    
    rot_labels = kmeans.predict(img_rot)
    return to_categorical(rot_labels,num_clusters)

#kmeans.labels_
#kmeans.predict
#kmeans.cluster_centers_

    return img_rot


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
    if t0 >= 0:
        return np.array([t0, t1, t2, t3])
    else:
        return -np.array([t0, t1, t2, t3])

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

def log_error(y_true, y_pred):
    square_dist = K.square(y_pred-y_true)
    exp_dist = K.exp(square_dist)-1.0
    #log_dist = K.log(K.clip(square_dist, K.epsilon(), np.inf) + 1.)
    return K.mean(exp_dist, axis=-1)
    #first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    #second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    #return K.mean(K.square(first_log - second_log), axis=-1)

def norm_rmse(y_true, y_pred):
    norm = K.l2_normalize(y_pred, axis=-1)
    y_norm = 0
    if norm==0: 
       y_norm = y_pred
       return K.mean(K.square(y_norm - y_true))
    else:
       y_norm = y_pred/(norm+.000000001)
       return K.mean(K.square(y_norm - y_true))

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Convolution2D(16, 3, 3, input_shape = input_dim, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 2
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 3
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 4
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 5
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(Convolution2D(16, 3, 3, border_mode = 'same', activation='relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # final features
    seq.add(Flatten(name='flatten'))
    seq.summary()
    return seq

#def create_feature_matching_network(flat_dim):
#    seq = Sequential()
#    seq.add(Dense(128, input_shape = flat_dim, activation='relu'))
    #seq.add(Dropout(0.1))
#    seq.add(Dense(128, activation='relu'))
    #seq.add(Dropout(0.1))
#    seq.add(Dense(128, activation='relu'))
#    seq.summary()
#    return seq

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


if __name__ == '__main__':
    # the data, shuffled and split between train and test sets
    [imgs, quats] = load_data(matfiles)
    input_dim = (img_size,img_size,1)
    flat_dim = (img_size*img_size*4,) #change the number
            
    # network definition
    base_network = create_base_network(input_dim)
    
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    #feature_matching_network = create_feature_matching_network(flat_dim)
    
    #feat_a = feature_matching_network(processed_a)
    #feat_b = feature_matching_network(processed_b)
    
    featureTopExpand = Lambda(expand_dims,name='expand_top')(processed_a)
    featureBotExpand = Lambda(expand_dims,name='expand_bot')(processed_b)
    
    custom_corr = merge([featureTopExpand, featureBotExpand], mode='concat', concat_axis=2)
    
    #crm layers
    conv6 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(custom_corr)
    conv6 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv6)
    pool6 = MaxPooling1D(pool_length=2)(conv6)
    
    conv7 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv6)
    conv7 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv7)
    pool7 = MaxPooling1D(pool_length=2)(conv7)

    conv8 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv7)
    conv8 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv8)
    pool8 = MaxPooling1D(pool_length=2)(conv8)
    
    conv9 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv8)
    conv9 = Convolution1D(16, 3, border_mode = 'same', activation='relu')(conv9)
    pool9 = MaxPooling1D(pool_length=2)(conv9)
    
    pool9 = Flatten(name='flatten')(pool9)
    
    # keeping fc small for now to keep num parameters small
    # camera pose estimation layer
    fc1 = Dense(32, activation='relu', name='fc1')(pool9)
    fc2 = Dense(32, activation='relu', name='fc2')(fc1)
    
    #classification = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_a, feat_b])
    #rotation = Dense(num_outputs, name='rotation')(fc2)
    rotation = Dense(num_clusters, activation='softmax', name='rotation')(fc2)
    
    #model = Model(input=[input_a, input_b], output=[classification])
    model = Model(input=[input_a, input_b], output=[rotation])
    #model = Model(input=[input_a, input_b], output=[classification, rotation])
    model.summary()
    
    # train
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss' )
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics = ['categorical_accuracy'])
    #model.compile(loss=norm_rmse, optimizer=rms)
    #model.compile(loss=log_error, optimizer=rms)
    #model.compile(loss=[contrastive_loss], optimizer=rms)
    #model.compile(loss=[contrastive_loss,'mse'],  loss_weights=[10.0, 1.0], optimizer=rms)
    #model.fit([np.expand_dims(tr_pairs[:, 0],axis=3), np.expand_dims(tr_pairs[:, 1],axis=3)], tr_y,
    #          validation_split = 0.1,
    #          batch_size=128,
    #          nb_epoch=nb_epoch)

    lr=1e-3
    for i in range(2): # num times to drop learning rate
            print('Learning rate: {0}'.format(lr))
            K.set_value(model.optimizer.lr, lr)
            for j in range(20): # num times to generate data ~8M images
                print("i,j={0},{1}".format(i,j))
                [img_pairs, img_rot, img_label] = gen_data(imgs, quats, num_pairs)
                rot_label = compute_clusters(img_rot)  # rotation cluster labels
                print(rot_label.shape)
                input_top = np.expand_dims(img_pairs[:,0], axis=3)
                input_bot = np.expand_dims(img_pairs[:,1], axis=3)

                model.fit([input_top, input_bot], [rot_label], validation_split = 0.1, 
                          batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[model_checkpoint])
                #model.fit([input_top, input_bot], [img_label, img_rot], shuffle=True, validation_split = 0.1, batch_size=128, nb_epoch=nb_epoch)
                # compute final accuracy on training and test sets
                #retval = model.predict([input_top, input_bot])
                #print(len(retval))
                #pred = retval
                #print pred
                #rot = retval[1]
                #tr_acc = compute_accuracy(pred, img_label)
                #print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
            lr = lr*.1
