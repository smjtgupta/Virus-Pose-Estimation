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
from __future__ import print_function
import numpy as np
import scipy.io as sio
import cv2
np.random.seed(1337)  # for reproducibility

import random
#from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AtrousConvolution2D, UpSampling2D, Deconvolution2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, Reshape, Permute, Cropping2D, merge, Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K

matfiles = ['../maps/5fct_map.mrc.mat', '../maps/3J4U_map.mrc.mat', '../maps/3E8K_map.mrc.mat', '../maps/5azi_map.mrc.mat', '../maps/2OM3_map.mrc.mat', '../maps/5b8c_map.mrc.mat', '../maps/5lvr_map.mrc.mat', '../maps/4AED_map.mrc.mat']

def load_data(matfiles):
# loads data into arrays
# assume matfiles of 2048 images
    img_size = 256
    imgs = np.zeros((img_size,img_size,0))
    #quats = np.zeros((0,4))

    for i in range(len(matfiles)):
        temp = sio.loadmat(matfiles[i])
        imgs = np.concatenate((imgs,temp['noisy_projections']),axis=2)
        #quats = np.concatenate((quats,temp['q']),axis = 0)
  
    imgs = np.expand_dims(imgs,axis = 3)
    img_mod = np.zeros((16384,28,28,1),dtype=np.float) 
    for i in range(16384):
        imglg = imgs[:,:,i,:]
        imgsm = cv2.resize(imglg,(28,28))
        img_mod[i,:,:,:] = np.expand_dims(imgsm,axis=2)
    return img_mod#, quats

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

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    #x.get
    return K.maximum(0.0,K.mean(x * y, axis=-1, keepdims=True))

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(8)]) - 1
    for d in range(8):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            #print("Positives:",z1,z2)
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 8)
            dn = (d + inc) % 8
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            #print("Negative:",z1,z2)
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    print(len(pairs))
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Convolution2D(32, 3, 3, input_shape = input_dim, border_mode = 'same', activation='relu',name='conv1_1'))
    seq.add(Convolution2D(32, 3, 3, border_mode = 'same', activation='relu',name='conv1_2'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 2
    seq.add(Convolution2D(64, 3, 3, border_mode = 'same', activation='relu',name='conv2_1'))
    seq.add(Convolution2D(64, 3, 3, border_mode = 'same', activation='relu',name='conv2_2'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 3
    #seq.add(Convolution2D(64, 3, 3, border_mode = 'same', activation='relu',name='conv3_1'))
    #seq.add(Convolution2D(64, 3, 3, border_mode = 'same', activation='relu',name='conv3_2'))
    #seq.add(Convolution2D(256, 3, 3, border_mode = 'same', activation='relu',name='conv3_3'))
    #seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Flatten(name='flatten'))
    #seq.add(Dense(128, activation='relu'))
    #seq.add(Dropout(0.1))
    #seq.add(Dense(128, activation='relu'))
    #seq.add(Dropout(0.1))
    #seq.add(Dense(128, activation='relu'))
    seq.summary()
    return seq

def create_feature_matching_network(flat_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape = flat_dim, activation='relu'))
    #seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    #seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.summary()
    return seq

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
imgs = load_data(matfiles)
X_train = imgs[:,:,:].reshape(16384,28*28)
X_train = X_train.astype('float32')
X_train /= 255
y_train = np.zeros(16384,dtype=np.uint8)
y_train[0:2048] = 7
y_train[2048:4096] = 6
y_train[4096:6144] = 5
y_train[6144:8192] = 4
y_train[8192:10240] = 3
y_train[10240:12288] = 2
y_train[12288:14336] = 1
input_dim = (28,28,1)
flat_dim = (3136,)
nb_epoch = 20

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(8)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)
print(tr_pairs.shape)
tr_pairs = tr_pairs.reshape(32752,2,28,28)

#digit_indices = [np.where(y_test == i)[0] for i in range(10)]
#te_pairs, te_y = create_pairs(X_test, digit_indices)

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

#fc11 = Dense(128, activation='relu')(processed_a)
#fc12 = Dense(128, activation='relu')(fc11)
#fc13 = Dense(128, activation='relu')(fc12)

#fc21 = Dense(128, activation='relu')(processed_b)
#fc22 = Dense(128, activation='relu')(fc21)
#fc23 = Dense(128, activation='relu')(fc22)

#seq.add(Dense(128, activation='relu'))
    #seq.add(Dropout(0.1))
    #seq.add(Dense(128, activation='relu'))
    #seq.add(Dropout(0.1))
    #seq.add(Dense(128, activation='relu'))

#featureTopExpand = Lambda(expand_dims,name='expand_top')(featureTop)
#featureBotExpand = Lambda(expand_dims,name='expand_bot')(featureBot)

#custom_corr = merge([featureTopExpand, featureBotExpand], mode='concat', concat_axis=2)
#print(custom_corr.shape)
# crm layers
#
#conv6 = Convolution1D(16, 3, border_mode = 'same', activation='relu',name='conv6_1')(custom_corr)
#conv6 = Convolution1D(16, 3, border_mode = 'same', activation='relu',name='conv6_2')(conv6)
#pool6 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#conv7 = Convolution1D(32, 3, border_mode = 'same', activation='relu',name='conv7_1')(conv6)
#conv7 = Convolution1D(32, 3, border_mode = 'same', activation='relu',name='conv7_2')(conv7)
#pool7 = Flatten(name='flatten')(conv7)

# keeping fc small for now to keep num parameters small
# camera pose estimation layer
#fc3 = Dense(128, activation='relu', name='fc1')(pool7)
#fc4 = Dense(128, activation='relu', name='fc2')(fc3)
#rotation_output = Dense(num_outputs, name='rotation')(fc4) #any activation needed here???????????

#distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([fc13, fc23])
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_a, feat_b])

model = Model(input=[input_a, input_b], output=distance)
model.summary()

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
#model.compile(loss='binary_crossentropy', optimizer=rms)
model.fit([np.expand_dims(tr_pairs[:, 0],axis=3), np.expand_dims(tr_pairs[:, 1],axis=3)], tr_y,
          validation_split = 0.1,
          batch_size=128,
          nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([np.expand_dims(tr_pairs[:, 0],axis=3), np.expand_dims(tr_pairs[:, 1],axis=3)])
tr_acc = compute_accuracy(pred, tr_y)
#pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
#te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
