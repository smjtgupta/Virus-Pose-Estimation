
from __future__ import absolute_import
import cv2
import os
import h5py
import numpy as np
import numpy.matlib as matlib
#from ImageDataGenerator import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AtrousConvolution2D, UpSampling2D, Deconvolution2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, Reshape, Permute, Cropping2D
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

K.set_image_dim_ordering('tf')  # Tensorflow ordering from now on
num_channels = 2


weights_path = '../models/vgg16_tf.h5'
train_raw_npy = '../data/train/train_im.npy'
train_label_npy = '../data/train/train_cam_gt.npy'
checkpoint_file = 'mynet.hdf5'

input_size = (188,620) 
batch_size = 8 

subtract_mean = 0
nb_epoch = 50
num_outputs = 3
 

# create data given directory containing all images and list
def create_data(data_path, list_file, npy_file, zoom=8,is_label=False):
    pass   


def load_data(raw, label):
    imgs_raw = np.load(raw)
    imgs_label = np.load(label)
    return imgs_raw, imgs_label

def load_net(weights_path,num_outputs=num_outputs,input_size=input_size):
    return vgg_like(weights_path,num_outputs=num_outputs,input_size=input_size)

def vgg_like(weights_path=None, num_outputs=3,input_size=None): 
    num_channels = 2
    if K.image_dim_ordering() == 'th':
	concat_axis = 1
        inputs = Input((num_channels,input_size[0],input_size[1]))
 	input_shape = (num_channels,input_size[0],input_size[1])
    else:
	concat_axis = 3
        inputs = Input((input_size[0],input_size[1],num_channels))
 	input_shape = (input_size[0],input_size[1],num_channels)

    conv1 = Convolution2D(64, 3, 3, activation='relu',name='conv1_1')(inputs)
    conv1 = ZeroPadding2D((1,1))(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu',name='conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = ZeroPadding2D((1,1))(pool1)
    conv2 = Convolution2D(128, 3, 3, activation='relu',name='conv2_1')(conv2)
    conv2 = ZeroPadding2D((1,1))(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu',name='conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = ZeroPadding2D((1,1))(pool2)
    conv3 = Convolution2D(256, 3, 3, activation='relu',name='conv3_1')(conv3)
    conv3 = ZeroPadding2D((1,1))(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu',name='conv3_2')(conv3)
    conv3 = ZeroPadding2D((1,1))(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu',name='conv3_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ZeroPadding2D((1,1))(pool3)
    conv4 = Convolution2D(512, 3, 3, activation='relu',name='conv4_1')(conv4)
    conv4 = ZeroPadding2D((1,1))(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu',name='conv4_2')(conv4)
    conv4 = ZeroPadding2D((1,1))(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu',name='conv4_3')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    conv5 = ZeroPadding2D((1,1))(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu',name='conv5_1')(conv5)
    conv5 = ZeroPadding2D((1,1))(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu',name='conv5_2')(conv5)
    conv5 = ZeroPadding2D((1,1))(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu',name='conv5_3')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    pool5 = Flatten(name='flatten')(pool5)
    fc6 = Dense(4096, activation='relu', name='fc1')(pool5)
    #fc6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', name='fc2')(fc6)
    #fc7 = Dropout(0.5)(fc7)
    output = Dense(num_outputs, name='predictions')(fc7)

    model = Model(input=inputs, output=output)

    #assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    #model.load_weights(weights_path, by_name=True)
    print('Model loaded.')

    model.summary()

    return model

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0]  = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_AREA)
    return imgs_p

def train(finetune=False,lr=1e-3):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
   

    train_raw, train_label = load_data(train_raw_npy,train_label_npy)
    train_raw = train_raw.astype('float32') 
    train_label = train_label.astype('float32')

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = load_net(weights_path,input_size=input_size)
    if finetune:
   	print("Finetuning using", checkpoint_file)
	model.load_weights(checkpoint_file)
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss' )

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    model.compile(loss='mse',
                  optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['mse'] )

    for i in range(10):
        print('Learning rate: {0}'.format(lr))
        K.set_value(model.optimizer.lr, lr)
        model.fit(train_raw, train_label, batch_size=batch_size,
                nb_epoch=nb_epoch, verbose=1, callbacks=[model_checkpoint,TensorBoard()])
        lr = lr *.1

def predict(image_file):
    palette = np.array([[0,0,0],[0,255,0],[255,0,0],[0,0,255],[255,0,255]],dtype=np.uint8)
 
    print ("Loading Image") 
    img = cv2.imread(image_file)
    transformed_img = img.astype(np.float32) - subtract_mean

    model = load_net(weights_path,num_outputs=num_outputs,input_size=img.shape)
    model.load_weights(checkpoint_file)

    if K.image_dim_ordering() == 'th':
        transformed_img = transformed_img.transpose(2,0,1)

    transformed_img = np.expand_dims(transformed_img,axis=0)

    if num_channels == 1:
        transformed_img = transformed_img[:,:,:,0]
        transformed_img = np.expand_dims(transformed_img,axis=3)
    
    image_size = img.shape
   
    print ("Predicting")
    prediction = model.predict(transformed_img, verbose=1)
    print prediction.shape
    prediction = prediction[0]
  

def validate():
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(checkpoint_file)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    train()
    pass


