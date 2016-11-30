from __future__ import absolute_import
import cv2
import os
import h5py
import numpy as np
import numpy.matlib as matlib
from sklearn.metrics import mean_squared_error
from math import sqrt
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

K.set_image_dim_ordering('tf')  # Tensorflow ordering from now on
num_channels = 2


weights_path = '../models/mynet.h5'
valid_raw_npy = '../data/valid/valid_im.npy'
valid_label_npy = '../data/valid/valid_cam_gt.npy'
checkpoint_file = 'mynet.hdf5'

input_size = (188,620) 
batch_size = 8 

subtract_mean = 0
nb_epoch = 50
num_outputs = 3
 

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

def predict(image_file):
     
	print ("Loading Image") 
	transformed_img = image_file.astype(np.float32)
	transformed_img = np.expand_dims(transformed_img,axis=0)
	print transformed_img.shape
	img_shape = (188,620)

	model = load_net(weights_path,num_outputs=num_outputs,input_size=img_shape)
	model.load_weights(checkpoint_file)

 
	print ("Predicting")
	prediction = model.predict(transformed_img, verbose=1)
    #print prediction.shape
    #prediction = prediction[0]
    #print prediction
	return prediction


def validate():
	print ("Loading Image") 
	valid_im = np.load(valid_raw_npy).astype(np.float32)
	valid_gt = np.load(valid_label_npy)[:,0:3]
	predicted_gt = np.zeros((valid_im.shape[0],3),dtype=np.float)
	img_shape = (188,620)
	model = load_net(weights_path,num_outputs=num_outputs,input_size=img_shape)
	model.load_weights(checkpoint_file)

 
	print ("Predicting")

	
	for i in range(valid_im.shape[0]):
		img_stack = np.expand_dims(valid_im[i],axis=0)   
		predicted_gt[i,:] = model.predict(img_stack)

	rmse = sqrt(mean_squared_error(valid_gt,predicted_gt))
	print (rmse)
	np.save("predicted_pose",predicted_gt)

if __name__ == '__main__':
	validate()
