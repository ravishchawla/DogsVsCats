
# coding: utf-8

# In[56]:

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import SGD, rmsprop

import datetime;

import os;
import os.path;
from os.path import expanduser;
import pickle;


# In[49]:

home = expanduser("~");
files_data_dir = home + "/workspace/machinelearning/datasets/dogs_vs_cats/";

train_data_dir = files_data_dir + "train/";
test_data_dir = files_data_dir + "test/";

cats_dir = "cats/";
dogs_dir = "dogs/";

num_train_samples = 2048;
num_test_samples = 2048;
num_epoch = 50;


# In[50]:

num_filters = {
    'wc1' : 32,
    'wc2' : 32,
    'wc3' : 64,
    'wc4' : 64,
    'fc1' : 64,
    'fc2' : 1,
}


filter_size = {
    'wc1' : 3,
    'wc2' : 3,
    'wc3' : 3,
    'wc4' : 3,
}

input_size = {
    'wc1' : (250, 250, 3),
    'wc2' : (125, 125, 32),
    'wc3' : (63, 63, 32),
    'wc4' : (16, 16, 64),
}

strides = {
    
    'wc1' : (1, 1),
    'wc2' : (1, 1),
    'wc3' : (1, 1),
    'wc4' : (1, 1),
}

activation_type = {
    'wc1' : 'relu',
    'wc2' : 'relu',
    'wc3' : 'relu',
    'wc4' : 'relu',
    'fc1' : 'relu',
    'fc2' : 'sigmoid',
}

pool_ratio = {
    'wc1' : (2, 2),
    'wc2' : (2, 2),
    'wc3' : (2, 2),
    'wc4' : (2, 2),
}

dropout_ratio = {
    'fc1' : 0.5,
}

init_type = 'glorot_normal';


# In[51]:

model = Sequential();



conv1 = Convolution2D(num_filters['wc1'], filter_size['wc1'], filter_size['wc1'], input_shape=input_size['wc1'], subsample=strides['wc1'], init=init_type);
model.add(conv1);

act1 = Activation(activation_type['wc1']);
model.add(act1);

pool1 = MaxPooling2D(pool_size=pool_ratio['wc1']);
model.add(pool1);



conv2 = Convolution2D(num_filters['wc2'], filter_size['wc2'], filter_size['wc2'], input_shape=input_size['wc2'], subsample=strides['wc2'], init=init_type);
model.add(conv2);

act2 = Activation(activation_type['wc2']);
model.add(act2);

pool2 = MaxPooling2D(pool_size=pool_ratio['wc2']);
model.add(pool2);



conv3 = Convolution2D(num_filters['wc3'], filter_size['wc3'], filter_size['wc3'], input_shape=input_size['wc3'], subsample=strides['wc3'], init=init_type);
model.add(conv3);

act3 = Activation(activation_type['wc3']);
model.add(act3);

pool3 = MaxPooling2D(pool_size=pool_ratio['wc3']);
model.add(pool3);



#conv4 = Convolution2D(num_filters['wc4'], filter_size['wc4'], filter_size['wc4'], input_shape=input_size['wc4'], subsample=strides['wc4']);
#model.add(conv4);

#act4 = Activation(activation_type['wc4']);
#model.add(act4);

#pool4 = MaxPooling2D(pool_size=pool_ratio['wc4']);
#model.add(pool4);


model.add(Flatten());



fc1 = Dense(num_filters['fc1'], init=init_type);
model.add(fc1);

act3 = Activation(activation_type['fc1']);
model.add(act3);

drop1 = Dropout(dropout_ratio['fc1']);
model.add(drop1);



fc2 = Dense(num_filters['fc2'], init=init_type);
model.add(fc2);

act4 = Activation(activation_type['fc2']);
model.add(act4);


# In[52]:

img_width = 250;
img_height = 250;

data_batch_size = 64;

data_class_mode = 'binary';


loss_func = 'binary_crossentropy';

optimizer_func = 'sgd';

metric_types = ['accuracy'];

learning_rate = 0.001;

decay_rate = 1e-10;


data_rescale_ratio = 1./255;

data_sheer_range = 0.2;

data_zoom_range = 0.2;

data_horizontal_flip = True;


# In[53]:

#opt = SGD(lr=learning_rate, decay=decay_rate);
opt = 'adadelta';

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['binary_accuracy'])


#model.load_weights('model_sgd_86')
train_datagen = ImageDataGenerator(rescale = data_rescale_ratio, shear_range=data_sheer_range, zoom_range=data_zoom_range, horizontal_flip=data_horizontal_flip);

test_datagen = ImageDataGenerator(rescale = data_rescale_ratio);


# In[54]:

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=data_batch_size, class_mode=data_class_mode);

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height), batch_size=data_batch_size, class_mode=data_class_mode);


# In[41]:

model.fit_generator(train_generator, samples_per_epoch=num_train_samples, nb_epoch=num_epoch, validation_data=test_generator, nb_val_samples=num_test_samples);


# In[62]:

model.save_weights(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"));


# In[ ]:



