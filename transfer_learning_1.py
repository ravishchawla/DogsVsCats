
# coding: utf-8

# In[1]:

import numpy;
import tensorflow as tf
import pickle;

from tflearn.data_utils import to_categorical

import os;
from glob import glob;

from skimage import color, io;
from scipy.misc import imresize;

from sklearn.cross_validation import train_test_split;
import sys;


# In[2]:

from tf_vgg import vgg16
from tf_vgg import utils


# In[3]:

def print_progess(currentProgress, maxProgress, numberDashes, message = ""):
    progress = numberDashes * (currentProgress / maxProgress) + 1;
    dashes = "=" * int(progress) + "> "
    percentage = int(100 * currentProgress / maxProgress) + 1;
    if percentage > 100:
        percentage = 100;
    sys.stdout.write("\r " + "<" + message + "> " + dashes + "{0:.2f}".format(percentage) + "%");
    sys.stdout.flush();


# In[4]:

files_path = '/Users/ravishchawla/workspace/machinelearning/datasets/dogs_vs_cats/train/';

cat_files_path = os.path.join(files_path, 'cats/cat*.jpg');
dog_files_path = os.path.join(files_path, 'dogs/dog*.jpg');

cat_files = sorted(glob(cat_files_path));
dog_files = sorted(glob(dog_files_path));

max_images_each = 2000;

num_files = max_images_each + max_images_each;
print('cat files: ' + str(len(cat_files)) + ' dog files: ' + str(len(dog_files)));

size_image = 224;



# In[5]:

#try:
#(all_x, all_y) = pickle.load(open('datafiles/all_x_y.dat', 'rb'));
#print('loaded');
#except:
all_x = numpy.zeros((num_files, size_image, size_image, 3), dtype='float64');
all_y = numpy.zeros(num_files);
count = 0;
set_count = 0;

for file in cat_files:
    if set_count > max_images_each:
        break;
    try:
        img = io.imread(file);
        new_img = imresize(img, (size_image, size_image, 3));
        all_x[count] = numpy.array(new_img);
        all_y[count] = 0;
        count = count + 1;
        set_count = set_count + 1;
        print_progess(set_count, max_images_each, 40);
    except f:
        print(f);

print("");
set_count = 0;
for file in dog_files:
    if set_count > max_images_each:
        break;
    try:
        img = io.imread(file);
        new_img = imresize(img, (size_image, size_image, 3));
        all_x[count] = numpy.array(new_img);
        all_y[count] = 1;
        count = count + 1;
        set_count = set_count + 1;
        print_progess(set_count, max_images_each, 40);
    except:
        continue

print("");

#pickle.dump((all_x, all_y), open('datafiles/all_x_y.dat', 'wb'));


# In[6]:

X, X_test, Y, Y_test = train_test_split(all_x, all_y, test_size = 0.1, random_state = 42);

Y = to_categorical(Y, 2);
Y_test = to_categorical(Y_test, 2);

num_epochs = 100;
display_step = 1;

print("X: ",X.shape);
print("X_t: ",X_test.shape);
print("Y: ",Y.shape);
print("Y_t: ", Y_test.shape);


# In[ ]:




# In[5]:

x = tf.placeholder("float", [None, 224, 224, 3]);
y = tf.placeholder("float", [None, 2]);

t_vars = tf.trainable_variables();
d_vars = [var for var in t_vars if 'fc' in var.name]

vgg = vgg16.Vgg16();
vgg.build(x);

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.prob, y));

optimzer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost);


#correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(y, 1));
#accuracy = tf.reduce_mean(tf.case(correct_prediction, tf.float32));

init = tf.initialize_all_variables();


# In[ ]:

init = tf.initialize_all_variables();
config = tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)));
batch_length = 500;

with tf.Session() as session:
    session.run(init);
    for epoch in range(0, num_epochs):
        avg_cost = 0;
        total_batch = int(len(X) / batch_length);
        X_batches = numpy.array_split(X, total_batch);
        Y_batches = numpy.array_split(Y, total_batch);
        
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i];
            
            _, c = session.run([optimzer, cost], feed_dict={x : batch_x,
                                                           y: batch_y});
            
            avg_cost = avg_cost + (c / total_batch);
        
        if epoch % display_step == 0:
            print(epoch, num_epochs, 40, "cost = " + "{:.9f}".format(avg_cost))
    print("Optimization finished");
    
    correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(y, 1));
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"));
    print("Accuracy: ", accuracy.eval({x: X_test, y: Y_test}));
    global result;
    result = tf.argmax(vgg.prob, 1).eval({x: X_test, y:Y_test});


# In[ ]:




# In[ ]:




# In[ ]:



