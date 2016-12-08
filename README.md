# Dogs vs Cats

## Ravish Chawla

This project repository contains code for implementing Neural Network Learning for the Dogs vs Cats classification task on [Kaggle](http://kaggle.com/c/dogs-vs-cats-redux-kernels-edition).

### Libraries
* Tensorflow
* Keras
* TF Learn
* Open CV
* Pandas/Numpy/Scipy

### Dataset

The dataset used for this project is provided by the Kaggle competition. The python notebooks expect the code to be in the following directory format:

../machinelearning/datasets/dogsvcats
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; train/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats###*.jpg
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dogs/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dogs###*.jpg
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; test/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats###*.jpg
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dogs/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dogs###*.jpg
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; kaggle/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ###*.jpg

The Train directory contains images used for training the model and the Test directory contains images used for testing the model. Both directories are labeled using their file names. The kaggle repository contains unlabeled images used for testing.

### Code Structure

Tensorflow is the main library, and Keras and TFLearn are high-level libraries that support a Tensorflow Backend. The work for this project is in different python notebooks, where each stage of the project is it's own notebook.

* **tensorflow_practice.ipnyb** - This notebook contains practice code to learn Tensorflow objects and functions, and understand how to use the library.

* **cnn_classifier.ipnyb** - First classifier written with Tensorflow. This notebook contains practice code written to learn Tensorflow and understand how to write a Neural Network to train the cats and dogs dataset.

* **cnn_classifier_2.ipnyb** - Revision of the first classifier, also in Tensorflow. Thie notebook uses OpenCV for image preprocessing. Average accuracy with this classifier was around 65% testing.

* **dogcatcnn.ipnyb** - This notebook is a fork of cnn_classifier_2.ipnyb which was used do test some visualization on the output.

* **keras_classifier.ipnyb** - Classifier written in Keras using a Tensorflow backend. This algorithm also uses image augmentation to increase the data sample. Average accuracy with this classifier was around 82% testing. 

* **cnn_classifier_3.ipnyb** - Classifier written in TF Learn using a Tensorflow backend. This algorithm also uses image augmentation and implements cross validation on the dataset. This classifier was also used to obtain predictions for the Kaggle testing sample. Average accuracy with this classifier was around 85% testing.

* **transfer_learning_1.ipnyb** - Transfer Learning implementation. This classifier uses the top convolution layers from Imagenet VGG16 Network, and adds bottom level Fully Connected Layers. The weights for the top layers are loaded from the pretrained network and for the bottom layers, bottom propagation would be used by freezing the weights for the top layers. Because of GPU restrictions, this classifier was not able to run on the validation data set. Average accuracy should be around 95%.

* **imagenet/imagenet_classifier.ipnyb** - Feature extractor for Imagenet. This notebook contains code for extracting features from the VGG16 model by feeding input images from the cat and dog dataset. Instead of classification, the final layer features are stored, which can later be trained using a simple supervised classifier.

### Executable Code

The Classifiers can be run individually to obtain results on the dataset. The recommended classifier to run is **cnn_classifier_3.ipynb**. The results from this classifier can also be visualized without running the classifier. In order to visualize them, use the following steps:

* Uncompress the archive **tensorboard.zip**
* Run the command `tensorboard --logdir tensorboard/`
* Open the following URL in a browser: [http://0.0.0.0:6006](0.0.0.0:6006)


### References

[fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models): This repository provided pre-trained imagenet models for Keras, which were used to extract features for the imagenet feature extractor.

[machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg): This repository provided pre-trained imagenet models for Tensorflow, which were used to write the Transfer Learning model. This model was not used verbatim in code, and modifications were made to the classifier to remove the existing bottom level layers and add new fully connected layers to work with this project.
