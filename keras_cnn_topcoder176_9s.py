import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
#For plotting only:
from keras.utils import plot_model

# dimensions of our images.
img_width, img_height = 768,256

#Directories of the training and the validation folders
train_data_dir = '/scratch2/mklingn_Data_neural/data/training'
validation_data_dir = '/scratch2/mklingn_Data_neural/data/validation'

#Automatically find number of train and validatin samples:
nb_train_samples =  float(sum([len(files) for r, d, files in os.walk(train_data_dir)]))
nb_validation_samples = float(sum([len(files) for r, d, files in os.walk(validation_data_dir)]))

#Define number of epochs, batch size and number of classes for classification
epochs = 50
batch_size = 100
n_classes = 176

#Preparing the image format in the right way
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

#Defining the model used
def baseline_model():
    """By defining a Sequantial model one can define each layer. Each Layer 
    consists of a Convolutional Layer, an Activation function, 
    a Max-Pooling-Layer and Batch Normalization to prevent Overfitting"""
    model = Sequential()
    
    model.add(Conv2D(16, (7, 7), input_shape=input_shape, padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=3, padding='same'))
    model.add(BatchNormalization())


    model.add(Conv2D(32, (5, 5), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=3, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())
    
    """In the end one adds a fully connected one-dimensional Layer and a 
    Logic Layer with Softmax-Activation function to generate "Probabilities" for
    the classes, the class with highest probability is chosen to by the 
    one the network chooses to be correct"""

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    
    """The optimizer is Stochastic Gradient Descent with momentum, where one 
    has to choose the Learning Rate and the momentum variable"""
	
    opt = optimizers.SGD(lr=0.003, momentum=0.9)
    model.compile(loss = "categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])
    return model

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator()

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator()

# Training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

# Validation data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

#Create Checkpoint Callback that saves the modal after each epoch
#Only saves the model when it has improved 
filename = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
callback = ModelCheckpoint(filename,
     monitor='val_acc',
    verbose=1, save_best_only=True, 
    mode='max', period=1)

callbacks_list = [callback]

#Train the network
model = baseline_model()
network = model.fit_generator(
    train_generator,                                        #training data
    steps_per_epoch=nb_train_samples // batch_size,         #steps per epochs
    epochs=epochs,                                          #number of epochs
    validation_data=validation_generator,                   #validations data
    validation_steps=nb_validation_samples // batch_size,   #steps per validation
    callbacks = callbacks_list)

#Save the network history
print("saving network history to keras_cnn_176_history.txt ...")
np.savetxt("keras_cnn_176_history.txt", np.transpose(newtork.history), header=str(newtork.history.keys()))
