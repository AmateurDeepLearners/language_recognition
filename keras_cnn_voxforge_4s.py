import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
#For plotting only:
from keras.utils import plot_model
# dimensions of our images.
img_width, img_height = 350,128

train_data_dir = 'data_augmentation/training'
validation_data_dir = 'data_augmentation/validation'
#Automatically find number of train and validatin samples:
nb_train_samples =  float(sum([len(files) for r, d, files in os.walk(train_data_dir)]))
nb_validation_samples = float(sum([len(files) for r, d, files in os.walk(validation_data_dir)]))
epochs = 100
batch_size = 500
n_classes = 5

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

#print K.image_data_format()
def baseline_model():
    model = Sequential()

    model.add(Conv2D(16, (5, 5), input_shape=input_shape, padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
	
    opt = optimizers.Adam()
    model.compile(loss = "categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

    return model

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    shear_range=0,#.2,
    zoom_range=0,#0.2,
    horizontal_flip=False)#True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()#rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

#Create Checkpoint Callback that saves the modal after each epoch
#Only saves the model when it has improved 
filename = 'voxforge_weights-5-Sprachen.{epoch:02d}-{val_acc:.2f}.hdf5'
callback = ModelCheckpoint(filename,
     monitor='val_acc',
    verbose=1, save_best_only=True, 
    mode='max', period=1)

callbacks_list = [callback]

#Train the network
model = baseline_model()#load_model("voxforge_weights-5-Sprachen.182-0.75.hdf5")
network = model.fit_generator(
    train_generator,
    steps_per_epoch=int(nb_train_samples) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks = callbacks_list)

#Save the network history
print("saving network history to keras_cnn_5_Sprachen_8kh_voxforge_new.txt ...")
np.savetxt("keras_cnn_5_Sprachen_8kh_voxforge_new.txt", np.transpose([network.history["loss"], network.history["val_loss"], network.history["acc"], network.history["val_acc"]]))
