import tensorflow as tf
model = tf.keras.applications.VGG19()

from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
model_new = Sequential()
for layer in model.layers[:-1]: # go through until last layer
  layer.trainable = False
  print(layer)
  model_new.add(layer)
model_new.add(Dense(3, activation='softmax'))

opt = Adam(learning_rate=0.01)
model_new.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

img_width = 224
img_height = 224
batch_size = 4
nb_train_samples = 96
nb_validation_samples = 24
epochs = 100

train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	'/content/gdrive/MyDrive/rice_disease_data/train',
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	'/content/gdrive/MyDrive/rice_disease_data/test',
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical')

model_new.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)