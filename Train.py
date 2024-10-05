from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

x_train = train_datagen.flow_from_directory("C:/Users/shres/Downloads/Project/Arrhythmia Detection/data/test", target_size=(64, 64), batch_size=32, class_mode="categorical")
x_test = test_datagen.flow_from_directory("C:/Users/shres/Downloads/Project/Arrhythmia Detection/data/test", target_size=(64, 64), batch_size=32, class_mode="categorical")

# Model building
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, kernel_initializer="random_uniform", activation="relu"))
model.add(Dense(units=128, kernel_initializer="random_uniform", activation="relu"))
model.add(Dense(units=128, kernel_initializer="random_uniform", activation="relu"))
model.add(Dense(units=128, kernel_initializer="random_uniform", activation="relu"))
model.add(Dense(units=128, kernel_initializer="random_uniform", activation="relu"))
model.add(Dense(units=6, kernel_initializer="random_uniform", activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit_generator(generator=x_train, steps_per_epoch=len(x_train), epochs=9, validation_data=x_test, validation_steps=len(x_test))

# Plotting the training and validation loss and accuracy
sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='train')
sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='valid')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(x=history.epoch, y=history.history['loss'], label='train')
sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='valid')
plt.title('Loss')
plt.tight_layout()

plt.savefig('epoch_history_dcnn.png')
plt.show()

# Saving the model
model.save('ECG.h5')
