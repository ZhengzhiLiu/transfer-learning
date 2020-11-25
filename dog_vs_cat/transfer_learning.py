import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(directory):
    images = []
    labels = []
    for cwd, dir, files in os.walk(directory):
        if not dir:
            for file in files:
                img = load_img(os.path.join(cwd, file), target_size=(224, 224))
                img = img_to_array(img) / 255.
                img = np.expand_dims(img, axis=0)
                images.append(img)
            labels.append([os.path.basename(cwd)] * len(files))
    images = np.concatenate(images, axis=0)
    labels = [0 if x == "dogs" else 1 for x in sum(labels, [])]
    labels = np.array(labels)
    return images, labels


def show_samples(images, labels, nrow=5, ncol=5):
    indices = np.random.randint(0, images.shape[0], ncol * nrow)
    for i in range(ncol * nrow):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(images[indices[i]])
        label = "dog" if labels[indices[i]] == 0 else "cat"
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()


train_dir = "./data/train"
test_dir = "./data/test"
validation_dir = "./data/validation"
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)
validation_images, validation_labels = load_data(validation_dir)
show_samples(train_images, train_labels)

epochs = 20

## binary classifier
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
for layer in vgg_model.layers:
    model.add(layer)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss="binary_crossentropy", metrics=['accuracy'])
model.summary()
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(validation_images, validation_labels))
model.evaluate(test_images, test_labels)
plt.figure("Learning curve")
plt.plot(history.history["accuracy"], "r*--", label="accuracy")
plt.plot(history.history["val_accuracy"], "b.-", label="val_accuracy")
plt.legend()
plt.show()
model.save("vgg_binary_classifer.h5")

## Global average pooling
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
for layer in vgg_model.layers:
    model.add(layer)
# model.get_layer("block5_conv3").trainable = True
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss="binary_crossentropy", metrics=['accuracy'])
model.summary()
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(validation_images, validation_labels))
model.evaluate(test_images, test_labels)
plt.figure("Learning curve")
plt.plot(history.history["accuracy"], "r*--", label="accuracy")
plt.plot(history.history["val_accuracy"], "b.-", label="val_accuracy")
plt.legend()
plt.show()
model.save("vgg_gobal_pooling.h5")
