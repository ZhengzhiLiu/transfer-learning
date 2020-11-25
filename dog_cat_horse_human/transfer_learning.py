import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(directory):
    images = []
    labels = []
    label_ = {"cats":0, "dogs":1, "horses":2,"Humans":3}
    for cwd, dir, files in os.walk(directory):
        if not dir:
            for file in files:
                img = load_img(os.path.join(cwd, file), target_size=(224, 224))
                img = img_to_array(img) / 255.
                img = np.expand_dims(img, axis=0)
                images.append(img)
            labels.extend([label_[os.path.basename(cwd)]] * len(files))
    images = np.concatenate(images, axis=0)
    labels = np.array(labels)
    return images, labels


def show_samples(images, labels, nrow=5, ncol=5):
    _label = { 0:"cats", 1:"dogs", 2:"horses", 3:"Humans"}
    indices = np.random.randint(0, images.shape[0], ncol * nrow)
    for i in range(ncol * nrow):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(images[indices[i]])
        plt.title(_label[labels[indices[i]]])
        plt.axis("off")
    plt.tight_layout()


images, labels = load_data("./data")
X_train, X_test, y_train, y_test = train_test_split(images, labels )
show_samples(images, labels)
# np.unique(labels,return_coten unts=True)


epochs = 20

## fully connected layer
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
for layer in vgg_model.layers:
    model.add(layer)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='softmax'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)
model.evaluate(X_test, y_test)
plt.figure()
plt.plot(history.history["accuracy"], "r*--", label="accuracy")
plt.plot(history.history["val_accuracy"], "b.-", label="val_accuracy")
plt.title("Learning curve - fc")
plt.legend()
plt.show()
model.save("vgg_binary_classifer.h5")
predictions = model.predict(X_test).argmax(axis=-1)
print(classification_report(y_test,predictions))

## Global average pooling
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
for layer in vgg_model.layers:
    model.add(layer)
# model.get_layer("block5_conv3").trainable = True
model.add(GlobalAveragePooling2D())
model.add(Dense(4, activation='sigmoid'))
model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)
model.evaluate(X_test, y_test)
plt.figure()
plt.plot(history.history["accuracy"], "r*--", label="accuracy")
plt.plot(history.history["val_accuracy"], "b.-", label="val_accuracy")
plt.title("Learning curve - global average pooling")
plt.legend()
plt.show()
model.save("vgg_gobal_pooling.h5")
