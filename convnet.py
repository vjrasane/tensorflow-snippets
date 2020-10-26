from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_data, test_data = train_data / 255.0, test_data / 255.0
# print (train_labels)
# IMG_INDEX = 1
# plt.figure()
# plt.imshow(train_data[IMG_INDEX])
# plt.grid(False)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

model = keras.Sequential([
  keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  keras.layers.MaxPooling2D((2,2)),
  keras.layers.Conv2D(64, (3, 3), activation='relu'),
  keras.layers.MaxPooling2D((2,2)),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(10)
])

print(model.summary())

model.compile(
  optimizer='adam',
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels) )

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(test_acc)