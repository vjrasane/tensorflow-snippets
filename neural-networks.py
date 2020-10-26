from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

print(train_data.shape)
print(train_data[0][23][23])
print(train_labels[:10])

class_names = ["tshirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# plt.figure()
# plt.imshow(train_data[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_data = train_data / 255.0
test_data = test_data / 255.0

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)),
  keras.layers.Dense(128, activation='relu'), # rectified linear unit
  keras.layers.Dense(10, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.fit(train_data, train_labels, epochs=8)
train_loss, train_acc = model.evaluate(train_data, train_labels, verbose=1)
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=1)

print('Train accuracy:', train_acc)
print('Test accuracy:', test_acc)

def predict(model, image, correct_label):
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]
  show_image(image, class_names[correct_label], predicted_class)

def show_image(image, label, guess):
  plt.figure()
  plt.imshow(image, cmap=plt.cm.binary)
  plt.title("Expected: " + label + ", Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()

predict(model, test_data[10], test_labels[10])
