import os
import numpy as numpy
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
  'cats_vs_dogs',
  split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
  with_info=True,
  as_supervised=True
)

get_label_name = metadata.features['label'].int2str

IMG_SIZE = 160

def format_example(image, label):
  image = tf.cast(image, tf.float32) # cast each pixel to float32 value
  image = (image/ (255/2) ) - 1  # adjust each value to be 0-1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) # resize image
  return image, label

train_data = raw_train.map(format_example)
validation_data = raw_validation.map(format_example)
test_data = raw_test.map(format_example)

# for image, label in train_data.take(2):
#   plt.figure()
#   plt.imshow(image)
#   plt.title(get_label_name(label))
#   plt.show()

for img, label in raw_train.take(2):
  print ("Original shape: ", img.shape)

for img, label in train_data.take(2):
  print ("New shape: ", img.shape)


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation_data.batch(BATCH_SIZE)
test_batches = test_data.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model_file = "models/dogs_vs_cats.h5"
if not os.path.isfile(model_file):
  base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
  )

  base_model.trainable = False

  model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1)
  ])

  base_learning_rate = 0.0001
  model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  initial_loss, initial_acc = model.evaluate(validation_batches, steps = 20)
  print(initial_acc)

  result = model.fit(train_batches, epochs=3, validation_data=validation_batches)
  acc = result.history["accuracy"]
  print(acc)

  model.save(model_file)
else:
  model = keras.models.load_model(model_file)

test_loss, test_acc = model.evaluate(test_batches, steps=1)
print(test_acc)