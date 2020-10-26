from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow as tf

CSV_COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
SPECIES = ["setosa", "versicolor", "virginica"]

train_path = tf.keras.utils.get_file(
  "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
test_path = tf.keras.utils.get_file(
  "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

train_data = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test_data = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# print(train_data.head())

# train_data.species.value_counts().plot(kind='barh')
# plt.show()

# print(train_data.describe())

train_labels = train_data.pop("species")
test_labels = test_data.pop("species")

print(train_data.head())
print(train_data.shape)

def input_fn(data, labels, training=True, batch_size=256):
  dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
  if training:
    dataset = dataset.shuffle(1000).repeat()
  return dataset.batch(batch_size)

feature_columns = []
for key in train_data.keys():
  feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[30, 10],
  n_classes=3
)

classifier.train(
  input_fn=lambda: input_fn(train_data, train_labels, training=True),
  steps=5000
)

result = classifier.evaluate(
  input_fn=lambda: input_fn(test_data, test_labels, training=False)
)

print(result)

def input_fn(data, batch_size=256):
  return tf.data.Dataset.from_tensor_slices(dict(data)).batch(batch_size)

predict = {
  "sepal_length" : [5.1, 5.9, 6.9],
  "sepal_width" : [3.3, 3.0, 3.1],
  "petal_length" : [1.7, 4.2, 5.4],
  "petal_width" : [0.5, 1.5, 2.1],
}

# print ("Please input numeric values as prompted")
# for feature in train_data.keys():
#   valid = False
#   while not valid:
#     val = input(feature + ": ")
#     if val.isdigit():
#       valid = True
#   predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
  class_id = pred_dict['class_ids'][0]
  probability = pred_dict['probabilities'][class_id]
  print ('Prediction is "{}" ({:.1f}%)'.format(
    SPECIES[class_id], 100 * probability
  ))
