from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib


import tensorflow as tf

dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")

y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

#print(dftrain["age"].loc[0], y_train.loc[0])
#print(dftrain.head())
#print(dftrain.describe())
#print(dftrain.shape)

# dftrain.age.hist(bins=20)
# plt.show()

# dftrain.sex.value_counts().plot(kind='barh')
# plt.show()

# dftrain['class'].value_counts().plot(kind='barh')
# plt.show()

# pd.concat([dftrain, y_train], axis=1
# ).groupby("sex").survived.mean().plot(kind='barh').set_xlabel('% survived')
# plt.show()

CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
NUMERIC_COLUMNS = ["age", "fare"]
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices( (dict(data_df), label_df) )
    if shuffle:
      ds = ds.shuffle(1000)
    return ds.batch(batch_size).repeat(num_epochs)
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result)

predictions = list(linear_est.predict (eval_input_fn))
print(predictions[0]['probabilities'])