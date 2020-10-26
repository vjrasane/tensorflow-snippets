

from tensorflow.keras.datasets import imdb
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

train_data = keras.preprocessing.sequence.pad_sequences(train_data, MAXLEN)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, MAXLEN)

model_file = "models/imdb_sentiments.h5"
if not os.path.isfile(model_file):
  model = keras.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, 32), # 32 dimensions of word embedding
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid")
  ])

  model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
  )

  history = model.fit(train_data, train_labels, epochs = 10, validation_split=0.2)
  results = model.evaluate(test_data, test_labels)
  print(results)

  model.save(model_file)
else:
  model = keras.models.load_model(model_file)

word_index = imdb.get_word_index()
def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return keras.preprocessing.sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was great and amazing and super good"
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
  text = ""
  for num in integers:
    if num != 0:
      text += reverse_word_index[num] + " "
  return text[:-1]

print (decode_integers(encoded))

def predict(text):
  encoded_text = encode_text(text)
  return model.predict(np.array([encoded_text]))

positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
print(predict(positive_review))

negative_review = "that movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
print(predict(negative_review))
