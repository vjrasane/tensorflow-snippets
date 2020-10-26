from tensorflow import keras
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()

train_data, test_data = train_data / 255.0, test_data / 255.0

datagen = keras.preprocessing.image.ImageDataGenerator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

test_img = train_data[14]
img = keras.preprocessing.image.img_to_array(test_img)
img = img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, save_prefix="test", save_format="jpeg"):
  plt.figure(i)
  plot = plt.imshow(keras.preprocessing.image.img_to_array(batch[0]))
  i += 1
  if i > 4:
    break

plt.show()