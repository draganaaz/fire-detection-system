import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image

import pathlib

batch_size = 1
img_height = 180
img_width = 180

base_dir = '\Training'
#validation_split = percentage of data used for validation
#differentiate data using training and validation as values for subset attribute
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  base_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  base_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#class names are created off subdirectories within the data folder
#sorted alphabetically
class_names = train_ds.class_names
print(class_names)


#TODO code used for visualizing data only.
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")


AUTOTUNE = tf.data.experimental.AUTOTUNE

#dataset.cache() keeps the images in memory after they're loaded off disk during the first epoch. 
#Dataset.prefetch() overlaps data preprocessing and model execution while training.

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#RGB values go from 0 to 255 which is too big for the neural network.
#Therefore, we'll rescale them to be between 0 and 1
# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (layers.experimental.preprocessing.Rescaling(1./255)(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image)) 

num_classes = 2


#Model consists of three convolution blocks with a max pool layer in each of them

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


#Code used for training a model
epochs=7
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 11
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#NOTE:Some images were commented out since they have .tif extension

#URLs for images with Smoke on them
smoke_url_1 = "https://drive.google.com/uc?id=1Xsvx7b_r8R3k-lSIWzvAJc-aUzknACZC&export=download"
smoke_url_2 = "https://drive.google.com/uc?id=1TsHOyTf_UtF7x8QTNjsq64zxuC3Pd5s_&export=download"
smoke_url_3 = "https://drive.google.com/uc?id=1nysY7Y5vuk45CtRz2Id-TYTc7gLuvtxj&export=download"
smoke_url_4 = "https://drive.google.com/uc?id=1QQGtABH5z8v7rEgONgVbATGnVWwzEdWE&export=download"
smoke_url_5 = "https://drive.google.com/uc?id=19WuN6vN1x0Kotka3I3DFY7sMNZlswnZo&export=download"
# smoke_url_6 = "https://drive.google.com/uc?id=1Sh-h8XuJEVHTVmUVhQ2BVvZLqr0T2LBK&export=download"
#Paths for images with Smoke on them
smoke_path_1 = tf.keras.utils.get_file('Smoke_1.jpg', origin=smoke_url_1) 
smoke_path_2 = tf.keras.utils.get_file('Smoke_2.png', origin=smoke_url_2)
smoke_path_3 = tf.keras.utils.get_file('Smoke_3.png', origin=smoke_url_3)
smoke_path_4 = tf.keras.utils.get_file('Smoke_4.jpg', origin=smoke_url_4)
#NOTE:smoke_path_5 leads to Smoke_5.jpg which our model will most likely
#classify as Image with No Smoke, due to lack of training data
#(Even naked eye can notice how this image contains less smoke than
#the training data usually had)
smoke_path_5 = tf.keras.utils.get_file('Smoke_5.jpg', origin=smoke_url_5)
# smoke_path_6 = tf.keras.utils.get_file('Smoke_6.tif', origin=smoke_url_6)

#URLs for images without Smoke on them
no_smoke_url_1 = "https://drive.google.com/uc?id=12Qy1YrSoF6C0JcZSyBZnGW7dbzb6huMR&export=download"
no_smoke_url_2 = "https://drive.google.com/uc?id=1LWWtHpp5R2u4XOFfYkJxeyryX-ufilua&export=download"
no_smoke_url_3 = "https://drive.google.com/uc?id=15UTsswbZKklodesXyKoQ5NnEmdtQzi62&export=download"
no_smoke_url_4 = "https://drive.google.com/uc?id=1cZs_Ymve3wLNlNPqxiUdtYFwkbkP6TEp&export=download"
no_smoke_url_5 = "https://drive.google.com/uc?id=1h46I2Ln2K0tCg2lCEchdV8_ZKjFjPHT0&export=download"
no_smoke_url_6 = "https://drive.google.com/uc?id=1_easyCwho3rzBHZDjToROcRM8OmoD2zk&export=download"

#Paths for images without Smoke on them
#NOTE:no_smoke_path_1 leads to No_Smoke_1.jpg which our model will most likely
#classify as Image with Smoke, due to lack of training data
#(Even naked eye can notice how this image contains more urban data than
#the training data usually had)
no_smoke_path_1 = tf.keras.utils.get_file('No_Smoke_1.jpg', origin=no_smoke_url_1)
no_smoke_path_2 = tf.keras.utils.get_file('No_Smoke_2.png', origin=no_smoke_url_2)
no_smoke_path_3 = tf.keras.utils.get_file('No_Smoke_3.png', origin=no_smoke_url_3)
# no_smoke_path_4 = tf.keras.utils.get_file('No_Smoke_4.tif', origin=no_smoke_url_4)
# no_smoke_path_5 = tf.keras.utils.get_file('No_Smoke_5.tif', origin=no_smoke_url_5)
no_smoke_path_6 = tf.keras.utils.get_file('No_Smoke_6.png', origin=no_smoke_url_6)


#Images with smoke
img_smoke_1 = keras.preprocessing.image.load_img(
    smoke_path_1, target_size=(img_height, img_width)
)

img_smoke_2 = keras.preprocessing.image.load_img(
    smoke_path_2, target_size=(img_height, img_width)
)

img_smoke_3 = keras.preprocessing.image.load_img(
    smoke_path_3, target_size=(img_height, img_width)
)

img_smoke_4 = keras.preprocessing.image.load_img(
    smoke_path_4, target_size=(img_height, img_width)
)

img_smoke_5 = keras.preprocessing.image.load_img(
    smoke_path_5, target_size=(img_height, img_width)
)

# img_smoke_6 = keras.preprocessing.image.load_img(
#     smoke_path_6, target_size=(img_height, img_width)
# )


#Images without smoke
img_no_smoke_1 = keras.preprocessing.image.load_img(
    no_smoke_path_1, target_size=(img_height, img_width)
)

img_no_smoke_2 = keras.preprocessing.image.load_img(
    no_smoke_path_2, target_size=(img_height, img_width)
)

img_no_smoke_3 = keras.preprocessing.image.load_img(
    no_smoke_path_3, target_size=(img_height, img_width)
)

# img_no_smoke_4 = keras.preprocessing.image.load_img(
#     no_smoke_path_4, target_size=(img_height, img_width)
# )

# img_no_smoke_5 = keras.preprocessing.image.load_img(
#     no_smoke_path_5, target_size=(img_height, img_width)
# )

img_no_smoke_6 = keras.preprocessing.image.load_img(
    no_smoke_path_6, target_size=(img_height, img_width)
)

#Images with smoke to array
img_smoke_1_array = keras.preprocessing.image.img_to_array(img_smoke_1)
img_smoke_1_array = tf.expand_dims(img_smoke_1_array, axis=0) 

img_smoke_2_array = keras.preprocessing.image.img_to_array(img_smoke_2)
img_smoke_2_array = tf.expand_dims(img_smoke_2_array, axis=0) 

img_smoke_3_array = keras.preprocessing.image.img_to_array(img_smoke_3)
img_smoke_3_array = tf.expand_dims(img_smoke_3_array, axis=0) 

img_smoke_4_array = keras.preprocessing.image.img_to_array(img_smoke_4)
img_smoke_4_array = tf.expand_dims(img_smoke_4_array, axis=0) 

img_smoke_5_array = keras.preprocessing.image.img_to_array(img_smoke_5)
img_smoke_5_array = tf.expand_dims(img_smoke_5_array, axis=0) 

# img_smoke_6_array = keras.preprocessing.image.img_to_array(img_smoke_6)
# img_smoke_6_array = tf.expand_dims(img_smoke_6_array, axis=0) 


#Images without smoke to array
img_no_smoke_1_array = keras.preprocessing.image.img_to_array(img_no_smoke_1)
img_no_smoke_1_array = tf.expand_dims(img_no_smoke_1_array, axis=0) 

img_no_smoke_2_array = keras.preprocessing.image.img_to_array(img_no_smoke_2)
img_no_smoke_2_array = tf.expand_dims(img_no_smoke_2_array, axis=0) 

img_no_smoke_3_array = keras.preprocessing.image.img_to_array(img_no_smoke_3)
img_no_smoke_3_array = tf.expand_dims(img_no_smoke_3_array, axis=0) 

# img_no_smoke_4_array = keras.preprocessing.image.img_to_array(img_no_smoke_4)
# img_no_smoke_4_array = tf.expand_dims(img_no_smoke_4_array, axis=0) 

# img_no_smoke_5_array = keras.preprocessing.image.img_to_array(img_no_smoke_5)
# img_no_smoke_5_array = tf.expand_dims(img_no_smoke_5_array, axis=0) 

img_no_smoke_6_array = keras.preprocessing.image.img_to_array(img_no_smoke_6)
img_no_smoke_6_array = tf.expand_dims(img_no_smoke_6_array, axis=0) 


#Predictions for images with smoke
predictions_for_smoke_1 = model.predict(img_smoke_1_array)
score_for_smoke_1 = tf.nn.softmax(predictions_for_smoke_1[0])

predictions_for_smoke_2 = model.predict(img_smoke_2_array)
score_for_smoke_2 = tf.nn.softmax(predictions_for_smoke_2[0])

predictions_for_smoke_3 = model.predict(img_smoke_3_array)
score_for_smoke_3 = tf.nn.softmax(predictions_for_smoke_3[0])

predictions_for_smoke_4 = model.predict(img_smoke_4_array)
score_for_smoke_4 = tf.nn.softmax(predictions_for_smoke_4[0])

predictions_for_smoke_5 = model.predict(img_smoke_5_array)
score_for_smoke_5 = tf.nn.softmax(predictions_for_smoke_5[0])

# predictions_for_smoke_6 = model.predict(img_smoke_6_array)
# score_for_smoke_6 = tf.nn.softmax(predictions_for_smoke_6[0])


#Predictions for images without smoke
predictions_for_no_smoke_1 = model.predict(img_no_smoke_1_array)
score_for_no_smoke_1 = tf.nn.softmax(predictions_for_no_smoke_1[0])

predictions_for_no_smoke_2 = model.predict(img_no_smoke_2_array)
score_for_no_smoke_2 = tf.nn.softmax(predictions_for_no_smoke_2[0])

predictions_for_no_smoke_3 = model.predict(img_no_smoke_3_array)
score_for_no_smoke_3 = tf.nn.softmax(predictions_for_no_smoke_3[0])

# predictions_for_no_smoke_4 = model.predict(img_no_smoke_4_array)
# score_for_no_smoke_4 = tf.nn.softmax(predictions_for_no_smoke_4[0])

# predictions_for_no_smoke_5 = model.predict(img_no_smoke_5_array)
# score_for_no_smoke_5 = tf.nn.softmax(predictions_for_no_smoke_5[0])

predictions_for_no_smoke_6 = model.predict(img_no_smoke_6_array)
score_for_no_smoke_6 = tf.nn.softmax(predictions_for_no_smoke_6[0])


#Printing the result of the predictions for images with smoke
print(
    "Prediction for the Smoke_1 with smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_smoke_1)], 100 * np.max(score_for_smoke_1))
)
print(
    "Prediction for the Smoke_2 with smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_smoke_2)], 100 * np.max(score_for_smoke_2))
)
print(
    "Prediction for the Smoke_3 with smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_smoke_3)], 100 * np.max(score_for_smoke_3))
)
print(
    "Prediction for the Smoke_4 with smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_smoke_4)], 100 * np.max(score_for_smoke_4))
)
print(
    "Prediction for the Smoke_5 with smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_smoke_5)], 100 * np.max(score_for_smoke_5))
)
# print(
#     "Prediction for the Smoke_6 with smoke: It most likely has {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score_for_smoke_6)], 100 * np.max(score_for_smoke_6))
# )

#Printing the results for the predictions for images without smoke
print(
    "Prediction for the No_Smoke_1 without smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_no_smoke_1)], 100 * np.max(score_for_no_smoke_1))
)
print(
    "Prediction for the No_Smoke_2 without smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_no_smoke_2)], 100 * np.max(score_for_no_smoke_2))
)
print(
    "Prediction for the No_Smoke_3 without smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_no_smoke_3)], 100 * np.max(score_for_no_smoke_3))
)
# print(
#     "Prediction for the No_Smoke_4 without smoke: It most likely has {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score_for_no_smoke_4)], 100 * np.max(score_for_no_smoke_4))
# )
# print(
#     "Prediction for the No_Smoke_5 without smoke: It most likely has {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score_for_no_smoke_5)], 100 * np.max(score_for_no_smoke_5))
# )
print(
    "Prediction for the No_Smoke_6 without smoke: It most likely has {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_for_no_smoke_6)], 100 * np.max(score_for_no_smoke_6))
)