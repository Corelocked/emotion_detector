import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ml_model import build_model 

fer_dataset = 'fer2013.csv'

df = pd.read_csv(fer_dataset) 

train_df = df[df['Usage'] == 'Training']
val_df = df[df['Usage'] == 'PublicTest']
test_df = df[df['Usage'] == 'PrivateTest']

def preprocess_pixels(pixels):
    pixels = np.array(pixels.split(), dtype='float32')
    pixels = pixels.reshape(48, 48)
    pixels = pixels / 255.0
    return pixels

train_pixels = np.array([preprocess_pixels(p) for p in train_df['pixels']])
val_pixels = np.array([preprocess_pixels(p) for p in val_df['pixels']])
test_pixels = np.array([preprocess_pixels(p) for p in test_df['pixels']])

train_labels = train_df['emotion'].values
val_labels = val_df['emotion'].values
test_labels = test_df['emotion'].values

train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=7)
val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes=7)
test_labels_cat = tf.keras.utils.to_categorical(test_labels, num_classes=7)

train_pixels = train_pixels.reshape(train_pixels.shape[0], 48, 48, 1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], 48, 48, 1)
test_pixels = test_pixels.reshape(test_pixels.shape[0], 48, 48, 1)

model = build_model()

history = model.fit(train_pixels, train_labels_cat, validation_data=(val_pixels, val_labels_cat), epochs=20, batch_size=64)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

model.save('emotion_model.h5')
