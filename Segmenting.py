import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from glob import glob
from PIL import Image
"""
- `os`, `glob`, and `PIL.Image` are used for file handling and reading images.
- `numpy` for numerical operations.
- `matplotlib.pyplot` for visualization.
- `tensorflow` for building and training the deep learning model.
"""


# Path to your dataset
IMAGE_DIR = '/content/Untitled Folder/dataset/unmasked'
MASK_DIR = '/content/Untitled Folder/dataset/masked'


# Parameters
IMG_SIZE = 128
BATCH_SIZE = 16
OUTPUT_CLASSES = 2  # Number of segmentation classes (2: background and foreground).


# Load and preprocess functions
"""This function takes the path to an image and its corresponding mask, reads them,
 preprocesses them, and returns them in a form suitable for training.

"""
def process_path(img_path, mask_path):
    img = tf.io.read_file(img_path)  # reads image and results ia raw byte code
    img = tf.image.decode_jpeg(img, channels=3)# Decodes the raw bytes into a 3-channel RGB image.
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])# Resizes the image to IMG_SIZE x IMG_SIZE pixels (e.g., 128x128).
    #ensuring uniform
    img = tf.cast(img, tf.float32) / 255.0#Converts pixels (0-255) to floats (0.0-1.0).train fast &better

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')

    # Ensure mask pixel values are 0 or 1:
    mask = tf.where(mask > 0, tf.ones_like(mask), tf.zeros_like(mask))
    # If your original mask has values other than 0 and 255 for classes.
    # You might need a different condition to map to 0 and 1
    # E.g., mask = tf.where(mask == 255, tf.ones_like(mask), tf.zeros_like(mask))

    return img, mask



# Create tf.data.Dataset
def load_dataset(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, '*')))
    mask_paths = sorted(glob(os.path.join(mask_dir, '*')))
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Split into train/test
all_dataset = load_dataset(IMAGE_DIR, MASK_DIR)
DATASET_SIZE = len(list(all_dataset))
train_size = int(0.8 * DATASET_SIZE)

train_dataset = all_dataset.take(train_size).cache().shuffle(100).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
test_dataset = all_dataset.skip(train_size).batch(BATCH_SIZE)

"""`cache()`: speeds up training.
- `shuffle(100)`: randomize batch.
- `repeat()`: repeats dataset for multiple epochs.
- `prefetch()`: prepares next batch while current one is being used."""



# Display sample
def display(display_list):
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Sample image
for image, mask in all_dataset.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

# U-Net model (same as your code)
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    x = inputs
    skips = []
""" Input shape: 128x128 RGB image.
- `skips`: stores encoder outputs for skip connections."""

    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        skips.append(x)

"""This loop runs 4 times, and in each iteration it:

Adds a convolutional layer with an increasing number of filters.

Applies max pooling to reduce the image size (downsampling).

Saves the result to a list called skips for later use in skip connections."""



    x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x)
#- Bottleneck: deepest part of the U-Net.

    for filters, skip in zip([512, 256, 128, 64][::-1], reversed(skips)):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
        # Crop the skip connection to match the shape of x
        skip = layers.CenterCrop(x.shape[1], x.shape[2])(skip)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(output_channels, 1, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

"""This loop iterates over decoder filter sizes and corresponding skip connections from the encoder.

[512, 256, 128, 64][::-1] → becomes [64, 128, 256, 512], which matches the reversed skips.

reversed(skips) gives us the stored encoder outputs in reverse order (from deep to shallow).

zip(...) pairs each decoder filter value with its corresponding encoder skip connection."""



# Compile and train
model = unet_model(OUTPUT_CLASSES)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training for 15 epochs
EPOCHS = 15
STEPS_PER_EPOCH = train_size // BATCH_SIZE
VALIDATION_STEPS = (DATASET_SIZE - train_size) // BATCH_SIZE

model.fit(train_dataset, epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=test_dataset,
          validation_steps=VALIDATION_STEPS)


# Predict and visualize
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

    #Converts softmax predictions into class labels.

    #Visualizes predicted masks along with input image and ground truth.

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image[tf.newaxis, ...])
            display([image, mask, create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()
