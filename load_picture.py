import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Specify the number of images you want to extract
num_images_to_extract = 20  # Change this number as needed

# Define the directory where you want to save the PNG files
output_directory = "mnist_images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load the MNIST dataset using TensorFlow
(x_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Randomly select `num_images_to_extract` images from the dataset
random_indices = np.random.choice(len(x_train), num_images_to_extract, replace=False)

# Iterate through the selected indices and save the images as PNG files
for idx, image_idx in enumerate(random_indices):
    image = x_train[image_idx]
    label = y_train[image_idx]
    image = Image.fromarray(image)
    image_path = os.path.join(output_directory, f"mnist_image_{idx}_label_{label}.png")
    image.save(image_path)
    print(
        f"Saved image {idx + 1}/{num_images_to_extract} with label {label} to {image_path}"
    )

print("Images saved successfully.")
