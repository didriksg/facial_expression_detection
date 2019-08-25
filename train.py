import os
import multiprocessing
from typing import List, Tuple

import cv2
import keras
import numpy as np
import pandas as pd

from keras_preprocessing.image import ImageDataGenerator
from keras import Model, Input
from keras.layers import MaxPooling2D, BatchNormalization, SeparableConv2D, Activation, Conv2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import layers
from keras.callbacks import ModelCheckpoint

from models import resnet
from utils import print_info, emotions_to_classes_dict, view_label_distribution, calculate_class_weights

# Paths.
TRAINING_DATA_CSV_PATH = 'legend.csv'
IMAGES_PATH = f'{os.path.dirname(os.path.realpath(__file__))}/images/'
MODEL_CHECKPOINT_PATH = 'models/checkpoints/'

# CSV column names.
CSV_IMAGE_COLUMN_NAME = 'image'
CSV_LABEL_COLUMN_NAME = 'emotion'

# Hyperparams
IMAGE_SIZE = 75
TRAINING_VAL_SPLIT = 0.2
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.00025


def load_dataset_from_csv(path: str) -> Tuple[List[str], List[int]]:
    """
    Extract data from a provided CSV

    :param path: Path to the CSV
    :return: Tuple containing the list of the images, and the label
    """
    df = pd.read_csv(path, delimiter=',')

    # Look for and use the 'image'
    if CSV_IMAGE_COLUMN_NAME in df.columns:
        image_paths = list(df['image'])
    else:
        raise KeyError(f'"{CSV_IMAGE_COLUMN_NAME}" is not a column in the provided CSV')

    if CSV_LABEL_COLUMN_NAME in df.columns:
        labels = list(df['emotion'])
        labels_mapped = [int(emotions_to_classes_dict[emotion.lower()]) for emotion in labels]
    else:
        raise KeyError(f'{CSV_LABEL_COLUMN_NAME}" is not a column in the provided CSV')

    return image_paths, labels_mapped


def get_training_val_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the loaded and ready-to-use training and validation data from a path to a CSV.

    :param path: Path to CSV containing all needed info to image paths and labels.
    :return: Tuple containing training images and labels, and validation images and labels.
    """
    # Extract dataset and split into training and validation.
    dataset = load_dataset_from_csv(path)

    # Extract images and labels into separate lists.
    image_paths, labels = dataset

    # Calculate an appropriate splitting point.
    split_point = int(len(image_paths) * TRAINING_VAL_SPLIT)

    # Split the dataset into training and validation.
    training_image_paths = image_paths[split_point:]
    training_labels = np.asarray(labels[split_point:])

    validation_image_paths = image_paths[:split_point]
    validation_labels = np.asarray(labels[:split_point])

    # Load the images from the provided paths.
    training_images_preprocessed = multiprocess_load_and_preprocess(training_image_paths)
    validation_images_preprocessed = multiprocess_load_and_preprocess(validation_image_paths)

    # Convert the list to a numpy list, as the model being trained needs it on this format.
    np_training_images = np.asarray(training_images_preprocessed)
    np_val_images = np.asarray(validation_images_preprocessed)

    return np_training_images, training_labels, np_val_images, validation_labels


def load_and_preprocess_image_from_path(img_path: np.ndarray, preprocess_image=True) -> np.ndarray:
    """
    Read an image from a provided path and preprocess the image.

    :param img_path: Path to image to be preprocessed.
    :param preprocess_image: Should the image be preprocessed. Should normally be set to true.
    :return: The preprocessed image.
    """

    # Get the absolute path from a relative path.
    img_path_complete_path = f'{IMAGES_PATH}/{img_path}'

    # Read the image on the given path.
    img = cv2.imread(img_path_complete_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image if specified
    if preprocess_image:
        img = preprocess(img)

    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a given image. Preprocessing consists of resizing and normalizing the image.

    :param img: Image to be preprocessed.
    :return: Preprocessed image.
    """
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize the image.
    img = np.multiply(img, 1 / 255)
    img = np.stack((img,) * 3, axis=-1)

    return img


def multiprocess_load_and_preprocess(images: List[str]) -> List[np.ndarray]:
    """
    Load images and preprocess them from a list of paths.

    :param images: List containing path to images.
    :return: Preprocessed images as np.ndarrays.
    """
    proc_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(proc_count) as p:
        data = p.map(load_and_preprocess_image_from_path, images)

        return data


def main():
    # Check that necessary paths exists, if not create them.
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print_info('Creating model checkpoint path...')
        os.makedirs(MODEL_CHECKPOINT_PATH)

    # Get the dataset from the CSV, load it and preprocess the data.
    print_info('Preparing training and validation data...')
    train_img, train_labels, val_img, val_labels = get_training_val_data(TRAINING_DATA_CSV_PATH)

    # Plot the distribution.
    # view_label_distribution(train_labels, title='Emotions in training set')
    # view_label_distribution(val_labels, title='Emotions in validation set')

    # Calculate class weights.
    print_info('Calculating class weights...')
    train_weights = calculate_class_weights(train_labels)

    # Load and compile the model with the correct input shape and number of classes.
    print_info('Loading model...')
    input_shape = train_img[0].shape
    num_of_classes = len(np.unique(train_labels))
    model = resnet(input_shape, num_of_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Calculate the number of steps taken during training.
    training_steps = len(train_img) / BATCH_SIZE

    # Add all callbacks in a list. Will be used during training.
    save_path = MODEL_CHECKPOINT_PATH+'weights_{epoch:03d}.hdf5'
    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [checkpoint]

    # Create an ImageDataGenerator. Eventual augmentations are done here.
    data_generator = ImageDataGenerator(rotation_range=20,
                                        horizontal_flip=True,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2)

    # Train the model.
    print_info('Starting training...')
    model.fit_generator(data_generator.flow(train_img, train_labels),
                        steps_per_epoch=training_steps, class_weight=train_weights, epochs=EPOCHS,
                        validation_data=(val_img, val_labels), shuffle=True, callbacks=callbacks,
                        use_multiprocessing=False, workers=multiprocessing.cpu_count()
                        )


if __name__ == '__main__':
    main()
