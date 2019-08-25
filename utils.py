from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Mapping dict. Maps a label from the CSV to a int valued class.
emotions_to_classes_dict = {'neutral': 0,
                            'happiness': 1,
                            'surprise': 2,
                            'anger': 3,
                            'disgust': 4,
                            'sadness': 5,
                            'fear': 6,
                            'contempt': 7}

# Mapping dict. Maps a int valued class to a emotions. Basically the inverse of the emotion_to_classes_dict.
classes_to_emotions_dict = {v: k for k, v in emotions_to_classes_dict.items()}


def print_info(msg: str):
    """Wrap the message so that is easier to see that this message is informational"""
    info_str = f'[INFO] {str(msg)}'
    print(info_str)


def view_label_distribution(labels: np.ndarray, title: str = ''):
    """
    Prints the distribution between the different classes in a dataset.

    :param labels: Array containing the labels.
    :param title: Title of the plot.
    """
    unique_classes, unique_count = np.unique(labels, return_counts=True)
    unique_emotions = [classes_to_emotions_dict[unique_class] for unique_class in unique_classes]

    plt.bar(np.arange(len(unique_emotions)), unique_count, align='center')
    plt.xticks(np.arange(len(unique_emotions)), unique_emotions)
    plt.ylabel('Emotion count')
    plt.title(title)

    plt.show()


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate the weights of each label.

    :param labels: Array containing the labels
    :return: Dict containing the weights. Classes are keys.
    """
    # Count the unique classes.
    unique_classes, unique_count = np.unique(labels, return_counts=True)

    # Use the most common label as benchmark for weights.
    most_normal_label = np.max(unique_count)

    weights = {}
    for i, unique_class in enumerate(unique_classes):
        unique_class_dict_index = unique_class
        weight = most_normal_label / unique_count[i]
        weights[unique_class_dict_index] = weight

    return weights
