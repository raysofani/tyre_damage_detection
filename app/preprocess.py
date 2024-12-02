import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_images(data_dir, label):
    """
    Load images from a directory

    Args:
        data_dir (str): Path to image directory
        label (int): Label for images

    Returns:
        tuple of numpy arrays (images, labels)
    """
    images = []
    labels = []
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        try:
            # Read image
            img = cv2.imread(img_path)

            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize with interpolation
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(images), np.array(labels)


def prepare_data():
    """
    Prepare training and testing data

    Returns:
        Train and test datasets
    """
    # Paths to good and defective tire images
    good_dir = r"F:\dataset\Good"
    defective_dir = r"F:\dataset\Defective"

    # Load images and labels
    good_images, good_labels = load_images(good_dir, 0)
    defective_images, defective_labels = load_images(defective_dir, 1)

    # Combine data
    X = np.concatenate((good_images, defective_images))
    y = np.concatenate((good_labels, defective_labels))

    # Normalize images
    X = X.astype('float32') / 255.0

    # Split data
    return train_test_split(X, y, test_size=0.2, random_state=42)