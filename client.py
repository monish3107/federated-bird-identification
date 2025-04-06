# client.py

import argparse
import flwr as fl
import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

server_address = "localhost:8000"
classes = ["bluetit", "jackdaw", "robin", "unknown_bird", "unknown_object"]
class_labels = {cls: i for i, cls in enumerate(classes)}
number_of_classes = len(classes)
IMAGE_SIZE = (160, 160)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, training_images, training_labels, test_images, test_labels):
        self.model = model
        self.training_images, self.training_labels = training_images, training_labels
        self.test_images, self.test_labels = test_images, test_labels

    def get_properties(self, config):
        raise Exception("Not implemented")

    def get_parameters(self, config):
        logging.info("Retrieving model parameters...")
        return self.model.get_weights()

    def fit(self, parameters, config):
        logging.info("Starting fit method...")
        logging.info(
            f"Training on {len(self.training_images)} samples with batch size {config['batch_size']} for {config['local_epochs']} epochs.")

        self.model.set_weights(parameters)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        try:
            history = self.model.fit(
                self.training_images,
                self.training_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                verbose=1  # Set verbose=1 to show progress bars for each epoch
            )

            parameters_prime = self.model.get_weights()
            num_examples_train = len(self.training_images)
            results = {
                "loss": history.history["loss"][-1],
                "accuracy": history.history["accuracy"][-1],
                "val_loss": history.history["val_loss"][-1],
                "val_accuracy": history.history["val_accuracy"][-1],
            }
            logging.info(f"Fit method completed successfully. Results: {results}")
            return parameters_prime, num_examples_train, results
        except Exception as e:
            logging.error(f"Error in fit method: {e}")
            raise

    def evaluate(self, parameters, config):
        logging.info("Starting evaluate method...")
        logging.info(f"Evaluating on {len(self.test_images)} samples.")

        try:
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.test_images, self.test_labels,
                                                 verbose=1)  # Set verbose=1 to show progress
            num_examples_test = len(self.test_images)
            logging.info(f"Evaluate method completed successfully. Loss: {loss}, Accuracy: {accuracy}")
            return loss, num_examples_test, {"accuracy": accuracy}
        except Exception as e:
            logging.error(f"Error in evaluate method: {e}")
            raise


def main() -> None:
    client_argumentparser = argparse.ArgumentParser()
    client_argumentparser.add_argument(
        '--client_number', dest='client_number', type=int,
        required=True,
        help='Used to load the dataset for the client')
    args = client_argumentparser.parse_args()
    client_number = args.client_number
    print(f"Client {client_number} has been connected!")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        pooling='avg'
    )
    base_model.trainable = False

    # Data augmentation layers as part of the model
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(160, 160, 3)),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.Rescaling(1. / 255)  # Rescale pixel values to [0-1]
    ])

    model = tf.keras.Sequential([
        data_augmentation,  # Add data augmentation as the first layer
        base_model,
        tf.keras.layers.Flatten(),  # Flatten after base model
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    try:
        logging.info("Loading dataset...")
        (training_images, training_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(
            client_number)
        logging.info(
            f"Dataset loaded. Training samples: {len(training_images)}, Validation samples: {len(val_images)}, Test samples: {len(test_images)}")

        training_images, training_labels = shuffle(training_images, training_labels, random_state=25)

        client = CifarClient(model, training_images, training_labels, test_images, test_labels)
        logging.info("Connecting to the server...")
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except Exception as e:
        logging.error(f"Error in main method: {e}")


def load_dataset(client_number):
    directory = f"datasets/dataset_client{client_number}"
    sub_directories = ["test", "train"]
    loaded_dataset = {}

    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)
        images = []
        labels = []

        print(f"Client dataset loading {sub_directory}")

        for folder in os.listdir(path):
            if folder not in class_labels:
                continue
            label = class_labels[folder]

            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(path, folder, file)
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Warning: Unable to read image {img_path}")
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, IMAGE_SIZE)
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    logging.error(f"Could not load image {img_path}: {e}")

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        loaded_dataset[sub_directory] = (images, labels)

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        loaded_dataset["train"][0], loaded_dataset["train"][1], test_size=0.2, random_state=42
    )

    # Get the test data
    test_images, test_labels = loaded_dataset["test"]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


if __name__ == "__main__":
    main()