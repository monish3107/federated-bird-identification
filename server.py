# server.py

from typing import Dict, Optional, Tuple
import flwr as fl
import tensorflow as tf
import os
import cv2
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

server_address = "localhost:8000"

classes = ["bluetit", "jackdaw", "robin", "unknown_bird", "unknown_object"]
class_labels = {cls: i for i, cls in enumerate(classes)}
number_of_classes = len(classes)
IMAGE_SIZE = (160, 160)

federated_learning_counts = 2
local_client_epochs = 20
local_client_batch_size = 32  # Increased batch size

def main() -> None:
    start_time = time.time()
    logging.info("Starting server...")

    # Initialize the base model
    logging.info("Initializing base model...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet",
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base model layers

    # Build the model
    logging.info("Building the model...")
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Adding dropout
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])

    # Compile the model
    logging.info("Compiling the model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define the federated learning strategy
    logging.info("Setting up federated learning strategy...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start the Flower server
    logging.info("Starting Flower server...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federated_learning_counts),
        strategy=strategy
    )
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken for model compilation and training: {total_time} seconds")

def load_dataset():
    directory = "datasets/dataset_server"
    sub_directories = ["test", "train"]

    loaded_datasets = {}
    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)
        images = []
        labels = []

        logging.info(f"Server dataset loading {sub_directory}")

        for folder in os.listdir(path):
            if folder not in class_labels:
                continue
            label = class_labels[folder]
            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(path, folder, file)
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"Unable to read image {img_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32') / 255.0
        labels = np.array(labels, dtype='int32')
        loaded_datasets[sub_directory] = (images, labels)

    return loaded_datasets["train"], loaded_datasets["test"]

def get_evaluate_fn(model):
    (training_images, training_labels), (test_images, test_labels) = load_dataset()
    logging.info(f"[Server] Loaded dataset. Test samples: {len(test_images)}")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        logging.info(f"======= Server Round {server_round}/{federated_learning_counts} Evaluate() =======")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)  # Set verbose=1 to show progress
        logging.info(f"======= Server Round {server_round}/{federated_learning_counts} Accuracy : {accuracy} =======")

        if server_round == federated_learning_counts:
            logging.info("Saving updated model in multiple formats...")
            save_dir = os.path.join(os.getcwd(), 'saved_models')
            os.makedirs(save_dir, exist_ok=True)

            # Save in .keras format (preferred for TF 2.x)
            model.save(os.path.join(save_dir, 'my_model.keras'))

            # Save in .h5 format (legacy support)
            model.save(os.path.join(save_dir, 'my_model.h5'))

            # Save in SavedModel format
            model.save(os.path.join(save_dir, 'final_model'))

            test_updated_model(model)

        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    config = {
        "batch_size": local_client_batch_size,
        "local_epochs": local_client_epochs,
    }
    logging.info(f"Configuring fit for round {server_round}: {config}")
    return config

def evaluate_config(server_round: int):
    config = {"val_steps": 4}
    logging.info(f"Configuring evaluate for round {server_round}: {config}")
    return config

def test_updated_model(model):
    test_image_path = "datasets/dataset_test/robin/(121).jpg"
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        logging.error(f"Unable to read test image: {test_image_path}")
        return

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, IMAGE_SIZE)

    logging.info("Testing the final model on an image.....")
    image_test_result = model.predict(np.expand_dims(test_image, axis=0))
    logging.info(f"Prediction results: {image_test_result[0]}")

    highest_prediction_score_index = np.argmax(image_test_result[0])
    most_confident_class = classes[highest_prediction_score_index]
    logging.info(
        f"The model mostly predicted {most_confident_class} with a score/confidence of {image_test_result[0][highest_prediction_score_index]}")

if __name__ == "__main__":
    main()