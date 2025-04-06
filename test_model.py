import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

classes = ["bluetit", "jackdaw", "robin"]
class_labels = {cls: i for i, cls in enumerate(classes)}
number_of_classes = len(classes)
IMAGE_SIZE = (160, 160)

def load_model_safe():
    """Safely load the model with error handling"""
    model_paths = [
        'saved_models/my_model.keras',  # New .keras format
        'saved_models/my_model.h5',     # Legacy H5 format
        'saved_models/final_model'      # Try legacy format as last resort
    ]
    
    for path in model_paths:
        try:
            logging.info(f"Attempting to load model from: {path}")
            if os.path.exists(path):
                return tf.keras.models.load_model(path)
        except Exception as e:
            logging.warning(f"Failed to load model from {path}: {e}")
            continue
    
    raise ValueError("Could not load model from any known location")

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype("float32") / 255.0
            images.append(img)
    return images

def evaluate_model(model, test_data, test_labels, classes):
    if len(test_data) == 0:
        logging.error("No test data provided")
        return
        
    try:
        predictions = model.predict(test_data)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        # Generate classification report
        report = classification_report(test_labels, pred_classes, target_names=classes)
        print("Classification Report:\n", report)
        
        # Plot confusion matrix
        cm = confusion_matrix(test_labels, pred_classes)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('static/confusion_matrix.png')
        plt.close()
        
        return report, cm
        
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def main():
    try:
        # Load model
        model = load_model_safe()
        logging.info("Model loaded successfully")
        
        total_accuracy = 0.0
        total_folders = 0
        all_images = []
        all_labels = []
        
        # Test directory path
        test_dir = "datasets/dataset_test"
        if not os.path.exists(test_dir):
            raise ValueError(f"Test directory not found: {test_dir}")
            
        for folder_name in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            if folder_name not in class_labels:
                logging.warning(f"Skipping unknown class folder: {folder_name}")
                continue
                
            true_label = folder_name
            logging.info(f"Testing model on: {true_label}")
            
            images = load_images_from_folder(folder_path)
            if not images:
                logging.warning(f"No valid images found in {folder_name}")
                continue
                
            all_images.extend(images)
            all_labels.extend([class_labels[true_label]] * len(images))
            
            predictions = model.predict(np.array(images), verbose=0)
            correct_predictions = sum(1 for pred in predictions.argmax(axis=1) 
                                   if classes[pred] == true_label)
            accuracy = correct_predictions / len(images)
            total_accuracy += accuracy
            total_folders += 1
            
            logging.info(f"Accuracy for {true_label}: {accuracy:.4f}")
        
        if total_folders > 0:
            average_accuracy = total_accuracy / total_folders
            logging.info(f"Average accuracy: {average_accuracy:.4f}")
            
            # Evaluate on all data
            evaluate_model(model, np.array(all_images), np.array(all_labels), classes)
        else:
            logging.warning("No valid test folders found")
            
    except Exception as e:
        logging.error(f"Test execution failed: {e}")
        raise

if __name__ == '__main__':
    main()