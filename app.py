from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from datetime import datetime, timedelta
import time
from flask_caching import Cache
import json
import warnings

# Suppress Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.saving.saving_lib")

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('./saved_models/my_model.keras', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classes = ["bluetit", "jackdaw", "robin", "unknown_bird", "unknown_object"]


# Change logging to use memory buffer instead of file
# This prevents Five Server from detecting constant file changes
class MemoryHandler(logging.Handler):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_logs(self):
        return self.buffer


# Configure memory-based logging
memory_handler = MemoryHandler(capacity=100)
memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(memory_handler)

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# In-memory storage for prediction history
prediction_history = []

# Add allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def log_prediction(image_name, prediction, confidence, processing_time):
    logger.info(
        f"Prediction: {image_name} - Class: {prediction} - Confidence: {confidence:.2f} - Time: {processing_time:.2f}s")


def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_img


def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = enhance_contrast(img)
    img = sharpen_image(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


# Add validation metrics
class ModelMetrics:
    def __init__(self):
        self.predictions = []
        self.processing_times = []
        self.confidences = []

    def add_prediction(self, prediction, confidence, processing_time):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.processing_times.append(processing_time)

    def get_metrics(self):
        return {
            'total_predictions': len(self.predictions),
            'average_confidence': np.mean(self.confidences) if self.confidences else 0,
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'accuracy_last_hour': self.get_accuracy_last_hour() if hasattr(self, 'get_accuracy_last_hour') else 0
        }


metrics = ModelMetrics()


@app.route('/')
def home():
    return send_from_directory('', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Size validation
        file_bytes = file.read()
        if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({'error': 'File too large'}), 400

        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        max_prob = np.max(predictions)
        predicted_class = classes[np.argmax(predictions)]

        processing_time = time.time() - start_time
        log_prediction(file.filename, predicted_class, max_prob, processing_time)

        response = {
            'prediction': predicted_class,
            'confidence': float(max_prob),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }

        # Store prediction history in memory instead of file
        prediction_history.append(response)
        if len(prediction_history) > 50:  # Keep only last 50 predictions
            prediction_history.pop(0)

        # Add to metrics
        metrics.add_prediction(predicted_class, max_prob, processing_time)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        # Use in-memory prediction history instead of file
        if not prediction_history:
            logger.info("No prediction history available")
            return jsonify({
                'total_predictions': 0,
                'average_confidence': 0,
                'average_processing_time': 0,
                'history': []
            })

        # Calculate stats
        total_predictions = len(prediction_history)
        avg_confidence = sum(float(p.get('confidence', 0)) for p in prediction_history) / total_predictions
        avg_processing_time = sum(float(p.get('processing_time', 0)) for p in prediction_history) / total_predictions

        response_data = {
            'total_predictions': total_predictions,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'history': prediction_history[-10:]  # Return last 10 predictions
        }

        logger.info(f"Stats calculated: {total_predictions} predictions")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
@cache.cached(timeout=300)  # Cache for 5 minutes
def get_metrics():
    return jsonify(metrics.get_metrics())


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'last_prediction_time': metrics.processing_times[-1] if metrics.processing_times else None
    })


@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify({
        'logs': memory_handler.get_logs()
    })

    
if __name__ == '__main__':
    logger.info("Flask server starting...")
    # Run the app on localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)