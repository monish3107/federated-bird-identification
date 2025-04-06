# Bird Classification with Federated Learning

A robust bird classification system utilizing Federated Learning to train a machine learning model across distributed clients. The system includes a Flask web application for real-time bird species classification.

## Overview

This project implements a complete machine learning pipeline:

1. **Federated Learning**: Train a model across multiple distributed clients while keeping data private
2. **Image Classification**: Identify bird species (bluetit, jackdaw, robin) and detect unknown birds/objects
3. **Web Application**: Web interface for uploading images and getting predictions

## Features

- **Federated Learning Architecture**: Uses Flower (flwr) to coordinate model training across clients
- **Pre-trained Model**: Leverages MobileNetV2 as feature extractor
- **Real-time Classification**: Flask API for bird image classification
- **Image Enhancement**: Preprocessing pipeline with contrast enhancement and sharpening
- **Monitoring**: Performance metrics, logs, and prediction history

## System Architecture

### Server Component
- Coordinates federated learning process
- Aggregates model updates from clients
- Evaluates global model performance
- Saves trained model in multiple formats

### Client Component
- Trains on local dataset
- Implements data augmentation
- Sends model updates to server
- Maintains data privacy

### Web Application
- Handles image uploads
- Preprocesses images
- Provides predictions with confidence scores
- Tracks performance metrics

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Flask
- OpenCV
- Flower (flwr)
- NumPy
- Pandas
- Scikit-learn

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bird-classification-fl.git
cd bird-classification-fl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare dataset directories:
```
datasets/
├── dataset_server/
│   ├── train/
│   └── test/
├── dataset_client1/
│   ├── train/
│   └── test/
├── dataset_client2/
│   ├── train/
│   └── test/
└── dataset_test/
```

## Usage

### Start the Federated Learning Server
```bash
python server.py
```

### Start Client Training (run in separate terminals)
```bash
python client.py --client_number 1
python client.py --client_number 2
```

### Run Web Application
```bash
python app.py
```
The web application will be available at http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Submit an image for classification |
| `/stats` | GET | Get prediction statistics |
| `/metrics` | GET | Get model performance metrics |
| `/health` | GET | Check application health |
| `/logs` | GET | View server logs |

### Example: Using the Prediction API

```python
import requests

url = "http://localhost:5000/predict"
files = {"file": open("path/to/bird_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**:
  - Dense layer (128 neurons, ReLU activation)
  - Dropout layer (0.2)
  - Output layer (5 neurons, Softmax activation)
- **Input Size**: 160x160x3 pixels (RGB)
- **Classes**: bluetit, jackdaw, robin, unknown_bird, unknown_object

## Performance Monitoring

The system includes comprehensive monitoring:

- Prediction confidence tracking
- Processing time analysis
- Model accuracy evaluation
- Request logging


## Acknowledgments

- [Flower](https://flower.dev/) - Federated Learning framework
- [TensorFlow](https://www.tensorflow.org/) - Machine learning framework
- [Flask](https://flask.palletsprojects.com/) - Web application framework
