import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_history_file():
    """Create prediction history file if it doesn't exist"""
    if not os.path.exists('prediction_history.json'):
        with open('prediction_history.json', 'w') as f:
            json.dump([], f)
        logging.info("Created new prediction_history.json file")

def plot_model_metrics():
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        logging.info("Created static directory")

    initialize_history_file()

    try:
        with open('prediction_history.json', 'r') as f:
            data = json.load(f)
            
        if not data:
            logging.warning("No prediction history data available yet")
            return
            
        history = pd.DataFrame(data)
        
        # Plot confidence over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=history, x='timestamp', y='confidence')
        plt.title('Model Confidence Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/confidence_trend.png')
        plt.close()
        
        # Plot processing time distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=history, x='processing_time')
        plt.title('Processing Time Distribution')
        plt.tight_layout()
        plt.savefig('static/processing_time_dist.png')
        plt.close()
        
        logging.info("Successfully generated metric plots")
        
    except Exception as e:
        logging.error(f"Error plotting metrics: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        plot_model_metrics()
    except Exception as e:
        logging.error(f"Failed to plot metrics: {str(e)}")
