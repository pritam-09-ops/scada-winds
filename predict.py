import xgboost as xgb
import numpy as np
import pandas as pd
from keras.models import load_model

# Load the trained models
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model.json')  # Change the path if necessary
lstm_model = load_model('lstm_model.h5')  # Change the path if necessary

# Function to preprocess the input data

def preprocess_data(data):
    # Implement your preprocessing logic here
    # Example: scaling, reshaping, etc.
    return processed_data

# Function to make predictions

def make_predictions(data):
    processed_data = preprocess_data(data)
    xgb_preds = xgb_model.predict(xgb.DMatrix(processed_data))
    lstm_preds = lstm_model.predict(processed_data.reshape((processed_data.shape[0], processed_data.shape[1], 1)))
    
    # Ensemble predictions (simple average or weighted average)
    final_predictions = (xgb_preds + lstm_preds.flatten()) / 2
    return final_predictions

# Example usage
if __name__ == '__main__':
    # Load new wind power data
    new_data = pd.read_csv('new_wind_data.csv')  # Change the path if necessary
    predictions = make_predictions(new_data)
    print(predictions)