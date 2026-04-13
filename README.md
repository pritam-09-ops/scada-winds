# SCADA Wind Power Prediction ML Pipeline

This repository contains the code and models for predicting wind power generation using machine learning techniques, specifically focusing on XGBoost and LSTM models. The goal is to develop an efficient and accurate prediction system that can help optimize wind power generation.

## Project Structure

- **data/**: Directory for raw and processed data.
- **features/**: Directory containing scripts for feature engineering.
- **models/**: Directory where trained models and model scripts will be stored.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and experimentation.
- **src/**: Source code for model training and prediction.
- **configs/**: Configuration files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pritam-09-ops/scada-winds.git
   cd scada-winds
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data and place it in the `data/` directory.
2. Run the feature engineering scripts located in `features/`.
3. Train models using scripts in the `src/` directory.
4. Optimize model parameters using Optuna.

## Author

- Pritam

## License

This project is licensed under the MIT License.