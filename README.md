# Transformer Model for Time-Series Prediction

This repository contains a Python implementation of a Transformer-based model designed for time-series forecasting tasks. The model leverages both encoder and decoder structures to effectively capture sequential dependencies in data. Below, you will find details about the features, structure, and usage of the code.

## Key Features
- **Multi-head Attention Mechanism:** Utilizes scaled dot-product attention to capture global dependencies in time-series data.
- **Custom Data Processing:** Generates features based on historical counts, months, weeks, days, day-of-week, and LightGBM outputs.
- **Positional Encoding:** Adds temporal awareness to the Transformer by encoding position information.
- **Custom Schedulers and Optimizers:** Provides fine-grained control of the learning process using schedulers.
- **MAPE Evaluation:** Evaluates model performance using Mean Absolute Percentage Error.


