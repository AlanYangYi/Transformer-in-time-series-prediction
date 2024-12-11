# Transformer Model for Time-Series Prediction

This repository contains a Python implementation of a Transformer-based model designed for time-series forecasting tasks. The model leverages both encoder and decoder structures to effectively capture sequential dependencies in data. Below, you will find details about the features, structure, and usage of the code.

## Key Features
- **Multi-head Attention Mechanism:** Utilizes scaled dot-product attention to capture global dependencies in time-series data.
- **Custom Data Processing:** Generates features based on historical counts, months, weeks, days, day-of-week, and LightGBM outputs.
- **Positional Encoding:** Adds temporal awareness to the Transformer by encoding position information.
- **Custom Schedulers and Optimizers:** Provides fine-grained control of the learning process using schedulers.
- **MAPE Evaluation:** Evaluates model performance using Mean Absolute Percentage Error.


## Code Overview

### Parameters
- **Device:** `cuda` for GPU-based computations.
- **Epochs:** 1000 for training iterations.
- **Encoder & Decoder Features:** Configurable input dimensions.
- **Transformer Parameters:** Customizable embedding size, feedforward dimensions, attention heads, and layers.

### Data Preparation
The `make_data` function creates time-series features:
- **Historical Count Features**
- **Temporal Features:** Day, week, month, day-of-week, quarter, etc.
- **LightGBM Features**

Example:
```python
training_set, test_set = make_data()
```

### Transformer Architecture
1. **Encoder:** Extracts feature embeddings and applies multi-head self-attention.
2. **Decoder:** Uses previous predictions and encoder outputs to refine predictions.
3. **Positional Encoding:** Adds positional context to inputs.

Example:
```python
model = Transformer()
```

### Training and Testing
- The model is trained using a custom `DataLoader` with L1 loss.
- Learning rate is controlled using various `torch.optim.lr_scheduler` strategies.


