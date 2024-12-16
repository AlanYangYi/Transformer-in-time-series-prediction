# Transformer Model for Time-Series Prediction

This repository contains a Python implementation of a Transformer-based model designed for time-series forecasting tasks, specifically for predicting order quantities. The model leverages advanced techniques like multi-head attention, positional encoding, and feature engineering to provide accurate predictions based on historical data.

---

## Key Features

### 1. **Transformer Architecture**
- **Multi-Head Attention Mechanism**: Captures long-term dependencies in time-series data by leveraging scaled dot-product attention.
- **Encoder-Decoder Structure**: The encoder processes input sequences, while the decoder predicts future values using both encoder outputs and autoregressive feedback.
- **Positional Encoding**: Integrates temporal information into the model, enhancing its ability to learn sequential patterns.

### 2. **Custom Feature Engineering**
The model incorporates rich feature engineering methods to enhance predictive accuracy:
- **Historical Features**: Generates lagged features from historical counts.
- **Temporal Features**: Includes month, week, day, day-of-week, day-of-year, and quarter indicators.
- **LightGBM Predictions**: Integrates features derived from an external LightGBM model.

### 3. **Data Handling**
- Processes input data from CSV files (e.g., `original_data.csv` and `lightGBM.csv`).
- Groups data by area (`area_id`) and creates multiple time-based features.
- Handles missing values by applying forward shifts and dropping rows with insufficient data.

### 4. **Training and Evaluation**
- **Loss Function**: Mean Absolute Error (MAE) used to optimize predictions.
- **Evaluation Metrics**: Mean Absolute Percentage Error (MAPE) is used for evaluating the model’s performance.
- **Early Stopping**: Terminates training if the model achieves a predefined MAPE threshold.





---

## Visualization

The architecture of the Transformer model is illustrated below:

![Transformer Model Architecture](https://github.com/AlanYangYi/Transformer-in-time-series-prediction/blob/main/Time-series-transformer-forecasting-based-model-architecture.png)

---

## Advanced Features

### Dynamic Learning Rate Scheduling
Four learning rate schedulers are implemented:
- **Scheduler 0**: Activates for the initial 40 epochs.
- **Scheduler 1**: Fine-tunes the model for epochs between 40 and 60.
- **Schedulers 2 & 3**: Gradually reduce the learning rate based on the loss plateau.

### Flexible Feature Selection
The `make_data` function generates features dynamically, allowing easy customization for additional temporal or external data sources. Supported features include:
- Count-based lagged features
- Temporal indicators (month, week, day, quarter, etc.)
- Integration with LightGBM output

---



### Example Output
- **Training Loss**: Monitored and logged for each epoch.
- **MAPE**: Provides an overview of prediction accuracy on the test set.
- **Test Loss**: Helps track generalization performance.

---

## Reference


```bibtex
@article{Wu2020DeepTM,
  title={Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case},
  author={Neo Wu and Bradley Green and Xue Ben and Shawn O’Banion},
  journal={ArXiv},
  year={2020},
  volume={abs/2001.08317},
  url={https://api.semanticscholar.org/CorpusID:210861210}
}
```

