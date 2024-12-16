# Transformer Model for Time-Series Prediction

This repository contains a Python implementation of a Transformer-based model designed for time-series forecasting tasks (Order quantity forecast). The model leverages encoder and decoder structures to effectively capture sequential dependencies in data and use the previous 20 days' data to predict the next week's order count.
## Key Features
- **Multi-head Attention Mechanism:** Utilizes scaled dot-product attention to capture global dependencies in time-series data.
- **Custom Data Processing:** Generates features based on historical counts, months, weeks, days, day-of-week, and LightGBM outputs.
- **Positional Encoding:** Adds temporal awareness to the Transformer by encoding position information.
- **Custom Schedulers and Optimizers:** Provides fine-grained control of the learning process using schedulers.
- **MAPE Evaluation:** Evaluates model performance using Mean Absolute Percentage Error.


![name-of-you-image](https://github.com/AlanYangYi/Transformer-in-time-series-prediction/blob/main/Time-series-transformer-forecasting-based-model-architecture.png)


## Reference
@article{Wu2020DeepTM,
  title={Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case},
  author={Neo Wu and Bradley Green and Xue Ben and Shawn O’Banion},
  journal={ArXiv},
  year={2020},
  volume={abs/2001.08317},
  url={https://api.semanticscholar.org/CorpusID:210861210}
}
