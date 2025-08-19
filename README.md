# Marketing Analytics Project

This repository contains a collection of end-to-end marketing analytics workflows implemented in Python and Jupyter notebooks.

## Overview

The project is composed of three main components:

1. **Customer Segmentation (RFM & KMeans)**: We compute Recency, Frequency and Monetary (RFM) metrics from transactional data and apply KMeans clustering to identify distinct customer segments.
2. **Review Sentiment Analysis**: We perform sentiment analysis on synthetic product reviews using TF-IDF vectorisation and logistic regression.
3. **Campaign Performance Analysis**: We analyse marketing campaign data to derive key metrics such as click-through rate, conversion rate, cost per conversion, and return on investment.

Each analysis is encapsulated in its own Jupyter notebook under the `notebooks/` directory and leverages helper functions defined in the `src/` package. Synthetic datasets used in the notebooks are stored in the `data/` directory.

## Structure

```
marketing-analytics-project/
├── data/                   # Synthetic input datasets
├── notebooks/              # Jupyter notebooks with step-by-step analysis
├── src/                    # Helper modules with reusable functions
├── README.md               # Project overview and instructions
├── requirements.txt        # List of Python dependencies
└── LICENSE                 # Project license
```

## Getting Started

1. Clone the repository to your local machine.
2. Create a virtual environment (optional) and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Launch Jupyter Notebook or JupyterLab and explore the notebooks in the `notebooks/` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
