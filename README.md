# Sentiment Analysis with LSTM and DistilBERT

## Introduction
This repository contains implementations of sentiment analysis models using LSTM and DistilBERT architectures. The models are trained and evaluated on the Yelp review dataset to classify sentiments as positive, negative, or neutral.

## Folder Structure
- **DistilBERT**: Contains files for the DistilBERT model implementation.
  - `distilbert_main.py`: Main script to run for training and evaluation.
  - `bert_model.py`: Contains the DistilBERT model architecture.

- **LSTM**: Contains files for the LSTM model implementation.
  - `main.py`: Main script to run for training and evaluation.
  - `model.py`: Contains the LSTM model architecture.

## Instructions to Run

### 1. DistilBERT Model
- Navigate to the `DistilBERT` directory.
- Install the required dependencies using `pip install -r requirements.txt`.
- Run `distilbert_main.py` script to train and evaluate the DistilBERT model.
  ```bash
  python distilbert_main.py

### 2. LSTM Model
- Navigate to the LSTM directory.
- Install the required dependencies using `pip install -r requirements.txt`.
- Run main.py script to train and evaluate the LSTM model.
    ```bash
    python main.py

### Dataset
- Download the Yelp review dataset from https://www.yelp.com/dataset.
- Place the dataset files (train.csv and test.csv) in the root directory.