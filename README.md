# CodeTech Artificial Intelligence Internship

**Intern:** Amaan Sayyed  
**Domain:** Artificial Intelligence  
**Duration:** 1 month

This repository contains two selected internship projects:

1. **Stock Price Predictor** — an educational baseline that uses historical market data and lagged features to estimate the next trading day's closing price.
2. **Handwritten Digit Recognizer** — a Convolutional Neural Network (CNN) trained on MNIST to classify handwritten digits from 0 to 9.

## Project 1 — Stock Price Predictor

The script downloads historical OHLCV data with yfinance, creates lagged closing-price and volume features, uses a chronological train/test split, trains a Linear Regression baseline, reports MAE/RMSE/R², and saves a prediction CSV and plot.

Run:
```bash
pip install -r requirements.txt
python stock_price_predictor.py --ticker AAPL --period 5y
```
Replace `AAPL` with a valid Yahoo Finance ticker. Internet access is needed to download data.

## Project 2 — Handwritten Digit Recognizer

The script loads MNIST, normalizes 28×28 grayscale images, trains a CNN for 10-class classification, evaluates test performance, creates training/confusion-matrix/sample-prediction plots, and saves the model.

Run:
```bash
pip install -r requirements.txt
python handwritten_digit_recognizer.py --epochs 5
```
TensorFlow downloads MNIST the first time, so internet access is needed initially. Training time depends on hardware.

## Requirements
Python 3.10 or newer is recommended. Install dependencies using `pip install -r requirements.txt`.

## Important notes
- The stock model is for learning only, not investment advice. Historical patterns do not guarantee future prices.
- Run the scripts yourself and review their outputs before submitting. Results depend on downloaded data and the training environment.
- Add your own observations and screenshots after running the projects. Do not claim results you have not generated.

## Repository structure
```
.
├── stock_price_predictor.py
├── handwritten_digit_recognizer.py
├── requirements.txt
└── README.md
```
