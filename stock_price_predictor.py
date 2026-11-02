"""Educational next-day stock closing-price baseline (not investment advice)."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def download_data(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker!r}; check ticker and internet.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if not {"Close", "Volume"}.issubset(data.columns):
        raise RuntimeError("Downloaded data must contain Close and Volume columns.")
    data = data[["Close", "Volume"]].dropna().copy()
    if len(data) < 80:
        raise RuntimeError("Not enough history. Try --period 5y or a longer period.")
    return data

def make_features(data: pd.DataFrame):
    frame = data.copy()
    for lag in range(1, 6):
        frame[f"close_lag_{lag}"] = frame["Close"].shift(lag)
    frame["volume_lag_1"] = frame["Volume"].shift(1)
    frame["target_next_close"] = frame["Close"].shift(-1)
    frame = frame.dropna()
    features = [f"close_lag_{lag}" for lag in range(1, 6)] + ["volume_lag_1"]
    return frame[features], frame["target_next_close"]

def main():
    parser = argparse.ArgumentParser(description="Predict next-day close from historical prices.")
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker (default: AAPL)")
    parser.add_argument("--period", default="5y", help="History period, e.g. 1y, 2y, 5y, max")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    data = download_data(args.ticker.upper(), args.period)
    X, y = make_features(data)
    split = int(len(X) * 0.8)
    if split < 30 or len(X) - split < 10:
        raise RuntimeError("Not enough rows for chronological train/test evaluation.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"Ticker: {args.ticker.upper()}")
    print(f"MAE:  {mean_absolute_error(y_test, predicted):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predicted)):.4f}")
    print(f"R²:   {r2_score(y_test, predicted):.4f}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({"actual_next_close": y_test, "predicted_next_close": predicted}, index=y_test.index)
    results.to_csv(output / f"{args.ticker.upper()}_predictions.csv", index_label="date")
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["actual_next_close"], label="Actual")
    plt.plot(results.index, results["predicted_next_close"], label="Predicted")
    plt.title(f"{args.ticker.upper()} — Actual vs Predicted Next-Day Close")
    plt.xlabel("Date"); plt.ylabel("Adjusted closing price"); plt.legend(); plt.tight_layout()
    plot_path = output / f"{args.ticker.upper()}_prediction_plot.png"
    plt.savefig(plot_path, dpi=150); plt.show()
    print(f"Saved CSV and plot in: {output}")

if __name__ == "__main__":
    main()
