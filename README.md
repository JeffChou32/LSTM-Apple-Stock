# From-Scratch LSTM for AAPL Next-Day Forecasting

A NumPy implementation of an LSTM that learns from a sliding window of recent prices to predict the next day’s close. Forward pass, BPTT, SGD, and gradient clipping are coded by hand, without any deep learning libraries. We evaluate with a chronological 80/10/10 train/val/test split using RMSE and R2, and compare to a simple persistence baseline.
