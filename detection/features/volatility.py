import numpy as np
import pandas as pd


def compute_volatility(df, features, window=50):
    ### Volatility and risk ###
    # Standard deviation of returns
    # High volatility may indicate uncertainty or manipulation attempts
    features["volatility_50"] = features["log_return"].rolling(window=window).std()
    features["volatility_100"] = features["log_return"].rolling(window=2*window).std()

    # Price range: captures extreme price movements within the window
    mid = features["mid_price"] if "mid_price" in features.columns else df["mid_price"]
    rolling_max = mid.rolling(window=window).max()
    rolling_min = mid.rolling(window=window).min()
    features["price_range_50"] = (rolling_max - rolling_min) / mid

    # Absolute velocity
    features["abs_velocity"] = features["log_return"].abs()

    # Time delta
    dt = df['xltime'].diff().fillna(1e-6)
    dt = dt.replace(0, 1e-6)
    features["dt"] = dt

    features['log_dt'] = np.log(features['dt'] + 1e-6)

    return features

