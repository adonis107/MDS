import numpy as np
import pandas as pd


def compute_event_flow(df, features, sma_window=10):
    """
    Add Poutré et al. rapidity / event-flow features to `data` (or new DataFrame).

    Quote stuffing: excessive number of messages (updates) per time interval. Rapidity captures this.
    Layering and spoofing: large number of cancellations vs. real trades. Separate cancellations from trades.

    Returns the DataFrame with new columns added.
    """

    # Bid/ask log returns
    features["bid_log_return"] = np.log(df['bid-price-1'] / df['bid-price-1'].shift(1))
    features["ask_log_return"] = np.log(df['ask-price-1'] / df['ask-price-1'].shift(1))

    # Volume deltas
    d_volume_bid = df['bid-volume-1'].diff().fillna(0)
    d_volume_ask = df['ask-volume-1'].diff().fillna(0)

    ### Event detection ###
    # Identify what kind of event triggered the LOB update
    # Cancellation
    is_bid_cancel = (df['bid-price-1'] == df['bid-price-1'].shift(1)) & (d_volume_bid < 0)
    is_ask_cancel = (df['ask-price-1'] == df['ask-price-1'].shift(1)) & (d_volume_ask < 0)
    # Trade
    is_bid_trade = df['ask-price-1'] != df['ask-price-1'].shift(1)
    is_ask_trade = df['bid-price-1'] != df['bid-price-1'].shift(1)

    ### Event sizes ###
    # Magnitude of fake orders (cancellations) vs. real orders (trades)
    features["size_cancel_bid"] = np.where(is_bid_cancel, abs(d_volume_bid), 0)
    features["size_cancel_ask"] = np.where(is_ask_cancel, abs(d_volume_ask), 0)
    features["size_trade_bid"] = np.where(is_bid_trade, df['bid-volume-1'].shift(1).fillna(0), 0)
    features["size_trade_ask"] = np.where(is_ask_trade, df['ask-volume-1'].shift(1).fillna(0), 0)

    # Simple moving averages (paper uses window=10)
    features["SMA_size_bid"] = df["bid-volume-1"].rolling(window=sma_window).mean()
    features["SMA_size_ask"] = df["ask-volume-1"].rolling(window=sma_window).mean()
    features["SMA_cancel_bid"] = features["size_cancel_bid"].rolling(window=sma_window).mean()
    features["SMA_cancel_ask"] = features["size_cancel_ask"].rolling(window=sma_window).mean()
    features["SMA_trade_bid"] = features["size_trade_bid"].rolling(window=sma_window).mean()
    features["SMA_trade_ask"] = features["size_trade_ask"].rolling(window=sma_window).mean()

    # Ensure dt exists (small epsilon to avoid divide-by-zero)
    if "dt" not in df.columns:
        dt = df.get('xltime', None)
        if dt is None:
            features["dt"] = 0.001
        else:
            dt = dt.diff().fillna(0.001).replace(0, 0.001)
            features["dt"] = dt

    ### Rapidity ###
    # Measure the density of events per unit time
    # High cancel rapidity may indicate spoofing activity
    features["rapidity_cancel_bid"] = is_bid_cancel.astype(int) / features["dt"]
    features["rapidity_cancel_ask"] = is_ask_cancel.astype(int) / features["dt"]
    features["rapidity_trade_bid"] = is_bid_trade.astype(int) / features["dt"]
    features["rapidity_trade_ask"] = is_ask_trade.astype(int) / features["dt"]

    # Price speed (return / delta_t)
    features["bid_price_speed"] = features["bid_log_return"].fillna(0) / features["dt"]
    features["ask_price_speed"] = features["ask_log_return"].fillna(0) / features["dt"]

    return features
