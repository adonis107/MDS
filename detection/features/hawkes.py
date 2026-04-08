import pandas as pd
import numpy as np


def compute_hawkes(df:pd.DataFrame, features, etas=None, betas=None,
                                     levels=10, price_scale=10000,
                                     halflife_short=5, halflife_long=50, deep_halflife=10):
    """
    Add Fabre & Challet features. Compute Hawkes-style memory features (limit + market flows), deep insertions and
    weighted multilevel limit order flows with spatial/time decay. 

    Hawkes (memory): captures self-exciting nature of order flows. A burst of buy orders increases the probability of more buy orders. EWMA captures this memory effect.
    Distance (spatial decay): spoofing orders are sensitive to their distance from the mid-price. Orders placed too far away have low impact, orders too close risk execution.
        eta (distance scale): controls how quickly the influence of an order decays as it gets further from the mid-price.
        beta (time scale): controls how quickly the memory of past orders fades over time.
    """
    if etas is None:
        etas = [0.001, 0.1, 1.0, 10.0]
    if betas is None:
        betas = [10, 100, 1000]

    # basic deltas used by flows
    d_volume_bid = df['bid-volume-1'].diff().fillna(0)
    d_volume_ask = df['ask-volume-1'].diff().fillna(0)

    ### Limit & Market order flows ###
    # Separate flow that adds liquidity (limit) from flow that consumes liquidity (market)
    # Market orders are the ground truth, limit orders can be spoofing candidates
    price_change_bid = df['bid-price-1'] != df['bid-price-1'].shift(1)
    price_change_ask = df['ask-price-1'] != df['ask-price-1'].shift(1)
    flow_L_bid = np.where((~price_change_bid) & (d_volume_bid > 0), d_volume_bid, 0)
    flow_L_ask = np.where((~price_change_ask) & (d_volume_ask > 0), d_volume_ask, 0)

    # Market order flows on best level (unified convention).
    # raw_M_bid: sell market orders consuming bid-side liquidity (bid price drops,
    #   or volume decreases at unchanged bid price).
    # raw_M_ask: buy market orders consuming ask-side liquidity (ask price rises,
    #   or volume decreases at unchanged ask price).
    d_price_bid = df['bid-price-1'].diff().fillna(0)
    d_price_ask = df['ask-price-1'].diff().fillna(0)

    raw_M_bid = np.where(d_price_bid < 0, df['bid-volume-1'].shift(1),
                         np.where((d_price_bid == 0) & (d_volume_bid < 0), -d_volume_bid, 0))
    raw_M_ask = np.where(d_price_ask > 0, df['ask-volume-1'].shift(1),
                         np.where((d_price_ask == 0) & (d_volume_ask < 0), -d_volume_ask, 0))

    ### Hawkes-style memory features ###
    # Short vs long term memory for limit orders
    # Manipulators can create short bursts of activity taht deviate from normal patterns
    features["Hawkes_L_bid_short"] = pd.Series(flow_L_bid, index=df.index).ewm(halflife=halflife_short).mean()
    features["Hawkes_L_ask_short"] = pd.Series(flow_L_ask, index=df.index).ewm(halflife=halflife_short).mean()
    features["Hawkes_M_bid_short"] = pd.Series(raw_M_bid, index=df.index).ewm(halflife=halflife_short).mean()
    features["Hawkes_M_ask_short"] = pd.Series(raw_M_ask, index=df.index).ewm(halflife=halflife_short).mean()

    features["Hawkes_L_bid_long"] = pd.Series(flow_L_bid, index=df.index).ewm(halflife=halflife_long).mean()
    features["Hawkes_L_ask_long"] = pd.Series(flow_L_ask, index=df.index).ewm(halflife=halflife_long).mean()

    ### Deep order insertions ###
    # Anomalous activity deep in the book (level 5) usually indicates layering
    if "bid-volume-5" in df.columns and "ask-volume-5" in df.columns:
        d_volume_bid_L5 = df["bid-volume-5"].diff().fillna(0)
        d_volume_ask_L5 = df["ask-volume-5"].diff().fillna(0)
        deep_insertion_bid = (d_volume_bid_L5 > 0).astype(int) * d_volume_bid_L5
        deep_insertion_ask = (d_volume_ask_L5 > 0).astype(int) * d_volume_ask_L5
        features["Deep_order_insertion_bid"] = pd.Series(deep_insertion_bid, index=df.index).ewm(halflife=deep_halflife).mean()
        features["Deep_order_insertion_ask"] = pd.Series(deep_insertion_ask, index=df.index).ewm(halflife=deep_halflife).mean()

    ### Weighted flow with spatial and time decay ###
    # Weight a new order by exp(-eta * distance_from_midprice)
    # If eta is high, only orders very close to mid-price matter
    # If eta is low, even distant orders have influence
    mid_price = (df['bid-price-1'] + df['ask-price-1']) / 2
    # initialize per-eta accumulators as Series
    total_weighted_flow_bid = {eta: pd.Series(0.0, index=df.index) for eta in etas}
    total_weighted_flow_ask = {eta: pd.Series(0.0, index=df.index) for eta in etas}

    for i in range(1, levels + 1):
        vol_col_bid = f"bid-volume-{i}"
        vol_col_ask = f"ask-volume-{i}"
        price_col_bid = f"bid-price-{i}"
        price_col_ask = f"ask-price-{i}"
        if vol_col_bid not in df.columns or price_col_bid not in df.columns:
            break

        # Only count volume increases when the price at this level is unchanged
        # (pure limit-order insertions, not price-level shifts)  # ALIGNED: report §3.1.5
        price_unch_bid = df[price_col_bid] == df[price_col_bid].shift(1)
        price_unch_ask = df[price_col_ask] == df[price_col_ask].shift(1)
        d_vol_bid = (df[vol_col_bid].diff().clip(lower=0).fillna(0)) * price_unch_bid
        d_vol_ask = (df[vol_col_ask].diff().clip(lower=0).fillna(0)) * price_unch_ask
        dist_bid = np.abs(df[price_col_bid] - mid_price)
        dist_ask = np.abs(df[price_col_ask] - mid_price)

        for eta in etas:
            spatial_decay_bid = np.exp(-eta * dist_bid * price_scale)
            spatial_decay_ask = np.exp(-eta * dist_ask * price_scale)
            total_weighted_flow_bid[eta] += d_vol_bid * spatial_decay_bid
            total_weighted_flow_ask[eta] += d_vol_ask * spatial_decay_ask

    # Time decay: build columns for combinations of betas and etas
    for beta in betas:
        for eta in etas:
            col_name_bid = f"Hawkes_L_bid_beta{beta}_Eta{eta}"
            col_name_ask = f"Hawkes_L_ask_beta{beta}_Eta{eta}"
            features[col_name_bid] = total_weighted_flow_bid[eta].ewm(halflife=beta).mean()
            features[col_name_ask] = total_weighted_flow_ask[eta].ewm(halflife=beta).mean()

    # Market flows with time decay (same raw_M_bid/raw_M_ask defined above)
    for beta in betas:
        features[f"Hawkes_M_bid_beta{beta}"] = pd.Series(raw_M_bid, index=df.index).ewm(halflife=beta).mean()
        features[f"Hawkes_M_ask_beta{beta}"] = pd.Series(raw_M_ask, index=df.index).ewm(halflife=beta).mean()

    return features

