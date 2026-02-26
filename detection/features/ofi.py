import pandas as pd
import numpy as np


def compute_ofi(df, features, levels=5):
    """
    Compute order flow imbalance features based on volume changes at each level.
    """
    for i in range(1, levels + 1):
        bid_p = df[f'bid-price-{i}']
        ask_p = df[f'ask-price-{i}']
        bid_v = df[f'bid-volume-{i}']
        ask_v = df[f'ask-volume-{i}']

        bid_p_prev = bid_p.shift(1)
        ask_p_prev = ask_p.shift(1)
        bid_v_prev = bid_v.shift(1)
        ask_v_prev = ask_v.shift(1)

        bid_flow = np.where(bid_p > bid_p_prev, bid_v,
                            np.where(bid_p == bid_p_prev, bid_v - bid_v_prev, -bid_v_prev))
        ask_flow = np.where(ask_p < ask_p_prev, ask_v,
                            np.where(ask_p == ask_p_prev, ask_v - ask_v_prev, -ask_v_prev))

        features[f'order_flow_imbalance_level_{i}'] = bid_flow - ask_flow

    return features

