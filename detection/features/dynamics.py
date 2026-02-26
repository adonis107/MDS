import numpy as np
import pandas as pd


def calculate_sweep_cost(prices, volumes, target_size=1000):
    """
    Calculate sweep-to-fill cost for a given row of orderbook data.
    Args:
        prices: array of prices at each level
        volumes: array of volumes at each level
    Returns:
        sweep_cost: the sweep-to-fill cost
    """
    cumulative_volume = np.cumsum(volumes, axis=1)

    cumulative_volume_prev = np.zeros_like(cumulative_volume)
    cumulative_volume_prev[:, 1:] = cumulative_volume[:, :-1]

    # Volume to take = Target - (Volume already taken in previous levels)
    volume_needed = np.maximum(0, target_size - cumulative_volume_prev)
    volume_taken = np.minimum(volumes, volume_needed)

    total_cost = np.sum(volume_taken * prices, axis=1)
    total_volume = np.sum(volume_taken, axis=1)

    sweep_cost = np.divide(total_cost, total_volume, out=np.full_like(total_cost, np.nan), where=total_volume!=0)
    return sweep_cost


def get_slope(prices, volumes, depth):
    """
    Calculate the slope (elasticity) of the orderbook side (ask or bid) for a given row.
    We want slope beta for: Price = alpha + beta * log(Cumulative Volume).
    Formula: beta = Cov(X, Y) / Var(X).
    Args:
            row: a row of the orderbook DataFrame
            volumes: volumes at each level
            depth: number of levels to consider
    Returns:
            slope: absolute value of the slope from linear regression
    """
    p_slice = prices[:, :depth]
    v_slice = volumes[:, :depth]

    x = np.log(np.cumsum(v_slice, axis=1) + 1)
    y = p_slice

    x_mean = np.mean(x, axis=1, keepdims=True)
    y_mean = np.mean(y, axis=1, keepdims=True)

    dx = x - x_mean
    dy = y - y_mean

    # Covariance and variance
    numerator = np.sum(dx * dy, axis=1)
    denominator = np.sum(dx ** 2, axis=1)

    slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    return np.abs(slope)


def compute_elasticity(df, features, n_levels=10, target_size=1000, slope_depth=5):
    ### Liquidity cost and elasticity ###
    # Prepare arrays for vectorized sweep cost and slope calculations
    P_bid = np.stack([df[f'bid-price-{i}'].values for i in range(1, n_levels + 1)], axis=1)
    V_bid = np.stack([df[f'bid-volume-{i}'].values for i in range(1, n_levels + 1)], axis=1)
    P_ask = np.stack([df[f'ask-price-{i}'].values for i in range(1, n_levels + 1)], axis=1)
    V_ask = np.stack([df[f'ask-volume-{i}'].values for i in range(1, n_levels + 1)], axis=1)

    # Sweep-to-fill cost
    # Effective price paid to execute a large order (target_size). Accounts for lack of liquidity at the top level.
    bid_sweep_cost = calculate_sweep_cost(P_bid, V_bid, target_size=target_size)
    ask_sweep_cost = calculate_sweep_cost(P_ask, V_ask, target_size=target_size)
    features["ask_sweep_cost"] = ask_sweep_cost - features["mid_price"]
    features["bid_sweep_cost"] = features["mid_price"] - bid_sweep_cost

    # Slope (elasticity) of orderbook sides
    # Measures how quickly prices change as volume is added/removed
    # A steep slope indicates low liquidity (small volume causes large price changes), flat slope indicates high liquidity
    features["ask_slope"] = get_slope(P_bid, V_bid, depth=slope_depth)
    features["bid_slope"] = get_slope(P_ask, V_ask, depth=slope_depth)

    return features


def compute_dynamics(df, features, window=50):
    ### Price dynamics ###
    # Mid-price and spread
    features["mid_price"] = (df['ask-price-1'] + df['bid-price-1']) / 2
    # Wide spreads can indicate illiquidity or uncertainty
    features["spread"] = df['ask-price-1'] - df['bid-price-1']

    ### Micro-structure deviation ###
    # Fair value deviation from mid-price based on L1 imbalance
    # If the micro-price deviates significantly from mid-price, it may indicate pressure to move price
    features["micro_price_deviation"] = features["L1_Imbalance"] * (features["spread"] / 2)

    ### Volume concentration ###
    # Ratio of volume deep in the book (levels 2-5) to volume the top level
    # High values may indicate high volume orders placed away from best price (layering)
    features['bid_depth_ratio'] = df[[f'bid-volume-{i}' for i in range(2, 6)]].sum(axis=1) / df['bid-volume-1'].replace(0, np.nan)
    features['ask_depth_ratio'] = df[[f'ask-volume-{i}' for i in range(2, 6)]].sum(axis=1) / df['ask-volume-1'].replace(0, np.nan)
    features[['bid_depth_ratio', 'ask_depth_ratio']] = features[['bid_depth_ratio', 'ask_depth_ratio']].fillna(0)

    ### Dynamics and velocity ###
    # Log returns
    features["log_return"] = np.log(features["mid_price"] / features["mid_price"].shift(1))
    # Volume deltas at L1
    features["bid_volume_delta"] = df['bid-volume-1'].diff()
    features["ask_volume_delta"] = df['ask-volume-1'].diff()

    features["mid_price_velocity"] = features['mid_price'].diff()
    features["mid_price_acceleration"] = features["mid_price_velocity"].diff()
    features["mid_price_volatility"] = features["mid_price"].rolling(window=window).std()

    # Net order flow at L1
    # Differentiates between volume changes due to price shifts vs. cancellations/additions at the same price level
    features["net_bid_flow"] = df['bid-volume-1'].diff() * (df['bid-price-1'] == df['bid-price-1'].shift(1)).astype(int)
    features["net_ask_flow"] = df['ask-volume-1'].diff() * (df['ask-price-1'] == df['ask-price-1'].shift(1)).astype(int)

    return features

