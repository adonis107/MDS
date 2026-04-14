import numpy as np
from scipy.special import erf
from scipy.stats import skewnorm
import torch



def _standard_normal_cdf(t):
    """Î¦(t) â€” standard normal CDF, vectorized via erf."""
    return 0.5 * (1.0 + erf(t / np.sqrt(2.0)))


def _standard_normal_pdf(t):
    """Ï†(t) â€” standard normal PDF, vectorized."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * t ** 2)


def skewed_gaussian_cdf(x, mu, sigma, alpha):
    """
    Skewed Gaussian CDF (scalar or array).
    Based on Fabre & Challet : Equation 35
    F_alpha((x-mu)/sigma)
    """
    cdf = skewnorm.cdf(x, a=alpha, loc=mu, scale=sigma)
    return cdf


def conditional_expectation_skewed_gaussian(x_thresh, mu, sigma, alpha,
                                            upper=True, F_alpha_z=None):
    """
    Conditional expectation E[X | X > x_thresh] or E[X | X <= x_thresh].
    Based on Fabre & Challet : Equation 36 and 37.

    All arguments may be scalars or arrays of the same shape.
    Pass *F_alpha_z* to reuse a previously computed CDF and avoid
    a redundant ``skewnorm.cdf`` call.
    """
    z = (x_thresh - mu) / sigma
    beta = alpha / np.sqrt(1.0 + alpha ** 2)

    if F_alpha_z is None:
        F_alpha_z = skewed_gaussian_cdf(x_thresh, mu, sigma, alpha)

    Phi_alpha_z = _standard_normal_cdf(alpha * z)
    exp_term = np.exp(-0.5 * z ** 2)

    if upper:
        Phi_sqrt = _standard_normal_cdf(np.sqrt(1.0 + alpha ** 2) * z)
        term1 = np.sqrt(2.0 / np.pi) * beta * (1.0 - Phi_sqrt)
        term2 = exp_term * Phi_alpha_z
        E_cond = mu + sigma * (term1 + term2) / (1.0 - F_alpha_z + 1e-10)
    else:
        Phi_sqrt = _standard_normal_cdf(np.sqrt(1.0 + alpha ** 2) * z)
        term1 = np.sqrt(2.0 / np.pi) * beta * Phi_sqrt
        term2 = exp_term * Phi_alpha_z
        E_cond = mu + sigma * (term1 - term2) / (F_alpha_z + 1e-10)

    return E_cond


def calculate_expected_cost(mu, sigma, alpha, spread, delta_a, delta_b, Q, q,
                            epsilon_plus=0.0, epsilon_minus=0.05,
                            p_bid=None, p_ask=None, side='ask'):
    """
    Calculate Expected Cost for spoofing detection (scalar or vectorized).
    Based on Fabre & Challet: Equations 27 and 28.

    All numerical arguments may be scalars or NumPy arrays of the same
    length.  Torch tensors are automatically converted.
    """
    if torch.is_tensor(mu):
        mu = mu.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()

    mu    = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)

    thresh_a = delta_a + 0.5 * spread
    thresh_b = -(delta_b + 0.5 * spread)

    if p_bid is None:
        p_bid = np.ones_like(spread)
    if p_ask is None:
        p_ask = np.ones_like(spread) + spread

    F_a = skewed_gaussian_cdf(thresh_a, mu, sigma, alpha)
    F_b = skewed_gaussian_cdf(thresh_b, mu, sigma, alpha)

    if side == 'ask':
        P_ask_filled = 1.0 - F_a
        P_bid_filled = F_b

        E_dp_ask_nf = conditional_expectation_skewed_gaussian(
            thresh_a, mu, sigma, alpha, upper=False, F_alpha_z=F_a)
        E_dp_bid_f  = conditional_expectation_skewed_gaussian(
            thresh_b, mu, sigma, alpha, upper=False, F_alpha_z=F_b)

        cost_1 = -P_ask_filled * (1 - epsilon_plus) * q * (p_ask + delta_a)
        cost_2 =  P_bid_filled * (1 + epsilon_plus) * Q * (p_bid - delta_b)
        cost_3 = -(1 - P_ask_filled) * (1 - epsilon_minus) * q * (p_bid + E_dp_ask_nf)
        cost_4 = -P_bid_filled * (1 - epsilon_minus) * Q * (p_bid + E_dp_bid_f)

    else:
        P_bid_filled = F_b
        P_ask_filled = 1.0 - F_a

        E_dp_bid_nf = conditional_expectation_skewed_gaussian(
            thresh_b, mu, sigma, alpha, upper=True, F_alpha_z=F_b)
        E_dp_ask_f  = conditional_expectation_skewed_gaussian(
            thresh_a, mu, sigma, alpha, upper=True, F_alpha_z=F_a)

        cost_1 =  P_bid_filled * (1 + epsilon_plus) * q * (p_bid - delta_b)
        cost_2 = -P_ask_filled * (1 - epsilon_plus) * Q * (p_ask + delta_a)
        cost_3 =  (1 - P_bid_filled) * (1 + epsilon_minus) * q * (p_ask + E_dp_bid_nf)
        cost_4 =  P_ask_filled * (1 + epsilon_minus) * Q * (p_ask + E_dp_ask_f)

    return cost_1 + cost_2 + cost_3 + cost_4


def compute_spoofing_gains_batch(mu_arr, sigma_arr, alpha_arr, spread_arr,
                                  delta_a, delta_b, Q, q, fees, side='ask'):
    """
    Vectorised spoofing-gain computation over *all* samples at once.

    Equivalent to looping ``calculate_expected_cost`` twice per sample,
    but ~50-200x faster because every CDF / conditional-expectation call
    processes the full array in one shot.

    Args:
        mu_arr, sigma_arr, alpha_arr: 1-D arrays (N,) â€” PNN output params.
        spread_arr: 1-D array (N,) â€” bid-ask spread per sample.
        delta_a, delta_b: Scalars â€” order distances.
        Q, q: Scalars â€” spoof / genuine order sizes.
        fees: dict with 'maker' and 'taker' keys.
        side: 'ask' or 'bid'.

    Returns:
        gains: 1-D array (N,) â€” spoofing gain per sample.
    """
    mu    = np.asarray(mu_arr,    dtype=np.float64)
    sigma = np.asarray(sigma_arr, dtype=np.float64)
    alpha = np.asarray(alpha_arr, dtype=np.float64)
    spread = np.asarray(spread_arr, dtype=np.float64)

    eps_p = fees['maker']
    eps_m = fees['taker']

    cost_no_spoof = calculate_expected_cost(
        mu, sigma, alpha, spread,
        delta_a=delta_a, delta_b=0.0, Q=0.0, q=q,
        epsilon_plus=eps_p, epsilon_minus=eps_m, side=side,
    )
    cost_with_spoof = calculate_expected_cost(
        mu, sigma, alpha, spread,
        delta_a=delta_a, delta_b=delta_b, Q=Q, q=q,
        epsilon_plus=eps_p, epsilon_minus=eps_m, side=side,
    )
    return cost_no_spoof - cost_with_spoof


def compute_spoofing_gain(model, x_original, x_spoofed, spread, delta_a,
                          delta_b, Q, q, fees, side='ask'):
    """
    Compute expected gain from spoofing strategy.
    Based on Fabre & Challet: Equation 31
    Delta_C(Q, delta) = E[C_spoof(0, delta_a, 0, q) | x0]
                      - E[C_spoof(delta, delta_a, Q, q) | x]
    """
    model.eval()
    with torch.no_grad():
        mu0, sigma0, alpha0 = model(x_original)
        cost_no_spoof = calculate_expected_cost(
            mu0, sigma0, alpha0, spread, delta_a, 0.0, 0.0, q,
            fees['maker'], fees['taker'], side=side)

        mu1, sigma1, alpha1 = model(x_spoofed)
        cost_with_spoof = calculate_expected_cost(
            mu1, sigma1, alpha1, spread, delta_a, delta_b, Q, q,
            fees['maker'], fees['taker'], side=side)

    return cost_no_spoof - cost_with_spoof
