import numpy as np
from scipy.optimize import minimize
from scipy.stats import vonmises
from scipy.stats import circmean, circstd
from scipy.stats import pearsonr, chi2

# Define the circular model
def circular_model(mu, beta, x):
    return (mu + 2 * np.arctan(beta * x)) % (2 * np.pi)

# Weighted residuals for WLS
def weighted_residuals(params, x, y_obs, weights):
    mu, beta = params
    y_pred = circular_model(mu, beta, x)
    residuals = np.angle(np.exp(1j * (y_obs - y_pred)))  # Circular difference
    return np.sum(weights * residuals**2)

# Negative log-likelihood for MLE
def negative_log_likelihood(params, x, y_obs, kappa=1):
    mu, beta = params
    y_pred = circular_model(mu, beta, x)
    # Von Mises log likelihood
    likelihoods = vonmises.pdf(y_obs, kappa, loc=y_pred)
    return -np.sum(np.log(likelihoods + 1e-9))  # Avoid log(0)

# Combined objective function
def combined_objective(params, x, y_obs, weights, kappa=1, wls_weight=0.5):
    wls_loss = weighted_residuals(params, x, y_obs, weights)
    mle_loss = negative_log_likelihood(params, x, y_obs, kappa)
    return wls_weight * wls_loss + (1 - wls_weight) * mle_loss

# Main optimization function
def fit_circular_model(x, y_obs, weights, kappa=1, wls_weight=0.5):
    initial_guess = [circmean(y_obs, nan_policy='omit'), 0.1]  # Initial guesses for mu and beta
    bounds = [(0, 2 * np.pi), (None, None)]  # mu is circular, beta is unrestricted
    result = minimize(
        combined_objective,
        initial_guess,
        args=(x, y_obs, weights, kappa, wls_weight),
        bounds=bounds,
        method="L-BFGS-B"
    )
    return result.x, result.fun  # Optimal parameters and objective value

# New functions for statistical testing
def fit_null_model(x, y_obs, weights, kappa=1, wls_weight=0.5):
    """Fit the null model with beta = 0"""
    def null_objective(mu, x, y_obs, weights, kappa, wls_weight):
        params = [mu[0], 0.0]  # Force beta = 0
        return combined_objective(params, x, y_obs, weights, kappa, wls_weight)
    
    initial_guess = [circmean(y_obs, nan_policy='omit'), 0.1]
    bounds = [(0, 2 * np.pi)]
    result = minimize(
        null_objective,
        initial_guess,
        args=(x, y_obs, weights, kappa, wls_weight),
        bounds=bounds,
        method="L-BFGS-B"
    )
    return [result.x[0], 0.0], result.fun

 # Permutation test for p-value
def permutation_test(x, y_obs, weights, fitted_beta, num_permutations=1000, kappa=1, wls_weight=0.5):
    beta_null = []
    for _ in range(num_permutations):
        # Shuffle x to break the relationship between x and y
        x_shuffled = np.random.permutation(x)
        # Fit the model to the shuffled data
        params, _= fit_circular_model(x_shuffled, y_obs, weights, kappa, wls_weight)
        beta_null.append(params[1])  # Store the beta value
    
    # Calculate p-value
    beta_null = np.array(beta_null)
    p_value = (np.sum(np.abs(beta_null) >= np.abs(fitted_beta)) + 1) / (num_permutations + 1)
    return p_value

def calculate_beta_significance(x, y_obs, weights, kappa=1, wls_weight=0.5):
    """
    """
    # Fit both models
    full_params, full_obj = fit_circular_model(x, y_obs, weights, kappa, wls_weight)
    null_params, null_obj = fit_null_model(x, y_obs, weights, kappa, wls_weight)
    
    # Calculate likelihood ratio test statistic
    # Use only the MLE component for the test
    full_nll = negative_log_likelihood(full_params, x, y_obs, kappa)
    null_nll = negative_log_likelihood(null_params, x, y_obs, kappa)
    lr_statistic = 2 * (null_nll - full_nll)
    
    # Calculate p-value (1 degree of freedom difference between models)
    p_value = chi2.sf(lr_statistic, df=1)
    
    return p_value
