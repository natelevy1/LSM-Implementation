import numpy as np
import matplotlib.pyplot as plt

def generate_gbm_paths(S0, ir, sigma, T, N, M):

    dt = T / N
    S_paths = np.zeros((M, N + 1))
    S_paths[:, 0] = S0
    for i in range(1, N + 1):
        z = np.random.normal(0, 1, M)
        S_paths[:, i] = S_paths[:, i - 1] * np.exp((ir - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths


def lsm_american_option(S_paths, K, r, dt, option_type, poly_degree):

    M, N_plus_1 = S_paths.shape
    N = N_plus_1 - 1  # Number of time steps

    # Compute the payoff at each time for each path.
    if option_type == 'put':
        payoff = np.maximum(K - S_paths, 0)
    elif option_type == 'call':
        payoff = np.maximum(S_paths - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # At maturity, the payoff is known.
    cashflow = payoff[:, -1].copy()  # cashflow realized if never exercised early
    exercise_time = np.full(M, N)  # record the time index when exercise occurs (default = maturity)

    # Work backward from time N-1 to 1.
    for t in range(N - 1, 0, -1):
        # Consider only paths that have not yet been exercised.
        alive = np.where(exercise_time > t)[0]
        if len(alive) == 0:
            continue

        # Among alive paths, focus on those that are in–the–money.
        itm_mask = payoff[alive, t] > 0
        if np.sum(itm_mask) == 0:
            continue
        itm_indices = alive[itm_mask]

        # For these paths, the “continuation value” is the (already determined) future cashflow,
        # discounted back from the recorded exercise time to the current time t.
        Y = cashflow[itm_indices] * np.exp(-r * dt * (exercise_time[itm_indices] - t))
        X = S_paths[itm_indices, t]

        # If there are too few points for a reliable regression, skip this time step.
        if len(X) < poly_degree + 1:
            continue

        # Fit a polynomial of the chosen degree to estimate the continuation value.
        coeffs = np.polyfit(X, Y, poly_degree)
        continuation_value = np.polyval(coeffs, X)

        # Immediate exercise payoff at time t.
        immediate_exercise = payoff[itm_indices, t]

        # Decide to exercise if the immediate payoff exceeds the continuation value.
        exercise_now = immediate_exercise > continuation_value

        # Update those paths where early exercise is optimal.
        exercise_indices = itm_indices[exercise_now]
        cashflow[exercise_indices] = payoff[exercise_indices, t]
        exercise_time[exercise_indices] = t

    # Discount each path's cashflow from its exercise time back to time 0.
    option_values = cashflow * np.exp(-r * dt * exercise_time)
    option_price = np.mean(option_values)
    return option_price


M = 100000  # Number of simulated paths
N = 365  # Number of time steps (e.g. daily steps for 1 year)
T = 1  # Time to maturity in years
S0 = 100  # Initial stock price
ir = 0.0417  # Stock's drift (or interest rate for GBM simulation)
r = 0.0417  # Risk-free interest rate (used for discounting)
sigma = 0.2  # Volatility (20% per annum)
K = 110  # Strike price of the option
dt = T / N  # Time step size
poly_degree = 3  # Degree of the polynomial used in regression


# Generate stock price paths
S_paths = generate_gbm_paths(S0, ir, sigma, T, N, M)

# Plot a subset of paths (here we plot 20 for clarity)
t_grid = np.linspace(0, T, N + 1)
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(t_grid, S_paths[i], lw=1)
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Simulated Stock Price Paths')
plt.show()


american_call_price = lsm_american_option(S_paths, K, r, dt, option_type='call', poly_degree=poly_degree)
print("Estimated American call option price:", american_call_price)
