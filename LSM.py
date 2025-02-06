import numpy as np
import matplotlib.pyplot as plt

def generate_gbm_paths(S0, ir, sigma, T, N, M):

    dt = T / N
    S_paths = np.zeros((M, N + 1))
    S_paths[:, 0] = S0
    for i in range(1, N + 1):
        z = np.random.normal(0, 1, M)

        # brownian motion
        S_paths[:, i] = S_paths[:, i - 1] * np.exp((ir - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths


def lsm_american_option(S_paths, K, r, dt, option_type, poly_degree):

    M, N_plus_1 = S_paths.shape
    N = N_plus_1 - 1

    # find payoff of each path
    if option_type == 'put':
        payoff = np.maximum(K - S_paths, 0)
    elif option_type == 'call':
        payoff = np.maximum(S_paths - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # payoff at expiration
    cashflow = payoff[:, -1].copy()

    # time index on exercise
    exercise_time = np.full(M, N)

    # backwards induction
    for t in range(N - 1, 0, -1):

        # exclude paths that have alr been exercised
        alive = np.where(exercise_time > t)[0]
        if len(alive) == 0:
            continue

        # only include itm paths for regression
        itm_mask = payoff[alive, t] > 0
        if np.sum(itm_mask) == 0:
            continue
        itm_indices = alive[itm_mask]

        # future discounted cash flows
        Y = cashflow[itm_indices] * np.exp(-r * dt * (exercise_time[itm_indices] - t))

        # current asset prices (itm only)
        X = S_paths[itm_indices, t]

        # ran into issues when there were too few asset prices in the money
        if len(X) < 4:
            continue

        # regress future discounted cash flows onto asset price at present w/ polynomial regression

        # This is the only part of the LSM algorithm that varies. You can implement neural networks
        # or other regression models for more accurate continuation values
        coeffs = np.polyfit(X, Y, poly_degree)
        continuation_value = np.polyval(coeffs, X)

        # value of option at present
        immediate_exercise = payoff[itm_indices, t]

        # exercise if immediate value is greater than continuation value
        exercise_now = immediate_exercise > continuation_value

        # update where early exercise is optimal
        exercise_indices = itm_indices[exercise_now]
        cashflow[exercise_indices] = payoff[exercise_indices, t]
        exercise_time[exercise_indices] = t

    # discount cash flows from exercise to t=0
    option_values = cashflow * np.exp(-r * dt * exercise_time)
    option_price = np.mean(option_values)
    return option_price


M = 100  # num of simulated paths
N = 365  # num of time steps
T = 1  # time to expiration
S0 = 100  # initial stock price
ir = 0.0417  # drift (assumed to equal risk-free interest rate b/c black scholes assumes risk-neutral)
r = 0.0417  # risk-free interest rate
sigma = 0.2  # volatility
K = 110  # strike price
dt = T / N  # time step
poly_degree = 3  # degree of polynomial regression


# generate stock price paths
S_paths = generate_gbm_paths(S0, ir, sigma, T, N, M)

# plot all generated stock paths
t_grid = np.linspace(0, T, N + 1)
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(t_grid, S_paths[i], lw=1)
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Simulated Stock Price Paths')
plt.show()

# estimate price of the option
option_price = lsm_american_option(S_paths, K, r, dt, option_type='call', poly_degree=poly_degree)
print("Estimated option price", option_price)
