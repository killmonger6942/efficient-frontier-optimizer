import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN'] # The Basket
num_portfolios = 5000                      # Number of combinations to test
risk_free_rate = 0.04                      # Assuming 4% risk-free rate (T-Bills)

start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

print(f"--- STARTING PORTFOLIO OPTIMIZATION ---")
print(f"1. Downloading data for: {tickers}...")

# --- 2. GET DATA ---
# Download all tickers at once
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# CLEANUP: Handle yfinance updates
if 'Adj Close' in data.columns:
    data = data['Adj Close']
elif 'Close' in data.columns:
    data = data['Close']

if data.empty:
    print("Error: No data downloaded.")
    exit()

# --- 3. CALCULATE METRICS ---
print("2. Calculating Returns and Covariance...")
# Log Returns are standard for mathematical modeling
log_returns = np.log(data / data.shift(1)).dropna()

# --- 4. THE MONTE CARLO LOOP ---
print(f"3. Simulating {num_portfolios} random portfolios...")

all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    # Create Random Weights
    weights = np.array(np.random.random(len(tickers)))
    weights = weights / np.sum(weights) # Rebalance to sum to 1 (100%)
    all_weights[i,:] = weights # Save weights

    # Expected Return (Annualized)
    ret_arr[i] = np.sum(log_returns.mean() * weights * 252)

    # Expected Volatility (Annualized)
    # This is the matrix math: sqrt(w.T * Cov * w)
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]

# --- 5. FIND THE BEST PORTFOLIO ---
print("4. Identifying the Optimal Allocation...")
max_sharpe_idx = sharpe_arr.argmax() # Index of the highest Sharpe Ratio
max_sharpe_ret = ret_arr[max_sharpe_idx]
max_sharpe_vol = vol_arr[max_sharpe_idx]

print(f"\n--- OPTIMAL PORTFOLIO (Max Sharpe: {sharpe_arr.max():.2f}) ---")
print(f"Return: {max_sharpe_ret*100:.2f}%")
print(f"Volatility (Risk): {max_sharpe_vol*100:.2f}%")
print("\nAllocation:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {all_weights[max_sharpe_idx,i]*100:.2f}%")

# --- 6. VISUALIZE (EFFICIENT FRONTIER) ---
plt.figure(figsize=(12,8))
# Plot all 5000 portfolios
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')

# Add a Red Star for the "Optimal" portfolio
plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', s=100, edgecolors='black', label='Optimal Portfolio')
plt.legend()
plt.title(f'Efficient Frontier: {tickers}')
plt.show()
