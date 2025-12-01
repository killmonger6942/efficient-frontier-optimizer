# efficient-frontier-optimizer
A Python implementation of Modern Portfolio Theory (MPT) to identify optimal asset allocation.
# Efficient Frontier Portfolio Optimizer

## Project Overview
This project implements **Modern Portfolio Theory (MPT)** to construct an optimal portfolio from a basket of assets (e.g., AAPL, MSFT, GOOG, AMZN). By simulating 5,000 unique portfolio combinations, the algorithm identifies the allocation that maximizes returns for a given level of risk.

## Key Features
* **Monte Carlo Simulation:** Generates 5,000 random portfolio weights to map the risk-return landscape.
* **Optimization Engine:** Identifies the "Max Sharpe Ratio" portfolio (The highest return per unit of risk).
* **Visual Analytics:** Plots the Efficient Frontier curve using Matplotlib.

## The Math: Sharpe Ratio
The optimizer seeks to maximize the Sharpe Ratio ($S_p$):

$$S_p = \frac{R_p - R_f}{\sigma_p}$$

Where:
* $R_p$: Expected Portfolio Return
* $R_f$: Risk-Free Rate (e.g., 4%)
* $\sigma_p$: Portfolio Volatility (Standard Deviation)

## Visual Output
The red star indicates the mathematically optimal portfolio allocation.
![Efficient Frontier](frontier_graph.png)

## Technologies Used
* Python 3.10+
* NumPy (Covariance Matrix calculations)
* Pandas (Data processing)
* yFinance (Market data API)

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the script: `python portfolio_optimizer.py`
