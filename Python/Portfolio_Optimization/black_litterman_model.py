# Author: Nolan Alexander

import pandas as pd
import numpy as np
import datetime as dt
import time
import sys

dir_to_python = 'Path/To/The/Code'
directory = dir_to_python + '/Python/Portfolio_Optimization'
sys.path.append(directory)
from assets_data_collection import read_in_assets_data, read_in_cur_price_and_outstanding_shares
from setup_stock_data import convert_to_log_growth, calc_mean_cov_matrix_and_size
from markowitz_portfolio_optimization import calc_markowitz_efficient_frontier, plot_efficient_frontier
from annual_conversion import convert_to_annual_returns

mkt_rf_data_file_source = directory + '/Data/F-F_Research_Data_Factors.csv'
# Mkt-RF Data Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html


# Implementation of the Black-Litterman Model.  The model uses a weighted combination of the market 
# capitalization views and investor's views to output better estimates of the excess expected returns

# Finds the market equilibrium portfolio weights based on market capitalization
def find_market_equilibrium_weights(assets):
    cur_value_adj_close, shares_outstanding = read_in_cur_price_and_outstanding_shares(assets)
    market_capitalization = cur_value_adj_close * shares_outstanding
    market_cap_weights = market_capitalization/market_capitalization.sum()
    return market_cap_weights

# Calculates the risk aversion factor based on the market and risk free rate data
def calculate_risk_aversion_factor(market_minus_risk_free_rate_data):
    avg_mkt_minus_rf = market_minus_risk_free_rate_data.mean()
    sd_mkt_minus_rf = market_minus_risk_free_rate_data.std()
    delta = avg_mkt_minus_rf/sd_mkt_minus_rf**2
    return delta

# performs reverse_optimization to find the equilibrium expected excess returns
def reverse_optimization(risk_aversion_factor, cov_matrix, market_equilibrium_portfolio_weights):
    # Set up notation
    delta = risk_aversion_factor
    sigma = cov_matrix
    
    # Reverse optimization derived formula
    pi = (delta * sigma).dot(market_equilibrium_portfolio_weights)
    return pi;

# Calculates the Black-Litterman estimates of expected returns based on prior views and view of the market
# Output (posterior): an estimate of expected returns based on views and market equilibrium,
# to be input to a Markowitz Portfolio Mean-Variance Optimizer
# Inputs (priors):
# risk_aversion_factor: = (mkt-rf)/sigma_mkt^2
# cov_matrix: the covariance matrix of asset growth data
# view_portfolios: k x n matrix, where k = num views and n = num assets,
#   each row is a view on an asset that adds up to 0 or 1
#   views on the diagonal are Absolute (asset returns %)
#   views not on the diagonal are Relative (row asset will outperfrom col asset by %), usually = 1
# expected_returns_each_view: a column vector of expected returns for each view-portfolio, 
#   e.g. np.array([0.1, 0.5, 0.3]).transpose() refers to an 0.1 change in asset 1, etc.
# levels_of_unconfidence: a column vector of standard deviations for expected returns
#   assuming normal distribution: 0.68 probability realization is in interval
# weight_on_views: a constant for the weight on the view and equilibrium portfolios,
#   conflicting literature: some suggest close to 1, others close to 0
# market_equilibrium_portfolio_weights: weights of equilibrium portfolio based on market views
def calc_black_litterman_expected_returns(cov_matrix, view_portfolios, expected_returns_each_view, 
                          levels_of_unconfidence, weight_on_views, equilibrium_expected_returns):
    # Set up Black-Litterman notation
    sigma = cov_matrix
    p = pd.DataFrame(view_portfolios)
    q_hat = np.array(expected_returns_each_view)
    omega = np.diag(levels_of_unconfidence**2)
    tau = weight_on_views
    pi = equilibrium_expected_returns
    
    # Black-Litterman derived formula
    mu_star = np.linalg.inv(pd.DataFrame(np.linalg.inv(tau*sigma)) + p.transpose().dot(np.linalg.inv(omega)).dot(p)).dot(
              (np.linalg.inv(tau*sigma).dot(pi) + p.transpose().dot(np.linalg.inv(omega)).dot(q_hat)))
    return mu_star

# Calculates the Black-Litterman optimal portfolio
def calc_black_litterman_optimal_portfolio_weights(black_litterman_expected_returns, risk_aversion_factor, cov_matrix):
    
    # Set up Black-Litterman notation
    mu_star = black_litterman_expected_returns
    delta = risk_aversion_factor
    sigma = cov_matrix
    
    # Black-Litterman derived formula
    w_star = np.linalg.inv(delta * sigma).dot(mu_star)
    w_star = w_star/sum(np.abs(w_star))
    return w_star
    
#=============================== Example Implementation ==============================================================
# Calculates the Black-Litterman estimated expected returns and optimal portfolio weights
# with a 7 year lookback from today.
# Literature recommends 7 years to account for market uncertainties
if __name__ == '__main__':
    start_time = time.time()
    
    # Extract Mkt-RF from Fama-French data
    mkt_rf_data = pd.read_csv(mkt_rf_data_file_source, skiprows=3, index_col = 0)
    mkt_rf_data.index = pd.to_numeric(mkt_rf_data.index, errors='coerce', downcast = 'integer')
    mkt_rf_data = mkt_rf_data.dropna()
    mkt_rf_data = mkt_rf_data[mkt_rf_data.index > 100000] # only monthly data
    mkt_rf_data = mkt_rf_data[np.logical_and((mkt_rf_data.index >= 201301),(mkt_rf_data.index <= 201912))]
    mkt_rf_data = mkt_rf_data.apply(pd.to_numeric, errors='coerce')/100
    
    # Read in market data with 7 year lookback
    example_assets = ['AAPL','ADBE','GE','IBM','LLY', 'MSFT', 'SNE','T']
    today = dt.date(2020, 9, 24) # When I last ran the script
    # today = dt.date.today() # May require changing parameters/tuning
    seven_years_ago = today - dt.timedelta(days=7*365) # An approximation - doesn't account for leap years
    assets_data = read_in_assets_data(example_assets, seven_years_ago, today, 
                                      True, directory + '/Data/2013to2019_assets_data.csv')
    
    # Calculates inputs to the Black-Litterman model
    market_equilibrium_portfolio_weights = find_market_equilibrium_weights(example_assets)
    assets_growth_data = convert_to_log_growth(assets_data)
    mean_cov_matrix_and_size = calc_mean_cov_matrix_and_size(assets_growth_data)
    markowitz_estimated_expected_returns, input_cov_matrix, num_days = mean_cov_matrix_and_size
    input_risk_aversion_factor = calculate_risk_aversion_factor(mkt_rf_data['Mkt-RF'])
    # In practice, some these parameters would require tuning against a benchmark portfolio
    # Using arbitrary values with 4 views, in practice, would require domain experts
    input_view_portfolios = np.matrix([[0,   0,   0, 0.5, 0,    0, -0.1, 0  ],
                                       [0,   0.6, 0, 0,  -0.2,  0,  0,   0  ],
                                       [0.1, 0,   0, 0.2, 0,    0,  0,  -0.2],
                                       [0,   0,   0, 0,   0,    0,  1,   0  ]])
    input_expected_returns_each_view = np.array([0.2 , 0.1 , 0.3 , 0.1]).transpose()
    input_levels_of_unconfidence = np.array(    [0.05, 0.02, 0.15, 0.06])
    input_weight_on_views = 0.01 # conflicting literature on weight_on_views
    input_equilibrium_expected_returns = reverse_optimization(input_risk_aversion_factor, input_cov_matrix, 
                                                                      market_equilibrium_portfolio_weights)
    
    # Calculate Black-Litterman model values
    black_litterman_estimated_expected_returns = calc_black_litterman_expected_returns(input_cov_matrix, input_view_portfolios, 
                                                                                       input_expected_returns_each_view, 
                                                                                       input_levels_of_unconfidence, input_weight_on_views, 
                                                                                       input_equilibrium_expected_returns)
    black_litterman_optimal_portfolio_weights = calc_black_litterman_optimal_portfolio_weights(black_litterman_estimated_expected_returns, 
                                                                                               input_risk_aversion_factor, input_cov_matrix)
    # Run through mean-variance optimizer and plot
    black_litterman_efficient_frontier_daily_points = calc_markowitz_efficient_frontier(black_litterman_estimated_expected_returns, input_cov_matrix, False)
    black_litterman_efficient_frontier_daily_points.to_csv(directory + '/Results/black_litterman_efficient_frontier_daily_returns_2013to2019_values.csv')
    plot_filename = directory + '/Graphs/black_litterman_efficient_frontier_daily_returns_2013to2019_graph.png'
    plot_efficient_frontier(black_litterman_efficient_frontier_daily_points, plot_filename)
    
    # Convert to annual and plot
    black_litterman_efficient_frontier_annual_points = convert_to_annual_returns(black_litterman_efficient_frontier_daily_points)
    black_litterman_efficient_frontier_annual_points.to_csv(directory + '/Results/black_litterman_efficient_frontier_annual_returns_2013to2019_values.csv')
    plot_filename = directory + '/Graphs/black_litterman_efficient_frontier_annual_returns_2013to2019_graph.png'
    plot_efficient_frontier(black_litterman_efficient_frontier_annual_points, plot_filename)
    
    print('\nBlack-Litterman Optimal Weights: ' + str(black_litterman_optimal_portfolio_weights))
    
    print('Program Run Time: ' + str(round(time.time() - start_time)) + 's')






