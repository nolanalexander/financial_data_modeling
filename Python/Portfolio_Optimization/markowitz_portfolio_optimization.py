# Author: Nolan Alexander

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime as dt
import time
import sys

dir_to_python = 'Path/To/The/Code'
directory = dir_to_python + '/Python/Portfolio_Optimization'
sys.path.append(directory)
from assets_data_collection import read_in_assets_data
from setup_stock_data import convert_to_log_growth, calc_mean_cov_matrix_and_size
from annual_conversion import convert_to_annual_returns


# Implementation of the Markowtiz Portfolio Model to solve for the Efficient Frontier
# by performing mean-variance portfolio optimization
# shorting_allowed controls the lower bound, 0 or -1
def calc_markowitz_efficient_frontier(stock_growth_means, cov_matrix, shorting_allowed):
    debug = False
    num_assets = len(stock_growth_means)
    #Set up optimization variables
    weights = pd.DataFrame({'Weights': np.zeros(num_assets)}, index=cov_matrix.index)
    def variance_daily(input_weights):
        return input_weights.transpose().dot(cov_matrix).dot(input_weights)
    def expected_return_daily(input_weights):
        return stock_growth_means.transpose().dot(input_weights)
    def neg_expected_return_daily(input_weights):
        return -1*expected_return_daily(input_weights)

    # Set up optimization constraints
    cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    upper_bound = 1.0
    if shorting_allowed: lower_bound = -1.0
    else: lower_bound = 0.0

    # Runs optimization to find the minimum variance point
    optimization = minimize(variance_daily, weights, method='SLSQP' ,
                   bounds = ((lower_bound, upper_bound),)*num_assets ,
                   options = {'disp':False, 'ftol': 1e-20, 'maxiter': 1000} ,
                   constraints=cons)
    min_var_point_variance = optimization.fun
    min_var_point_expected_return = stock_growth_means.transpose().dot(optimization.x)
    
    # Runs optimization to find the maximum expected return point
    optimization = minimize(neg_expected_return_daily, weights, method='SLSQP' ,
                   bounds = ((lower_bound, upper_bound),)*num_assets ,
                   options = {'disp':False, 'ftol': 1e-20, 'maxiter': 1000} ,
                   constraints=cons)
    max_er_point_expected_return = -optimization.fun # neg because we are maximizing
    max_er_point_variance = optimization.x.transpose().dot(cov_matrix).dot(optimization.x)

    # Finds points on efficient frontier
    # Creates num_ef_points evenly spaced expected return points
    num_ef_points = 50 # Can adjust to change granularity of efficient frontier
    ef_expected_returns = np.array([])
    for i in range(num_ef_points):
        er = (min_var_point_expected_return+(max_er_point_expected_return-num_ef_points*
                                            min_var_point_expected_return)/(num_ef_points-1))*(i+1)
        er = er-((max_er_point_expected_return-num_ef_points*min_var_point_expected_return)/(num_ef_points-1))
        ef_expected_returns = np.append(ef_expected_returns, float(er))

    ef_weights = pd.DataFrame(columns=list(cov_matrix));
    ef_variances = np.array([])
    ef_variances = np.append(ef_variances, min_var_point_variance)
    
    # Runs optimization to find the variance for each expected return
    for i in range(num_ef_points-2):
        cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
        cons.append({'type':'eq', 'fun': lambda x: 
            ef_expected_returns[i+1]-stock_growth_means.transpose().dot(x)})
        optimization = minimize(variance_daily, weights, method='SLSQP' ,
                       bounds = ((lower_bound, upper_bound),)*num_assets ,
                       options = {'disp':False, 'ftol': 1e-20, 'maxiter': 1000} ,
                       constraints=cons)
        ef_weights.loc[i+1] = optimization.x
        ef_variances = np.append(ef_variances, float(optimization.fun))
    ef_variances = np.append(ef_variances,max_er_point_variance)

    # Format data to return
    ef_stdevs = np.sqrt(ef_variances)
    ef_points = pd.DataFrame({'Expected Returns': ef_expected_returns, 
                              'Standard Deviations': ef_stdevs})

    if(debug):
        print('====================================================================')
        print(optimization)
        print('min_var_point_variance:\t      ', min_var_point_variance)
        print('min_var_point_expected_return:', min_var_point_expected_return)
        print('max_er_point_variance:\t     ', max_er_point_variance)
        print('max_er_point_expected_return:', max_er_point_expected_return)
    return ef_points

# Plots a Markowitz Efficient Frontier
def plot_efficient_frontier(ef_expected_returns_and_sds_df, plot_filename):
    ef_expected_returns = ef_expected_returns_and_sds_df['Expected Returns']
    ef_stdevs = ef_expected_returns_and_sds_df['Standard Deviations']
    plt.plot(ef_stdevs,ef_expected_returns)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Expected Return")
    plt.title("Markowitz Efficient Frontier")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print('The efficient frontier plot has been saved to your Plot folder')


#=============================== Example Implementation ==============================================================
# Calculates the Markowitz Efficient Frontier with data from today 
# with a 7 year lookback.
# Literature recommends 7 years to account for market uncertainties
if __name__ == '__main__':
    start_time = time.time()
    
    # Read in market data with 7 year lookback
    example_assets = ['AAPL','ADBE','GE','IBM','LLY', 'MSFT', 'SNE','T']
    today = dt.date(2020, 9, 24) # When I last ran the script
    # today = dt.date.today() # May require changing parameters/tuning
    seven_years_ago = today - dt.timedelta(days=7*365) # An approximation - doesn't account for leap years
    assets_data = read_in_assets_data(example_assets, seven_years_ago, today, 
                                      True, directory + '/Data/2013to2019_assets_data.csv')
    
    # Perform Markowitz portfolio optimization
    assets_growth_data = convert_to_log_growth(assets_data)
    mean_cov_matrix_and_size = calc_mean_cov_matrix_and_size(assets_growth_data)
    stock_growth_means, cov_matrix, num_days = mean_cov_matrix_and_size
    efficient_frontier_daily_points = calc_markowitz_efficient_frontier(stock_growth_means, cov_matrix,
                                                                  shorting_allowed = False)
    # Save efficient frontier points and plot them
    efficient_frontier_daily_points.to_csv(directory + '/Results/markowitz_efficient_frontier_daily_returns_2013to2019_values.csv')
    plot_filename = directory + '/Graphs/markowitz_efficient_frontier_daily_returns_2013to2019_graph.png'
    plot_efficient_frontier(efficient_frontier_daily_points, plot_filename)
    
    # Convert to annual and plot
    efficient_frontier_annual_points = convert_to_annual_returns(efficient_frontier_daily_points)
    efficient_frontier_annual_points.to_csv(directory + '/Results/markowitz_efficient_frontier_annual_returns_2013to2019_values.csv')
    plot_filename = directory + '/Graphs/markowitz_efficient_frontier_annual_returns_2013to2019_graph.png'
    plot_efficient_frontier(efficient_frontier_annual_points, plot_filename)
    
    print('Program Run Time: ' + str(round(time.time() - start_time)) + 's')
