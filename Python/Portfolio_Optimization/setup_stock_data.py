# Author: Nolan Alexander

import numpy as np
import warnings

# convert sets of time series to log growth 
def convert_to_log_growth(assets_data):
    warnings.filterwarnings("ignore") # ignores the warning for the ln(0)
    assets_growth_data = np.log(assets_data) - np.log(assets_data.shift(1)) # log growth
    assets_growth_data.drop(assets_growth_data.head(1).index,inplace=True)
    return assets_growth_data
# Calculates the stock growth means, covariance matrix, number of assets and number of days
def calc_mean_cov_matrix_and_size(assets_growth_data):
    assets_growth_means = assets_growth_data.mean() # compounded daily
    cov_matrix = assets_growth_data.cov()
    num_days = len(assets_growth_data.index)
    return [assets_growth_means, cov_matrix, num_days]