# Author: Nolan Alexander

import pandas as pd
import numpy as np

# Converts a set of daily expected returns and standard deviations into annual
def convert_to_annual_returns(ef_expected_returns_and_sds_df):
    num_days = 253 # Average number of working days in a year
    ef_expected_returns = ef_expected_returns_and_sds_df['Expected Returns']
    ef_variances = ef_expected_returns_and_sds_df['Standard Deviations']**2
    
    # Using geometric compounding to calculate expected returns and variance
    # variances calculated assuming independence between each day's return
    # and using product distributions
    for i in range(len(ef_variances)):
        ef_variances[i] = ((ef_variances[i] + (1+ef_expected_returns[i])**2)**num_days 
                          - ((1+ef_expected_returns[i])**2)**num_days)
    ef_expected_returns = (1+ef_expected_returns)**num_days-1
    
    # Format data to return
    ef_stdevs = np.sqrt(ef_variances)
    ef_annual_points = pd.DataFrame({'Expected Returns': ef_expected_returns, 
                                     'Standard Deviations': ef_stdevs})
    return ef_annual_points
