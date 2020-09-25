# Author: Nolan Alexander

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import sys
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from arch import arch_model # pip install arch

dir_to_python = 'Path/To/The/Code'
directory = dir_to_python + '/Python/Portfolio_Optimization'
sys.path.append(directory)
from assets_data_collection import read_in_assets_data
from setup_stock_data import convert_to_log_growth
directory = dir_to_python + '/Python/Time_Series_Analysis'

# Fit GARCH model to Goldman Sachs data with 7 year lookback to predict volatility
# tuning p and q with window rolling forecast on RMSE grid search
start_time = time.time()

# Read in market data with 7 year lookback
example_assets = ['GS']
today = dt.date(2020, 9, 24) # When I last ran the script
# today = dt.date.today() # May require changing parameters/tuning
seven_years_ago = today - dt.timedelta(days=7*365) # An approximation - doesn't account for leap years
asset_data = read_in_assets_data(example_assets, seven_years_ago, today, 
                                  True, directory + '/Data/gs_time_series_7_years.csv')
asset_growth_data = convert_to_log_growth(asset_data)[example_assets[0]]

# Visualize volatility measured every week (5 days)
asset_var = asset_growth_data # To get proper date indices, will overwrite
for i in range(0,len(asset_growth_data)-1,5):
    asset_var[i:i+5] = [np.var(asset_growth_data[i:i+5])] * 5
plt.plot(asset_var)
plt.title('GS Volatility Over the Past 7 years')
plt.xlabel('Time')
plt.ylabel('Variance')
plot_filename = directory + '/Graphs/gs_volatility.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The plot of the variance calculated every 5 days has been saved')
print('The average variance is: ' + str(np.mean(asset_var)))

# Scale by 100 because optimizer fails if the values are too close to 0
# due to gradient of underlying sciPy SLSQP algorithm
asset_growth_data_scaled = asset_growth_data * 100
asset_var_scaled = asset_var * 100**2

# Prepare training, validation, and test sets
# 70-15-15 split
train_size = int(len(asset_growth_data_scaled) * 0.7)
validation_size = int((len(asset_growth_data_scaled)-train_size) * 0.5)
train = asset_growth_data_scaled[0:train_size] 
validation = asset_growth_data_scaled[train_size:(train_size+validation_size)]
test = asset_growth_data_scaled[(train_size+validation_size):]
validation_var_scaled = asset_var_scaled[train_size:(train_size+validation_size)]
test_var_scaled = asset_var_scaled[(train_size+validation_size):]

# Create acf and pacf plot
squared_train = pd.Series(np.array(train)**2) # squared = variance since mean is 0
plot_acf(squared_train) # Helps determine p
plot_filename = directory + '/Graphs/acf_plot.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The acf plot has been saved to your Plot folder')
plot_pacf(squared_train) # Helps determine q
plot_filename = directory + '/Graphs/pacf_plot.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The pacf plot has been saved to your Plot folder')

# Calculates RMSE
def rmse(observed, predictions):
    return np.sqrt(np.mean((observed - predictions)**2))

# Generates forecasts of a GARCH model with a window rolling forecast
def garch_rolling_forecast(train, validation, p, q):
    window = train
    forecasts = pd.Series()
    for i in range(len(validation)):
        model = arch_model(window, p=p, q=q)
        model_fit = model.fit(disp='off')
        yhat = model_fit.forecast(horizon=1, method='simulation').variance.iloc[-1:,0]
        forecasts = forecasts.append(yhat)
        window = window.append(pd.Series(validation[i])) # Shift window forward by adding obs to window
        window.iloc[1:]                                  # and removing first in series
        # print('predicted: ' + str(yhat))
        # print('expected: ' + str(obs))
    return forecasts

# p and q values in the grid, select based on acf and pacf plots
p_vals = range(1, 4+1) # num lag variances, 1-4 based on acf
q_vals = range(1, 11+1) # num residual errors, 1-11 based on pacf

# Grid search of p and q to find the optimal garch parameters to minimize RMSE
lowest_rmse = float("inf")
for cur_p in p_vals:
    for cur_q in q_vals:
        validation_garch_forecasts_scaled = garch_rolling_forecast(train, validation, cur_p, cur_q)
        cur_rmse = rmse(validation_var_scaled.values, validation_garch_forecasts_scaled.values)
        if cur_rmse < lowest_rmse:
            lowest_rmse, optimal_p, optimal_q = cur_rmse, cur_p, cur_q
        print('GARCH(' + str(cur_p) + ',' + str(cur_q) + ') RMSE = ' + str(cur_rmse))

print('Optimal model: GARCH(' + str(optimal_p) + ',' + str(optimal_q) + ') RMSE = ' + str(lowest_rmse))

# It looks like GARCH(1,1) was the best model as other models overfit

# Define and fit model with optimal params, then forecast the test set
model = arch_model(train, mean='Zero', vol='GARCH', 
                   p=optimal_p, q=optimal_q)
model_fit = model.fit()
test_garch_forecasts_scaled = garch_rolling_forecast(train, test, cur_p, cur_q)
test_garch_forecasts_scaled.index = test.index

# Calculate and plot the actual variance from each week
test_var = test_var_scaled/100**2 # rescale back
plt.plot(test_var)
plt.xlabel('Time')
plt.ylabel('Variance')
plot_filename = directory + '/Graphs/gs_volatility_test_set.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The test set plot has been saved to your Plot folder')

# Plot forecast variance
test_garch_forecasts = test_garch_forecasts_scaled/100**2
plt.plot(test_garch_forecasts) # rescale back
plt.xlabel('Time')
plt.ylabel('Variance')
plot_filename = directory + '/Graphs/garch_model_forecast_test_set.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The forecast plot has been saved to your Plot folder')

# It looks like the GARCH model was overestimated the volatility spike around 
# March-April 2020, which was around the time when the COVID-19 lockdown was implemented
# If the GARCH model is run after when I ran it, the volatility spike around the time 
# of the 2020 election can be tested

# Print RMSE
model_rmse = rmse(test_var, test_garch_forecasts) # rescale back
print('\nThe RMSE of the forecast is: ' + str(model_rmse))

print('Program Run Time: ' + str(round(time.time() - start_time)) + 's')



