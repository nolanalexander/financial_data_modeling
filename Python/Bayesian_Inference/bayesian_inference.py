# Author: Nolan Alexander

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import datetime as dt
import time
import sys
import pymc3 as pm # pip install pymc3

dir_to_python = 'Path/To/The/Code'
directory = dir_to_python + '/Python/Portfolio_Optimization'
sys.path.append(directory)
from assets_data_collection import read_in_assets_data
from setup_stock_data import convert_to_log_growth
directory = dir_to_python + '/Python/Bayesian_Inference'

# Performs Bayesian Inference with Markov-Chain Monte-Carlo using the No U-Turn Sampling 
# algorithm to estimate the posterior distribution of Goldman Sachs stock data 
# based on todays data with a lookback of seven years.  

# See the bayesian_model.png file for the stochastic model layout/assumptions. The stochastic model was 
# designed to account for some aspects of the returns but a truely accurate model would be far more 
# complex. The purpose of this is just to demonstrate how to perform bayesian inference with pymc3.
start_time = time.time()
debug = False

# Read in market data with 7 year lookback
example_assets = ['GS']
today = dt.date(2020, 9, 24) # When I last ran the script
# today = dt.date.today() # May require changing parameters/tuning
seven_years_ago = today - dt.timedelta(days=7*365) # An approximation - doesn't account for leap years
asset_data = read_in_assets_data(example_assets, seven_years_ago, today, 
                                  True, directory + '/Data/gs_time_series_7_years.csv')
asset_growth_data = convert_to_log_growth(asset_data)[example_assets[0]]

# Visualize returns
asset_mu = np.mean(asset_growth_data) 
print("Average GS growth over the past 7 years: " + str(asset_mu))
plt.plot(asset_growth_data)
plt.title('GS Growth Over the Past 7 years')
plt.xlabel('Time')
plt.ylabel('Variance')
plot_filename = directory + '/Graphs/gs_growth_data.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()

# Distribution fitting to find the prior distribution

# Transforms data to become larger scale and right-skewed
# because some skewed distributions (e.g. lognormal) are only right-skewed
y = -asset_growth_data * 100
plot_abs_min_max = max(max(y), abs(min(y)))
plt.hist(y, bins=50, density = True)
plt.xlabel('Negative Growth Scaled by 100')
plt.ylabel('Density')

# The histogram shows that the distribution is right-skewed (after transformation) 
# with a high kurtosis
# Hypothesized prior distributions: norm as baseline, then selected common 
# distributions that are skewed and/or have high kurtosis
dist_names = ['norm', 'lognorm', 'gamma', 'beta', 'cauchy', 'invgamma', 
              'laplace', 'exponnorm', 'logistic', 'loggamma']
dist_fits = pd.DataFrame(np.empty((len(dist_names),2)), columns = ['SSE', 'MAD'])
dist_fits.index = dist_names

# Fit all distributions, plot, and compute fit metrics
for dist_name in dist_names:
    dist = getattr(sp.stats, dist_name)
    param = dist.fit(y)
    x = np.linspace(-plot_abs_min_max, plot_abs_min_max, len(y))
    pdf_fitted = dist.pdf(x, loc=param[-2], scale=param[-1], *param[:-2])
    plt.plot(x, pdf_fitted, label=dist_name)
    plt.xlim(-plot_abs_min_max, plot_abs_min_max)
    sse = sum((y - pdf_fitted) ** 2)
    dist_fits.at[dist_name, 'SSE'] = sse
    mad = max(abs(y - pdf_fitted))
    dist_fits.at[dist_name, 'MAD'] = mad
plt.legend(loc='upper right')
plot_filename = directory + '/Graphs/prior_distribution_fitting.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print(dist_fits)

# Based on the plots and SSE and MAD scores, the alpha (stable) distribution 
# seems to fit the empirical data the best.
logistic_dist = getattr(sp.stats, 'logistic')
logistic_param = logistic_dist.fit(y) # mu, shape
# Logistic distribution does not account for skew, so we will fit a lognormal distribution also.
lognorm_dist = getattr(sp.stats, 'lognorm')
lognorm_param = lognorm_dist.fit(y) # shape, loc, scale

# Find the SD of SD each week
sd = [0] * len(y)
for i in range(0,len(y)-1,5):
    sd[i:i+5] = [np.std(y[i:i+5])] * 5
sd_sd = np.std(sd)

# The Ergotic Theorem for Markov Chains allows us to find the posterior distribution
# by simulating a large sample size with  Monte Carlo
with pm.Model() as gs_model:
    
    # Assume returns follow a log-normal distribution, 
    # common assumption for stock returns because it can account for the skew
    # semi-informed with SD of fitted lognorm likelihood estimation
    mu = pm.Lognormal('mu', sigma=lognorm_param[0])
    
    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
    sigma = PositiveNormal('sigma', mu=np.std(y), sigma=sd_sd)
    
    # Assume prior returns follows a gaussian random walk because stock returns are nonstationary
    # so this helps models the stochastic process
    # semi-informed with SD likelihood estimation
    returns = pm.GaussianRandomWalk('returns', mu=mu, sigma=sigma, shape=len(y))
    
    # Assume shape follows a positive normal distribution centered around the prior shape
    # semi-informed with SD equal to the 0.5
    shape = PositiveNormal('shape', mu=logistic_param[1], sd=0.5)
    
    # Likelihood function of observed data follows a logistic distribution based on distribution fitting
    obs = pm.Logistic('obs', mu=returns, s=shape, observed=y)
    
    # MCMC sampling methods
    # step = pm.Metropolis() # Metropolis-Hastings is often not as accurate, but is fast
    step = pm.NUTS() # No U-turn Sampler is more accurate, but takes longer
    trace = pm.sample(10000, step=step)
    
    # Traceplot
    pm.traceplot(trace)
    plot_filename = directory + '/Graphs/trace_plot.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

# Plot posterior distribution
burn_in = 500 # Use burn-in to account for sampling to get to a proper state
plt.hist(-trace[burn_in:]['mu']/100, bins=50) # transforms back to left skew by multiplying by -1, removes scale
plt.title('Growth Posterior Distribution')
plt.xlabel('Growth')
plt.ylabel('Trace Count')
plot_filename = directory + '/Graphs/posterior_distribution.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()

print('Program Run Time: ' + str(round(time.time() - start_time)) + 's')

# For debugging invalid parameter values or those that approach infinity, find log-posterior
if(debug):
    for RV in gs_model.basic_RVs:
        print(RV.name, RV.logp(gs_model.test_point))


