# Author: Nolan Alexander

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
import datetime as dt
import time
import sys

dir_to_python = 'Path/To/The/Code'
directory = dir_to_python + '/Python/Portfolio_Optimization'
sys.path.append(directory)
from assets_data_collection import read_in_assets_data
from setup_stock_data import convert_to_log_growth, calc_mean_cov_matrix_and_size


# Uses hierarchical clustering to find assets that are similar and different to 
# each other based on data from the covariance matrix of the assets
start_time = time.time()

# Read in market data with 7 year lookback from today
example_assets = ['AAPL','H', 'Y', 'SNE', 'GS', 'K', 'NKE','LEVI'] # Stocks from different sectors of the market
today = dt.date(2020, 9, 24) # When I last ran the script
# today = dt.date.today() # May require changing parameters/tuning
seven_years_ago = today - dt.timedelta(days=7*365) # An approximation - doesn't account for leap years
assets_data = read_in_assets_data(example_assets, seven_years_ago, today, 
                                  True, directory + '/Data/2013to2019_assets_data_for_clustering.csv')
assets_growth_data = convert_to_log_growth(assets_data)
mean_cov_matrix_and_size = calc_mean_cov_matrix_and_size(assets_growth_data)
stock_growth_means, cov_matrix, num_days = mean_cov_matrix_and_size

# Splitting and preprocessing
scaler = StandardScaler()
scaled_cov_matrix = pd.DataFrame(scaler.fit_transform(cov_matrix), columns = cov_matrix.columns)

# Hierarchy Visualization with Dendrogram
plt.figure(figsize=(10,7))  
plt.title("Assets Dendrogram")  
plt.xlabel('Assets')
plt.ylabel('Distance')
dend = shc.dendrogram(shc.linkage(scaled_cov_matrix, method='ward'))
cutoff_point = 6 # Select a threshold of 5 to create 2 clusters (may change depending on when you run the progra)
plt.axhline(y=cutoff_point, color='r', linestyle='--')
plot_filename = directory + '/Graphs/assets_dendrogram.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The dendrogram plot has been saved to your Plot folder')

# Fit agglomerative bottom-up model
model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
predictions = model.fit_predict(scaled_cov_matrix)
predictions_names = scaled_cov_matrix.columns

# Visualize Predictions (takes a while to create)
fig, ax = plt.subplots(len(example_assets), len(example_assets), figsize=(10,10))
for i in range(len(example_assets)):
    for j in range(len(example_assets)):
        ax[i, j].scatter(scaled_cov_matrix[example_assets[i]], 
          scaled_cov_matrix[example_assets[j]], c=model.labels_)
        ax[i, j].label_outer()
for i in range(len(example_assets)):
    ax[i, 0].set_ylabel(example_assets[i])
    ax[len(example_assets)-1, i].set_xlabel(example_assets[i])
plot_filename = directory + '/Graphs/assets_scatterplot_matrices_clustered.png'
fig.savefig(plot_filename, bbox_inches='tight')
plt.close()
print('The scatterplot matrix plot has been saved to your Plot folder')

# Measure model performance using silhouette score [-1,1], 
# where closer to 1, the clusters are more dense
sil_score = silhouette_score(scaled_cov_matrix, predictions)
print('The silhouette score is: ' + str(sil_score))

print('Program Run Time: ' + str(round(time.time() - start_time)) + 's')




