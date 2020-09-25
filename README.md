# financial_data_modeling

## Purpose
This repository consists of the full code to model certain financial data in Python including all the necessary code to implement these models in practice. 

## From Scratch vs Implementing Specific Packages
The Markowitz Portfolio Optimization, Black-Litterman model, and modules to set up or transform the portfolio parameters were implemented from scratch with the SciPy optimizer. The rest were examples demonstrating the use of specific packages such as pymc3 and arch, but include the full code to allow implementation of these models in practice.

## Repository Structure
The python directory is divided into three subdirectories: Bayesian Inference, Portfolio Optimization, and Time Series Analysis. Each subdirectory contains additional subdirectory: Graphs, to store to visualizations of data or results; and Data, to store data pulled from Yahoo Finance to verify the data input to the models.

### Bayesian Inference 
Contains an example of using pymc3 to forecast the posterior distribution and prior distribution fitting with SciPy.

### Portfolio Optimization
Contains implementations of Markowitz Mean-Variance optimization, the Black-Litterman Model, modules to set up and transform the data, and Hierarchical Clustering using scikit-learn.

### Time Series Analysis 
Contains an example of a GARCH model with a window rolling forecast to measure RMSE using arch.

## Version and Running Code
The code uses Python 3.8.3, pymc3 3.9.3, arch 4.15, and yfinance 0.1.54.  It was developed in Spyder 4.1.4 on a Mac 10.11.6. To run the code, change the dir_to_python variable to the directory on your computer where the python folder is stored, then the program can be run in segments or all at once.
