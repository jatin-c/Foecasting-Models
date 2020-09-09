#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[2]:


def auto_hwm(timeseries, val_split_date, alpha=[None], beta=[None], gamma=[None], phi=[None], 
              trend=None, seasonal=None, periods=None, verbose=False):

    '''The auto_hwm (short for auto holt winters model) function to search for the best possible parameter
        combination for the Exponential Smoothing model i.e. smoothing level, smoothing slope, 
        smoothing seasonal and damping slope based on mean absolute error.

        ****Paramters****

        timeseries: array-like

                  Time-Series

        val_split_date: str

                  The datetime to split the time-series for validation

        alpha: list of floats (optional)

                  The list of alpha values for the simple exponential smoothing parameter

        beta: list of floats (optional)

                  The list of beta values for the Holt’s trend method parameter

        gamma: list of floats (optional)

                  The list of gamma values for the holt winters seasonal method parameter

        phi: list of floats (optional)

                  The list of phi values for the damped method parameter

        trend: {“add”, “mul”, “additive”, “multiplicative”, None} (optional)

                  Type of trend component.

        seasonal: {“add”, “mul”, “additive”, “multiplicative”, None} (optional)
                  
                  Type of seasonal component.

        periods: int (optional)
                  
                  The number of periods in a complete seasonal cycle

        ****Returns****

        best_params: dict

                  The values of alpha, beta, gamma and phi for which the 
                  validation data (val_split_date) gives the least mean absolute error
    '''

    best_params = []
    actual = timeseries[val_split_date:]

    print('Evaluating Exponential Smoothing model for', len(alpha) * len(beta) * len(gamma) * len(phi), 'fits\n')

    for a in alpha:
        for b in beta:
            for g in gamma:
                for p in phi:

                    if(verbose == True):
                        print('Checking for', {'alpha': a, 'beta': b, 'gamma': g, 'phi': p})

                    model = ExponentialSmoothing(timeseries, trend=trend, seasonal=seasonal, seasonal_periods=periods)
                    model.fit(smoothing_level=a, smoothing_slope=b, smoothing_seasonal=g, damping_slope=p)
                    f_cast = model.predict(model.params, start=actual.index[0])
                    score = np.float64(mean_absolute_error(actual, f_cast)/actual).mean()

                    best_params.append({'alpha': a, 'beta': b, 'gamma': g, 'phi': p, 'mae': score})

    return min(best_params, key=lambda x: x['mae'])


# In[ ]:




