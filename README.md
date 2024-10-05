# Adidas Sales Prediction Project

This project aims to predict future sales of the Adidas brand using quarterly sales data. Time series analysis and SARIMA modeling are used to make sales forecasts.

## Contents

1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Project Steps](#project-steps)
4. [Results](#results)
5. [COVID-19 Impact](#covid-19-impact)

## Requirements

To run this project, you need to install the following Python libraries:

```
pandas
numpy
matplotlib
seaborn
statsmodels
plotly
```

You can install these libraries using pip:

```
pip install pandas numpy matplotlib seaborn statsmodels plotly
```

## Dataset

The project uses a CSV file named 'adidas-quarterly-sales.csv'. This file contains Adidas' quarterly sales data.

## Project Steps

1. Data Loading and Preprocessing
2. Data Visualization
3. Time Series Analysis
4. SARIMA Model Creation
5. Future Sales Prediction
6. Analysis of COVID-19 Impact

### 1. Data Loading and Preprocessing

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('adidas-quarterly-sales.csv')
```

### 2. Data Visualization

```python
import plotly.express as px
fig = px.line(df, x='Time Period', y='Revenue')
fig.show()
```

### 3. Time Series Analysis

```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Revenue'], model='multiplicative', period=30)
fig = result.plot()
fig.set_size_inches(15, 10)
fig.show()
```

### 4. SARIMA Model Creation

```python
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(df['Revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model = model.fit()
print(model.summary())
```

### 5. Future Sales Prediction

```python
y_pred = model.predict(start=len(df), end=len(df)+20)
df['Revenue'].plot()
y_pred.plot()
plt.show()
```

### 6. Analysis of COVID-19 Impact

```python
df_without_corona = pd.DataFrame(df['Revenue'][:75], columns=['Revenue'])
model = sm.tsa.statespace.SARIMAX(df_without_corona['Revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model = model.fit()
predictions = model.predict(start=len(df_without_corona), end=len(df_without_corona)+20)
df_without_corona.plot()
predictions.plot()
plt.show()
```

## Results

The project analyzes Adidas' sales data and provides future sales predictions. The predictions made using the SARIMA model capture the general sales trend and seasonal variations.

### SARIMA Model Results

The SARIMA(1,1,1)x(1,1,1,12) model yielded the following results:

```
                                     SARIMAX Results                                      
==========================================================================================
Dep. Variable:                            Revenue   No. Observations:                   88
Model:             SARIMAX(1, 1, 1)x(1, 1, 1, 12)   Log Likelihood                -548.282
Date:                            Sat, 08 Jun 2024   AIC                           1106.564
Time:                                    23:21:52   BIC                           1118.152
Sample:                                         0   HQIC                          1111.191
                                             - 88                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.7176      0.053     13.636      0.000       0.614       0.821
ma.L1         -0.9981      0.545     -1.833      0.067      -2.065       0.069
ar.S.L12      -0.5921      0.422     -1.404      0.160      -1.419       0.235
ma.S.L12      -0.2022      0.552     -0.366      0.714      -1.285       0.880
sigma2      1.154e+05   6.61e+04      1.747      0.081   -1.41e+04    2.45e+05
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):               180.79
Prob(Q):                              0.93   Prob(JB):                         0.00
Heteroskedasticity (H):               8.54   Skew:                            -1.12
Prob(H) (two-sided):                  0.00   Kurtosis:                        10.27
===================================================================================
```

#### Key Observations:

1. The model is based on 88 observations of quarterly revenue data.
2. The Log Likelihood of the model is -548.282, with AIC, BIC, and HQIC values of 1106.564, 1118.152, and 1111.191 respectively.
3. The AR(1) term (ar.L1) is statistically significant (p-value < 0.05), indicating a strong autoregressive component in the time series.
4. The MA(1) term (ma.L1) and seasonal components (ar.S.L12, ma.S.L12) are not statistically significant at the 5% level.
5. The Ljung-Box test (Q-statistic) suggests that the model residuals are independently distributed (p-value = 0.93).
6. The Jarque-Bera test indicates that the residuals are not normally distributed (p-value < 0.05).
7. There is evidence of heteroskedasticity in the residuals (Heteroskedasticity test p-value < 0.05).
8. The residuals show negative skewness (-1.12) and high kurtosis (10.27), indicating a heavy-tailed distribution.

These results suggest that while the model captures some of the time series patterns, there might be room for improvement, particularly in addressing the non-normality and heteroskedasticity of the residuals.



## COVID-19 Impact

In the final section of the project, a prediction was made using pre-pandemic data to analyze the impact of the COVID-19 pandemic. This analysis visualizes the potential effect of the pandemic on Adidas sales.

