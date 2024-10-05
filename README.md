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

## COVID-19 Impact

In the final section of the project, a prediction was made using pre-pandemic data to analyze the impact of the COVID-19 pandemic. This analysis visualizes the potential effect of the pandemic on Adidas sales.

---

This project presents a case study on data analysis and time series prediction. If you have any questions or feedback, please don't hesitate to get in touch.
