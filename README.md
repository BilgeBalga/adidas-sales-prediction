# Adidas Satış Tahmini Projesi

Bu proje, Adidas markasına ait çeyrek dönemlik satış verilerini kullanarak gelecekteki satışları tahmin etmeyi amaçlamaktadır. Zaman serisi analizi ve SARIMA modellemesi kullanılarak satış tahminleri yapılmıştır.

## İçerik

1. [Gereksinimler](#gereksinimler)
2. [Veri Seti](#veri-seti)
3. [Proje Adımları](#proje-adımları)
4. [Sonuçlar](#sonuçlar)
5. [COVID-19 Etkisi](#covid-19-etkisi)

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerini kurmanız gerekmektedir:

```
pandas
numpy
matplotlib
seaborn
statsmodels
plotly
```

Bu kütüphaneleri pip kullanarak kurabilirsiniz:

```
pip install pandas numpy matplotlib seaborn statsmodels plotly
```

## Veri Seti

Proje, 'adidas-quarterly-sales.csv' adlı bir CSV dosyasını kullanmaktadır. Bu dosya, Adidas'ın çeyrek dönemlik satış verilerini içermektedir.

## Proje Adımları

1. Veri Yükleme ve Ön İşleme
2. Veri Görselleştirme
3. Zaman Serisi Analizi
4. SARIMA Modeli Oluşturma
5. Gelecek Satış Tahmini
6. COVID-19 Etkisinin Analizi

### 1. Veri Yükleme ve Ön İşleme

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

### 2. Veri Görselleştirme

```python
import plotly.express as px
fig = px.line(df, x='Time Period', y='Revenue')
fig.show()
```

### 3. Zaman Serisi Analizi

```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Revenue'], model='multiplicative', period=30)
fig = result.plot()
fig.set_size_inches(15, 10)
fig.show()
```

### 4. SARIMA Modeli Oluşturma

```python
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(df['Revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model = model.fit()
print(model.summary())
```

### 5. Gelecek Satış Tahmini

```python
y_pred = model.predict(start=len(df), end=len(df)+20)
df['Revenue'].plot()
y_pred.plot()
plt.show()
```

### 6. COVID-19 Etkisinin Analizi

```python
df_without_corona = pd.DataFrame(df['Revenue'][:75], columns=['Revenue'])
model = sm.tsa.statespace.SARIMAX(df_without_corona['Revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model = model.fit()
predictions = model.predict(start=len(df_without_corona), end=len(df_without_corona)+20)
df_without_corona.plot()
predictions.plot()
plt.show()
```

## Sonuçlar

Proje, Adidas'ın satış verilerini analiz ederek gelecekteki satış tahminlerini sunmaktadır. SARIMA modeli kullanılarak yapılan tahminler, genel satış trendini ve mevsimsel değişimleri yakalamaktadır.

## COVID-19 Etkisi

Projenin son bölümünde, COVID-19 pandemisinin etkisini analiz etmek için pandemi öncesi veriler kullanılarak bir tahmin yapılmıştır. Bu analiz, pandeminin Adidas satışları üzerindeki potansiyel etkisini görselleştirmektedir.

---

Bu proje, veri analizi ve zaman serisi tahmini konularında bir örnek çalışma sunmaktadır. Herhangi bir soru veya geri bildiriminiz varsa, lütfen iletişime geçmekten çekinmeyin.
