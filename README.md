# Laporan Proyek Machine Learning
### Nama : Refadli Dwi Ilham
### Nim : 211351121
### Kelas : Pagi B

## Domain Proyek

Estimasi harga ponsel ini boleh digunakan sebagai patokan bagi semua orang yang ingin membeli atau menjual ponsel
## Business Understanding

Lebih menghemat waktu agar tidak perlu menanyakan harga yang cocok untuk menjual atau membeli ponsel

Bagian laporan ini mencakup:

### Problem Statements

- Tidak mungkin seseorang yang ingin menjual atau membeli ponsel harus menanyakan kepada setiap orang yang memiliki ponsel agar tau harga yang pas

### Goals

- mencari solusi untuk memudahkan orang-orang yang mencari harga yang cocok untuk menjual atau membeli ponsel


    ### Solution statements
    - Pengembangan Platform Pencarian Harga yang cocok untuk membeli atau menjual ponsel Berbasis Web, Solusi pertama adalah mengembangkan platform pencarian Harga yang cocok untuk membeli atau menjual ponsel mengintegrasikan data dari Kaggle.com untuk memberikan pengguna akses cepat dan mudah ke informasi tentang estimasi Harga yang cocok untuk membeli atau menjual ponsel
    - Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Dataset yang saya gunakan berasal jadi Kaggle yang berisi Harga yang cocok untuk membeli atau menjual ponsel.Dataset ini mengandung 162 baris dan lebih dari 14 columns.

kaggle datasets download -d mohannapd/mobile-price-prediction 

### Variabel-variabel sebagai berikut:
- Sale  : Penjualan Ponsel
- weight    : Berat Ponsel
- ppi       : Ukuran Resolusi Pada Layar Ponsel
- cpu core  : Processor CPU
- ram       : memori jangka pendek
- Front_Cam : Kamera Depan
- battery   : Penyimpan Daya Listrik
- thickness : Ketebalan Ponsel
- price     : Harga Ponsel

## Data Preparation

- DESKRIPSI LIBRARY
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
- MEMANGGIL DATASET
```python
df= pd.read_csv('/content/drive/MyDrive/ml1/Cellphone.csv')
```
- DESKRIPSI DATASET
```python
df.head()
```
```python
df.info()
```
```python
sns.heatmap(df.isnull())
```
```python
df.describe()
```
- PERUBAHAN TYPE DATA
```python
df['Front_Cam'] = df['Front_Cam'].astype('int64')

df['thickness'] = df['thickness'].astype('int64')

df['ram'] = df['ram'].astype('int64')

df['weight'] = df['weight'].astype('int64')
```
- VISUALISASI DATA
```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
```

```python
Product_id = df.groupby('Product_id').count()[['ram']].sort_values(by='ram',ascending=True).reset_index()
Product_id = Product_id.rename(columns={'ram':'numberOfmobile'})
```
```python
fig = plt.figure(figsize=(16,5))
sns.barplot(x=Product_id['Product_id'],y=Product_id['numberOfmobile'], color="green")
plt.xticks(rotation=50)
```

```python
df['ram'].value_counts().plot(kind='bar')
```
```python
plt.figure(figsize=(15,5))
sns.distplot(df['weight'])
```
```python
plt.figure(figsize=(15,5))
sns.distplot(df['Price'])
```
- SELEKSI FITUR

Menentukan Label dan Attribute
```python
attribute = ['Sale','weight','ppi','cpu core','ram','Front_Cam','battery','thickness']
x = df[attribute]
y = df['Price']
x.shape, y.shape
```
SPILIT DATA TRAINING & DATA TESTING
```python
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```
## Modeling

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
```python
score = lr.score(X_test, y_test)
print('Akurasi Regresi Linear = ', score)
```
Akurasi Regresi Linear =  0.9128914726843869

- MENCOBA MELAKUKAN INPUTAN
```python
input_data = np.array([[10,135,424,8,3,8,2610,7.4]])
prediction = lr.predict(input_data)

print('Estimasi Harga Ponsel',prediction)
```
Estimasi Harga Ponsel [2744.23253243]

dan keluar hasil estimasi harga yang cocok untuk menjual atau membeli ponsel

- selanjutnya kita rubah modelnya menjadi bentuk sav
```python
iimport pickle

filename = 'estimasi_harga_ponsel.sav'
pickle.dump(lr,open(filename,'wb'))
```
## Evaluation
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
```python
score = lr.score(X_test, y_test)
print('Akurasi Regresi Linear = ', score)
```
Akurasi Regresi Linear =  0.9128914726843869

metode statistik yang digunakan untuk menganalisis hubungan antara satu atau lebih variabel independen dan variabel dependen biner, yang digunakan untuk klasifikasi.
## Deployment
https://guadli-app-phone.streamlit.app/
