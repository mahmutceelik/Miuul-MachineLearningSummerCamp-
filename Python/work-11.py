###### Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("persona.csv")
print(df.head())
   PRICE   SOURCE   SEX COUNTRY  AGE
0     39  android  male     bra   17
1     39  android  male     bra   17
2     49  android  male     bra   17
3     29  android  male     tur   17
4     49  android  male     tur   17

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
2

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()
6

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()
29    1305
39    1260
49    1031
19     992
59     212
9      200
Name: PRICE, dtype: int64

 #  Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
usa    2065
bra    1496
deu     455
tur     451
fra     303
can     230
Name: COUNTRY, dtype: int64

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE":"sum"})
PRICE
COUNTRY	
bra	51354
can	7730
deu	15485
fra	10177
tur	15689
usa	70225

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df.groupby("SOURCE").agg({"PRICE":"count"})
PRICE
SOURCE	
android	2974
ios	2026

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE":"mean"})
PRICE
COUNTRY	
bra	34.327540
can	33.608696
deu	34.032967
fra	33.587459
tur	34.787140
usa	34.007264

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE":"mean"})
PRICE
SOURCE	
android	34.174849
ios	34.069102

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})
PRICE
COUNTRY	SOURCE	
bra	android	34.387029
ios	34.222222
can	android	33.330709
ios	33.951456
deu	android	33.869888
ios	34.268817
fra	android	34.312500
ios	32.776224
tur	android	36.229437
ios	33.272727
usa	android	33.760357
ios	34.371703