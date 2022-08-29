import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from datetime import date
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df = pd.read_csv("Hitters.csv")
df.head()


# Bakış
def check_df(dataframe, head=5):
    print("############## Shape ##############")
    print(dataframe.shape)
    
    print("############## Types ##############")
    print(dataframe.dtypes)
    
    print("############## Head ##############")
    print(dataframe.head(head))
    
    print("############## Tail ##############")
    print(dataframe.tail(head))
    
    print("############## NA ##############")
    print(dataframe.isnull().sum())
    
    print("############## Quantiles ##############")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)

num_cols = [col for col in df.columns if col !="Salary" and df[col].dtype in ["int64", "float64"]]
cat_cols = [col for col in df.columns if col !="Salary" and col not in num_cols] # if df[col].dtype in ["object", "bool", "category"]

print("Nümerik kolonlar (Bağımlı değişken hariç): ", num_cols)
print("#"*11)
print("Kategorik kolonlar: ", cat_cols)

# Aykırı değer 

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    filter1 = (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    return dataframe[filter1].any(axis=None)

# Salary bağımlı değişkenimizin ortalaması

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)
    print("#"*20)

