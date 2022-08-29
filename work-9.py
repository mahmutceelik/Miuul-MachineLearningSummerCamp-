###Fonksiyonlara Özellik ve Docstring Ekleme

#Görev 1: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argümanla biçimlendirilebilir olmalı.
#Var olan özelliğide argümanla kontroledilebilir hale getirebilirsiniz.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name:df[col_name].value.counts,"Ratio":100 * dataframe[col_name].valuecounts()/ len(dataframe)}))
    print["#########"]
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

cat_summary(df,"survived",True)

#Görev check_df(), cat_summary() fonksiyonlarına 4 bilgi barındıran numpy tarzı docstring yazınız.

def cat_summary(dataframe, col_name, plot= False):
    """
    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    col_name : str
        kolon ismine string tip ile girmeliyiz.
    plot : bool
        Grafik çizdirmek amacımıza uygun istiyorsak True / istemiyorsak False değer vermeliyiz.
    Returns
    -------
    """
        print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")
    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show()

cat_summary(df,"survived", True)

# check_df() fonksiyonumuz
def check_df(dataframe, head=5):
    """
    Parameters
    ----------
    dataframe :dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    head : int
        Veri setinin başından başlayarak istenilen satır sayısıdır.
    Returns
    -------
    """
    print("###### shape ####")
    print(dataframe.shape)
    print("###### types ####")
    print(dataframe.dtypes)
    print("###### head ####")
    print(dataframe.head(head))
    print("###### tail ####")
    print(dataframe.tail(head))
    print("###### eksik değerler ####")
    print(dataframe.isnull().sum())
    print("###### quantiles  ####")
    #print(dataframe.describe([0, 0.5, 0.50, 0.95, 0.99, 1]).T)

check_df(df)