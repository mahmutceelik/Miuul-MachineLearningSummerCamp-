#Görev 8:  List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden
#farklı olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
new_cols

# Out =  ['total', 'speeding',
# 'alcohol', 'not_distracted', 'ins_premium', 'ins_losses']

new_df = df[new_cols]
new_df.head(5)
#head(5) fonksiyonu dataframe'de 5 satırı görüntüler.

#   total  speeding  alcohol  not_distracted  ins_premium  ins_losses
#0   18.8     7.332    5.640          18.048       784.55      145.08
#1   18.1     7.421    4.525          16.290      1053.48      133.93
#2   18.6     6.510    5.208          15.624       899.47      110.35
#3   22.4     4.032    5.824          21.056       827.34      142.39
#4   12.0     4.200    3.360          10.920       878.41      165.63