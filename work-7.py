# Görev 7:  List Comprehension yapısı kullanarak car_crashes verisinde
# isminde"no" barındırmayan değişkenlerin isimlerinin sonuna"FLAG" yazınız.

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns
df.columns = [col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]
df.columns

# Out = Index(['TOTAL_FLAG', 'SPEEDING_FLAG', 'ALCOHOL_FLAG', 'NOT_DISTRACTED',
#       'NO_PREVIOUS', 'INS_PREMIUM_FLAG', 'INS_LOSSES_FLAG', 'ABBREV_FLAG'],
#     dtype='object')
