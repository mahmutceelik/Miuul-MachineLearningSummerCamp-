# Görev 4:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# Adım1: Key değerlerine erişiniz.
# Adım2: Value'lara erişiniz.
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım4: Key değeri Ahmet value değeri[Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5: Antonio'yu dictionary'den siliniz.

dict = {"Christian":["America",18],"Daisy":["England",12],"Antonio":["Spain",22],"Dante":["Italy",25]}
type(dict)
# Out = dict

dict.keys()
# Out = dict_keys(['Christian', 'Daisy', 'Antonio', 'Dante'])

dict.values()
#Out = dict_values([['America', 18], ['England', 12], ['Spain', 22], ['Italy', 25]])

dict["Daisy"][1]=13
dict
#Out = {'Christian': ['America', 18],'Daisy': ['England', 13],'Antonio': ['Spain', 22],'Dante': ['Italy', 25]}

dict.update({"Ahmet":["Turkey",24]})
dict
#Out = {'Christian': ['America', 18],'Daisy': ['England', 13],'Antonio': ['Spain', 22],'Dante': ['Italy', 25],'Ahmet': ['Turkey', 24]}

dict.pop("Antonio")
dict
#Out = {'Christian': ['America', 18],'Daisy': ['England', 13],'Dante': ['Italy', 25],'Ahmet': ['Turkey', 24]}