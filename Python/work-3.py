# Görev 3:
# Verilen listeye aşağıdaki adımları uygulayınız.
# Adım1: Verilen listenin eleman sayısına bakınız.
# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım4: Sekizinci indeksteki elemanı siliniz.
# Adım5: Yeni bir eleman ekleyiniz.
# Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.


list = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

len(list)
# Out = 11
print(list[0], list[10])
# D E
new_list= ["D", "A", "T", "A"]
# Out = new_list= ["D", "A", "T", "A"]
list[8]
# Out = "N"
list.pop(8)
# Out = 'N'
list
# Out = ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E']
list.insert(8,"M")
list
# Out = ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'M', 'C', 'E']
list[8]
# Out = 'M'
list.pop(8)
# Out = 'M'
list.insert(8,"N")
#Out = ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E']
