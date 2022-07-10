# Görev 5:Argüman olarak bir liste alan,
# listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return
# eden fonksiyon yazınız.Liste elemanlarına tek tek erişmeniz gerekmektedir.

list = [2, 13, 18, 93, 22]


def func(list):
    even_list= []
    odd_list= []
    for i in list:
        if i % 2 ==0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

#İlk başta fonksiyon yazdık ardından fonksiyon içerisinde 2 ayrı liste daha oluşturduk bu listelere 
#atamaları gerçekleştireceğiz. ardından for döngüsü oluşturduk asıl listenin içerisinde gezebilmek için
#sonrasında koşul cümlelerimizi yazdık ve return ettik print ile ekrana basılmasını sağladık.

odd_list, even_list = func(list)
print(odd_list, even_list)