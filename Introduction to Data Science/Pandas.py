#dataframe yapılarını kullanmak için oluşturulmuş bir kütüphanedir

# 1) pandas hızlı ve etkili for dataframs
# 2) csv ve text dosyalarını açıp inceleyip sonuçlarımızıda bu doya tiplerine rahat bir şekilde kaydedebiliriz
# 3) pandas missingframede etkilidir
# 4) reshape yapıp datayı daha etkili bir şekilde kullanabiliriz
# 5) slicing ve indexing kolaylığı
# 6) time series data analizinde çok yardımcı

import pandas as pd 

dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren"],
            "AGE":[15,16,17,33,45,66],
            "Maas":[100,150,240,350,110,220]}

dataFrame1 = pd.DataFrame(dictionary)

head = dataFrame1.head()
tail = dataFrame1.tail()

#%% Pandas basic method
print(dataFrame1.columns) #index içeriklerini verir

print(dataFrame1.info())

print(dataFrame1.dtypes) #herbir columnın içeriğini verir

print(dataFrame1.describe()) #sadece nümerik featureleri alır
#%% indexing and slicng

print(dataFrame1["NAME"])
print(dataFrame1.AGE)

dataFrame1["yeni feature"] = [-1,-2,-3,-4,5,6]

print(dataFrame1.loc[:,"AGE"])

print(dataFrame1.loc[:3,"AGE"])

print(dataFrame1.loc[:3,"NAME":"Maas"])

print(dataFrame1.loc[:3,["AGE","NAME"]])

print(dataFrame1.loc[::-1,:])#tame tersi

print(dataFrame1.loc[:,:"Maas"])

print(dataFrame1.loc[:,"Maas"])#locationlar obje alabilir

print(dataFrame1.iloc[:,2])#integer locationlar index alır
#%% filtering 

filtre1 = dataFrame1.Maas > 200

filtrelenmis_Data = dataFrame1[filtre1]

filtre2 = dataFrame1.AGE <20

dataFrame1[filtre1 & filtre2]

dataFrame1[dataFrame1.AGE > 60]
#%% list compheretion 
import numpy as np 
ortalama_maas = dataFrame1.Maas.mean()

#ortalama_maas_np = np.mean(dataFrame1.Maas)

dataFrame1["maas_seviyesi"] = ["dusuk"if ortalama_maas > each else "Yuksek"for each in dataFrame1.Maas]

"""for each in dataFrame1.Maas:
    if(ortalama_maas > each):
        print("yüksek")
    else:
        print("dusuk")"""

dataFrame1.columns = [each.lower() for each in dataFrame1.columns]
#Büyük harflerle yazılmıl columnsları küçültür

dataFrame1.columns = [each.split()[0] + "_" +each.split()[1] if(len(each.split())>1) else each for each in dataFrame1.columns]

#%% drop and concatenaning 

dataFrame1.drop(["yeni_feature"],axis = 1,inplace = True)
                

data1 = dataFrame1.head()
data2 = dataFrame1.tail()                

#vertical
data_concat = pd.concat([data1,data2],axis = 0)




maas = dataFrame1.maas
age = dataFrame1.age

data_h_concat = pd.concat([maas,age],axis = 1)
#horinzontal
# %% transforming data

dataFrame1["list_comp"] = {each*2 for each in dataFrame1.age}

def multiply(age):
    return age*2

dataFrame1["apply_metodu"] = dataFrame1.age.apply(multiply)





