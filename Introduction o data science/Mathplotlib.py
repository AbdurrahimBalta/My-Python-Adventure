""" Matplotlib kütüphanesi
    görselleştirme kütüphanesi
    line ploy, scatter plot, bar plot, sublots, histogram
"""

import pandas as pd

df = pd.read_csv("iris.csv")


print(df.columns)

print(df.Species)

print(df.Species.unique()) # farklı türleri yazdır 
"sdfdsfdfdf
"

print(df.info())

print(df.describe())

setosa = df[df.Species == "Iris-setosa"]

versicolor = df[df.Species == "Iris-versicolor"]

print(setosa.describe())
print(versicolor.describe())

# %%
import matplotlib.pyplot as plt

df1 = df.drop(["Id"],axis = 1 ) #ıdyi dorp et 


setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]


plt.plot(setosa.Id ,setosa.PetalLengthCm, color = "red",label = "setosa - PetalLengthCm" )
plt.plot(versicolor.Id ,versicolor.PetalLengthCm, color = "green",label = "versicolor" )
plt.plot(virginica.Id ,virginica.PetalLengthCm, color = "blue",label = "virginica" )


plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.legend()
plt.show()



df1.plot(grid = True,linestyle = ":")
plt.show()
# %% histogram 

plt.hist(setosa.PatelLengthCm, bins = 30)
plt.xlabel("PatelLengthCm Values")
plt.ylabel("frekans")
plt.title("hist")
plt.show()
#%% bar plot

import numpy as np 
"""
x = np.array([1,2,3,4,5,6,7])

y = x*2+5

plt.bar(x,y)
plt.title("bar plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
""" 

x = np.array([1,2,3,4,5,6,7])
a = {"turkey","usa","turkey","turkey","turkey","turkey","turkey"}
y = x*2+5

plt.bar(a,y)
plt.title("bar plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
 




















