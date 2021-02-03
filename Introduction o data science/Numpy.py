# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 02:02:58 2020

@author: LENOVO
"""

#%% importing

import numpy as np 

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #Vector

print(array.shape)

a = array.reshape(3,5)

print("shape",a.shape)
#print("dimension",a.ndin)

print("data type", a.dtype.name)
print("size:",a.size)

print("type",type(a))

array1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11]])

np.zeros((3,4)) #Yer ayırma 3-4 lük 0 lardan oluşan matris

zeros = np.zeros((3,4))

zeros[0,0] = 5
print(zeros)

np.ones((3,4)) # 1 lerden

np.empty(2,3) # 0 a çok yakın ve sıfır kabul edilen sayılardan

a = np.arange(10,50,5) #10dan 50 ye kadar 50 dahil değil 5 er 5 er art
 
a = np.linspace(10,50,20) #10 la 50 arasına 20 tane sayı  yaz

#%% Numby basic operations

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)

print(np.sin(a))

print(a<2)

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])

#element wise prodcut
print(a*b)


#matris product
a.dot(b.T) # t= transpoz

print(np.exp(a)) #Exponential üstel fonksiyonlar

a = np.random.random((5,5))


print(a.sum()) #topla
print(a.max()) #max random
print(a.min()) #min random


print(a.sum(axis=0))
print(a.sum(axis=1))


print(np.sqrt(a))
print(np.square(a)) 

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])
print(a.dot(b.T))

#%% indexing and slicing

array = np.array([1,2,3,4,5,6,7]) #vectör 

print(array[0])

print(array[0:4])

array1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array1[1,1])

#%% shape manipulation


array = np.array([[1,2,3],[4,5,6],[7,8,9]])
#flatting
a = array.ravel()

array2 = a.reshape(3,3)

arrayT = array2.T

print(arrayT.shape)

array5 = np.array([[1,2],[3,4],[4,5]])

array5.resize(3,2) #resize reshapein farklı değişken gerektirmeyen halidir
#%% stacking arrays


array1 = np.array([[1,2],[3,4]])

array2 = np.array([[-1,-2],[-3,-4]])

array3 = np.vstack((array1,array2))

array4 = np.hstack((array1,array2))
#%% Convert and copy

liste = [1,2,3,4] #list

array = np.array([1,2,3,4]) #np array

liste2 = list(array)


a = np.array([1,2,3])

b = a

c = a

d = np.array([1,2,3])

e = d.copy()


















