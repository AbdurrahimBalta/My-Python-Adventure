from abc import ABC, abstractmethod

class Animal(ABC): #super class
    
    @abstractmethod
    def walk(self): pass
    
    @abstractmethod    
    def run(self): pass
class Bird(Animal): # Sub class
    
    def __init__(self):
        print("bird")
        
    def walk(self):
        print("walk")
    
    def run(self):
        print("run")
        
b1 = Bird()

#OVERRİDİNG

class Animal: #Parent
    def toString(self):
        print("animal")
        
class Monkey(Animal):
    
    def toString(self):
       print("monkey") 

a1 = Animal()
a1.toString()

m1 = Monkey() #Monkey calls everring method
m1.toString() #Monkeyin to stringi methodu aminalı geçersiz kıldı