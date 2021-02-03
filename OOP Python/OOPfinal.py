"""
OOP : Object Orianted Programing
    +class
    object
    +attribute
    +encapsulation/abstraction
    +inheritance
"""
from abc import ABC, abstractmethod

class Shape(ABC):
    "Shape = super class / Abstract class"    
    def area(self):pass
    # abstract method

    def parimater(self):pass

    #overriding and poliymorphism
    def toString(self):pass

#child
class Square(Shape):
    "sub clas"
    def __init__(self,edge):
        self.__edge = edge #encapsulatin private attribute
        
    def area(self):
        result = self.__edge**2
        print("Square area",result)
    
    def perimater(self):
        result = self.__edge*4
        print("Square perimater",result)
        
    #Override and polymopism
    def toString(self):
        print("Square edge:",self.__edge)
        
class Circle(Shape):
    "circle class"
    PI = 3.14
    def __init__(self,radius):
        self.__radius = radius
        
    def area(self):
        result = self.PI*self.__radius**2 
        print("Circle area:",result)
        
    def perimater(self):
        result = 2*self.PI*self.__radius
        print("Circle perimater",result)
        
    def toSting(self):
        print("Circle radius",self.__radus)
        
c = Circle(5)
c.area()
c.perimater()
c.toString()        
        
        
        
        
        
        
    
    
    
    
    
    