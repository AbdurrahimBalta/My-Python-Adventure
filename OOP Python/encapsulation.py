class BankAccount(object):
    
    def __init__(self, name, money, address):
        self.name = name # global
        self.__money = money #private
        self.adress = address
   
    #get and set
    def getMoney(self):
        return self.__money
    
    def setMoney(self,amount):
        self.__money = amount
    
    def __incriease(self):
        self.__money = self.__money + 500
        
        #private
p1 = BankAccount("messi", 1000, "barcelona")
p2 = BankAccount("neymar", 2000, "paris")

print("get method",p1.getMoney())

p1.setMoney(5000)

print("after set method",p1.getMoney())

#p1.__incriease()
#print("after raise: ",p1.getMoney())