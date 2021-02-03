#parent class
import datetime
class VehicleRent:
    "araçkiralama"
    
    def __init__(self,stock):
        self.stock = stock
        self.now = 0 

    def displayStock(self):
        """
            display stock
        """
        print("{} vehicle available to rent".format(self.stock))
        return self.stock
    
    def rentHourly(self, n):
        
        if n <= 0:
            print("number should be positive")
            return None
        elif n > self.stock:
            print("Sorry {} vehicle available to rent".format(self.stock))
        else:
            self.now = datetime.datetime.now()#Şuanki tarih
            print("Rented a {} vehicle for hprutly at {} hours".format(n,self.now.hour))
            
            self.stock -= n
            
            return self.now
            
    def rentDaily(self, n):
        
        if n <= 0:
            print("number should be positive")
            return None
        elif n > self.stock:
            print("Sorry {} vehicle available to rent".format(self.stock))
        else:
            self.now = datetime.datetime.now()#Şuanki tarih
            print("Rented a {} vehicle for daily at {} hours".format(n,self.now.hour))
            
            self.stock -= n
            
            return self.now
        
    
    def returnVehicle(self, request, brand):
        """
        return a bill
        """
        car_h_price = 10
        car_d_price = car_h_price*8/10*24
        bike_h_price = 5
        bike_d_price = bike_h_price*7/10*24
        
        rentalTime, rentalBasis, numOfVehicle = request
        bill = 0
        
        if brand == "car":
            if rentalTime and rentalBasis and numOfVehicle:
                self.stock += numOfVehicle
                now = datetime.datetime.now()
                rentalPeriod = now - rentalTime 
                
                if rentalBasis == 1: # hourly
                    bill = rentalPeriod.seconds/3600*car_h_price*numOfVehicle
                    #fatura
                elif rentalBasis == 1: # hourly
                    bill = rentalPeriod.seconds/3600*car_d_price*numOfVehicle
                
                if (2 <= numOfVehicle):
                    print("You have extra 20% discount")
                    bill = bill*0.8
                print("Thank you for returing your car")
                print("Price: $ {}".format(bill))
                
                return bill
        
        elif brand == "bike":
            if rentalTime and rentalBasis and numOfVehicle:
                self.stock += numOfVehicle
                now = datetime.datetime.now()
                rentalPeriod = now - rentalTime 
                
                if rentalBasis == 1: # hourly
                    bill = rentalPeriod.seconds/3600*bike_h_price*numOfVehicle
                    #fatura
                elif rentalBasis == 1: # hourly
                    bill = rentalPeriod.seconds/3600*bike_d_price*numOfVehicle
                
                if (4 <= numOfVehicle):
                    print("You have extra 20% discount")
                    bill = bill*0.8
                print("Thank you for returing your bike")
                print("Price: $ {}".format(bill))
                return bill
        else:
            print("you do not rent a vehicle")
            return None
#child class
            
class CarRent(VehicleRent):
    
    global discount_rate
    discount_rate = 15
    
    def __init__(self, stock):
        super().__init__(stock)
        
    
    def discount(self, b):
        "araç giralayan gişi"
        bill = b - (b*discount_rate)/100
        return bill
    
#child class 2
class BikeRent(VehicleRent):
    
    def __init__(self, stock):
        super().__init__(stock)
        
        

class Customer:
    
    def __init__(self):
        self.bikes = 0
        self.rentelBasis_b = 0
        self.rentelTime_b = 0
        
        self.cars = 0
        self.rentelBasis_c = 0
        self.rentelTime_c = 0
    
    def requestVehicle(self, brand):
        "take a request bike or car from custumer"
        if brand == "bike":
            bikes = input("how many bikes would like to rent?")
            try:
                bikes = int(bikes)
            except ValueError:
                print("number should be number")
                return -1
            
            if bikes < 1:
                print("number of bikes should be greater than zero")
            else:
                self.bikes = bikes
            return self.bikes
        
        elif brand == "car":
            cars = input("How many cars would you like to rent")
            
            try: 
                cars = int(cars)
            except ValueError:
                print("number should be number")
                return -1
            
            if cars < 1:
                print("number of cars should be grearer than zero")
                return -1
            else:
                self.cars = cars
            return self.bikes
                
        else:
            print("request vehicle error")
    
    def returnVehicle(self, brand):
        "return bikes or cars"
        
        if brand == "bike":
            if self.rentelTime_b and self.rentelBasis_b and self.bikes:
                return self.rentelTime_b, self.rentelBasis_b, self.bikes
            else:
                return 0,0,0
        elif brand == "cars":
            if self.rentelBasis_c and self.rentelBasis_c and self.cars:
                return self.rentelTime_c, self.rentelBasis_c, self.cars
            else:
                return 0,0,0                
                
        else:
            print("return vehicle error")
            
            
        
        
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        