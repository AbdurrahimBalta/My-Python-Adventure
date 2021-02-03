#%% import data
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_regression.csv",sep = ";")

x = df.araba_max_hiz.values.reshape(-1,1)
y = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")
plt.show()

# linear regression y = b0 * + b1*x
# multiple linear regression y = b0 + b1*x1 + b2*x2

#%% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%%predict
y_head = lr.predict(x)

plt.plot(x,y_head,color = "red", label = "linear")
plt.show()

lr.predict(10000)
#%% polynomial
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2) # degree = n

x_polynomial = polynomial_regression.fit_transform(x)
#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

#%%
y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2, color = "green",label = "poly")
plt.legend()
plt.show()





