import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("randomforest.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% 
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42) # kaç tree kulanıcalak
rf.fit(x,y)

print(rf.predict([[7.5]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#%% visualize

plt.scatter(x,y,color = "red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
#%% 
from sklearn.metrics import r2_score

print("r_score:", r2_score(y,y_head))