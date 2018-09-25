import csv
import numpy as np
import pandas as pd
from sklearn import linear_model

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

with open('/home/sentos/Documents/PycharmProjects/clustering/ex1data1.txt', 'rt') as f:
  reader = csv.reader(f, delimiter=',', skipinitialspace=True)

  lineData = list()


  cols =["population","profits"]
  print(cols)

  for col in cols:
    # Create a list in lineData for each column of data.
    lineData.append(list())


  for line in reader:
    for i in range(0, len(lineData)):
      # Copy the data from the line into the correct columns.
      lineData[i].append(line[i])

  data = dict()

  for i in range(0, len(cols)):
    # Create each key in the dict with the data in its column.
    data[cols[i]] = lineData[i]


for key in data.keys():
    for i in range(0,len(data[key])):
        data[key][i]=float(data[key][i])

data=pd.DataFrame({"one":np.ones(97),"population":data["population"],"profits":data["profits"]})

X=data.drop("profits",axis=1)

y=data.profits

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.4,random_state=0)

lr=linear_model.LinearRegression()

lr.fit(X_train,Y_train)

y_pred=lr.predict(X_test)
mse=mean_squared_error(Y_test,y_pred)
y_error=Y_test-y_pred


plt.scatter(Y_test,y_pred,marker="*")
plt.xlabel("measured")
plt.ylabel("Predicted")
plt.title("predicted vs measured")
plt.show()