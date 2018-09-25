from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn import datasets,linear_model
from matplotlib import pyplot as plt
import pandas as pd
import sklearn

dataset=datasets.load_boston()

bos=pd.DataFrame(dataset.data)
bos.columns=dataset.feature_names

bos['PRICE']=dataset.target

Y=bos['PRICE']
X=bos.drop('PRICE',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

lr=linear_model.LinearRegression()

lr.fit(X_train,y_train)

Y_pred=lr.predict(X_test)

mse=sklearn.metrics.mean_squared_error(y_test,Y_pred)
print(mse)

plt.scatter(y_test,Y_pred)
plt.ylabel("Prices : $Y_i$")
plt.xlabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices Vs Predicted Prices : $Y_i$ vs $\hat{Y}_i$")

plt.show()
