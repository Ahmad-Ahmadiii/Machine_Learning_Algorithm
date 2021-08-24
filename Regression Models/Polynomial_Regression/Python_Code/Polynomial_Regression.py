# This is an algorithm for Polynomial Regression:
# Importing crucial packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn:
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics

print("This program has been written on 10th Tir 1400, 8:04 PM")
while True:
    try:
        name = input('Enter the file name: ')
        name = name + '.xlsx'
        data_file = pd.read_excel(name)
        break
    except:
        print('**Enter a valid name**')
        continue
print('The statistical information of this data is: ', data_file.describe())

data = data_file[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='g')
plt.xlabel('The engine size')
plt.ylabel('CO2 emissions')
plt.show()

x = data['ENGINESIZE'].values
x = x.reshape(x.shape[0], 1)
y = data['CO2EMISSIONS'].values
y = y.reshape(y.shape[0], 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
print('train set size: ', 'x=', x_train.shape, 'y=', y_train.shape)
print('test set size: ', 'x=', x_test.shape, 'y=', y_test.shape)

poly_feature = PolynomialFeatures(degree=2)
x_train_poly = poly_feature.fit_transform(x_train)
model = linear_model.LinearRegression().fit(x_train_poly, y_train)
print('Coefficients: ', model.coef_)
print('intercept: ', model.intercept_)

x_test_poly = poly_feature.fit_transform(x_test)
y_hat = model.predict(x_test_poly)
print('Mean Square Error: ', np.mean((y_hat - y_test)**2))
print('Mean Absolute Error: ', np.mean(np.absolute(y_hat - y_test)))
print('R2_Score: ', metrics.r2_score(y_hat, y_test))

x_train_df = pd.DataFrame(x_train, columns=['ENGINESIZE'])
y_train_df = pd.DataFrame(y_train, columns=['CO2EMISSIONS'])
plt.scatter(x_train_df.ENGINESIZE, y_train_df.CO2EMISSIONS)
x_sample = np.arange(0, 10, 0.1)
y_sample = model.coef_[0, 1] * x_sample + model.coef_[0, 2] * np.power(x_sample, 2) + model.intercept_[0]
plt.plot(x_sample, y_sample, 'r')
plt.xlabel('Engine Size of Car')
plt.legend(['polynomial prediction', 'train data'])
plt.xlabel('Engines Size of Car')
plt.ylabel('CO2 Emissions')
plt.show()

