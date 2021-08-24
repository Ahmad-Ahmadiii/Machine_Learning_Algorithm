# This is an algorithm for Multiple Linear Regression:
# Importing crucial packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn:
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

print("This program's been written on 10th Tir 1400, 2:41 AM")
while True:
    try:
        name = input('Enter the file name: ')
        name = name + '.xlsx'
        data_file = pd.read_excel(name)
        break
    except:
        print('**Enter a valid name**')
        continue
print('the statistical information of the data is: ', data_file.describe())

data = data_file[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                  'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
plot_1 = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
plot_1.hist(color='b')
plt.show()

plt.subplot(2, 1, 1)
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='yellow')
plt.xlabel('The size of Engine')
plt.ylabel(' CO2 Emissions of Cars')

plt.subplot(2, 1, 2)
plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], color='green')
plt.xlabel('the Fuel consumption of cars')
plt.ylabel('CO2 Emissions of Cars')
plt.subplots_adjust(hspace=0.3)
plt.show()

x = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].values
y = data[['CO2EMISSIONS']].values

x_norm = preprocessing.StandardScaler().fit(x).transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.3, random_state=4)

model = linear_model.LinearRegression().fit(x_train, y_train)
print('the Coefficients of the model is: ', model.coef_)
print('the bias of the model is: ', model.intercept_)

y_hat = model.predict(x_test)
print('Mean Absolute Error (MAE): ', np.mean(np.absolute(y_hat - y_test)))
print('Mean Square Error (MSE): ', np.mean(y_hat - y_test)**2)
print('R2_score: ', r2_score(y_hat, y_test))
print('The End')