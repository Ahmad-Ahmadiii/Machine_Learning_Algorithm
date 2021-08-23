# This is an algorithm for Simple Linear Regression Model:
# Importing crucial packages for this method:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

print("This program's been written on 9th Tir 1400, 11:50 PM")
while True:
    try:
        name = input('Enter the file name: ')
        name = name + '.xlsx'
        data_file = pd.read_excel(name)
        break
    except:
        print('**Enter a valid name**')
        continue

print(" Let's see the Statistical information of this datatable:\n", data_file.describe())

data = data_file[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
plot_1 = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
plot_1.hist()
plt.show()

plt.subplot(2, 1, 1)
plot_2 = plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], color='r')
plt.xlabel('Engine Size of the Car')
plt.ylabel('Fuel Consumption of the Car')

plt.subplot(2, 1, 2)
plot_3 = plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='g')
plt.xlabel('Engine size of the Car')
plt.ylabel('Co2 emissions of the Car')
plt.subplots_adjust(hspace=0.3)
plt.show()

x = data[['ENGINESIZE']].values
y = data[['CO2EMISSIONS']].values

x_norm = preprocessing.StandardScaler().fit(x).transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=4)
model = linear_model.LinearRegression().fit(x_train, y_train)
print("There is one Coefficient, because it's linear", model.coef_)
print('The only bias is: ', model.intercept_)

y_hat = model.predict(x_test)
print('Mean Absolute Error (MAE): %.2f' % np.mean(np.absolute(y_hat - y_test)))
print('Mean Square Error (MSE): %.2f' % np.mean((y_hat - y_test)**2))
print('R2_Score: %.2f' % metrics.r2_score(y_hat, y_test))
print('The End')

plot_4 = plt.scatter(x_train, y_train, color='y')
plot_5 = plt.plot(x_train, model.coef_*x_train + model.intercept_, color='b')
plt.show()
