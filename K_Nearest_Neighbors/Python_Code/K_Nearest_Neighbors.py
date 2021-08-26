# This is a K Nearest Neighbors algorithm as a classifier:
# Importing crucial packages for (KNN):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Reading excel data file:
data_file = pd.read_excel('tele_cust.xlsx')
print('first five data: ')
print(data_file.head())

# to use scikit learn library, we need to Convert pandas dataframe to numpy array:
# and Creating the input (X) & output (Y) data:
x = data_file[['region', 'tenure', 'age', 'marital', 'address', 'income'
    , 'education', 'employ', 'retire', 'gender', 'reside', 'custcat']].values.astype(float)
y = data_file[['custcat']].values.astype(float)

# Standardization of  data:
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

# splitting data set (X, Y) to train & test set:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print('The train set: ', x_train.shape, y_train.shape)
print('The test set: ', x_test.shape, y_test.shape)
y_train = y_train.reshape(y_train.shape[0], )
y_test = y_test.reshape(y_test.shape[0], )

# Creating KNN model:
while True:
    try:
        Ks = input('Enter (k): ')
        Ks = int(Ks)
    except:
        if Ks == 'done': break
        print('Enter a valid number for (k)')
        continue
    k_acc = np.zeros((Ks - 1))
    for k in range(1, Ks):
        model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
        yhat = model.predict(x_test)
        k_acc[k - 1] = metrics.accuracy_score(y_test, yhat)
    print('KNN Accuracy from K= 1 until k=', k + 1, k_acc)
    print('Max Accuracy is: ', k_acc.max(), 'and K is: ', k_acc.argmax())

    # Plotting:
    plt.subplot(2, 1, 1)
    plt.scatter(data_file['age'], data_file['income'], color='g')
    plt.xlabel('age')
    plt.ylabel('income')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, Ks), k_acc, color='r')
    plt.xlabel('K in KNN algorithm')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()
