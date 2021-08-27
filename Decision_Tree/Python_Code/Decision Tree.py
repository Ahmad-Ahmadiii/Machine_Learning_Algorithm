# This is Decision Tree as a classifier:
# Importing crucial packages:

import pydotplus
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from sklearn:
from sklearn import tree
from sklearn import metrics  # for evaluation of model:
from sklearn import preprocessing  # for preprocessing and standardization of  inputs data.
from sklearn.tree import DecisionTreeClassifier  # selection the model algorithm.
from sklearn.model_selection import train_test_split  # split data to train and test

data_file = pd.read_excel('drug_data.xlsx')
print('the first five of data:')
print(data_file.head())

# Converting pandas dataframe to numpy array:
x = data_file[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = data_file[['Drug']].values

# Data preprocessing:
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:, 1] = le_sex.transform(x[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
x[:, 2] = le_BP.transform(x[:, 2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])
x[:, 3] = le_chol.transform(x[:, 3])

# Splitting data to train & set :
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

# Creating Model Structure:
model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(x_train, y_train)
yhat = model.predict(x_test)
print("Decision Tree Accuracy: ", metrics.accuracy_score(y_test, yhat))

# showing the graph of  Decision Tree model:
dot_data = StringIO()
filename = "drug_tree.png"
featureNames = data_file.columns[0:5]
out = tree.export_graphviz(model, feature_names=featureNames,
                           out_file=dot_data, class_names=np.unique(y_train), filled=True, special_characters=True,
                           rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
# figure:
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
plt.show()
