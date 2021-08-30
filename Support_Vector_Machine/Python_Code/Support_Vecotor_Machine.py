# This is algorithm for Support Vector Machine:
# basic packages:
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn:
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

excel_data = pd.read_excel('cell_samples_data.xlsx')
print('10 rows of the data:\n', excel_data.head(10))

# visualizing data:
ax = excel_data[excel_data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                                                     color='DarkBlue', label='Malignant')
excel_data[excel_data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                                                color='Yellow', label='Benign', ax=ax)
plt.show()

# Data Preprocessing:
excel_data = excel_data[pd.to_numeric(excel_data['BareNuc'], errors='coerce').notnull()].copy()
excel_data['BareNuc'] = excel_data['BareNuc'].astype(int)

# defining X (as inputs):
x_inp = excel_data[
    ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
x_inp = np.asarray(x_inp)

# defining Y as output:
y_out = excel_data[['Class']]
y_out = np.asarray(y_out).reshape((y_out.shape[0] * y_out.shape[1],))

# now it's time to set data to train/test split:
x_train, x_test, y_train, y_test = train_test_split(x_inp, y_out, test_size=0.2, random_state=4)

# Creating our SVM model as Classifier:
print('kernel= [rbf, linear, poly, sigmoid]', '\n')
kernel_name = input('Enter kernel name(from above) ==> ')
svm_model_1 = svm.SVC(kernel=kernel_name)  # SVC = Support Vector Classifier
svm_model_1.fit(x_train, y_train)

# Using (svm_model) to predict on test data:
y_hat1 = svm_model_1.predict(x_test)


# defining a functin for confusion matrix plotting:

def confusion_matrix_plot(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot non-normalized confusion matrix:
eval_0 = confusion_matrix(y_test, y_hat1, labels=[2, 4])
np.set_printoptions(precision=2)

plt.figure(figsize=(6, 6))
confusion_matrix_plot(eval_0, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')
plt.show()

# using metrics methods to evaluate our model:

eval_1 = f1_score(y_test, y_hat1, average='weighted')
eval_2 = jaccard_score(y_test, y_hat1, pos_label=2)
eval_3 = confusion_matrix(y_test, y_hat1, labels=[2, 4])

print('f1 score:\n ', eval_1, '\n')
print('jaccard score:\n', eval_2, '\n')
print('confusion matrix: \n', eval_3, '\n')

##################################################
# Extra Part:
##################################################
# which kernel perfumes better ?! (f1 score)
kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']
eval_list = []
for i, kernel in enumerate(kernel_list):
    svm_model = svm.SVC(kernel=kernel).fit(x_train, y_train)
    y_hat = svm_model.predict(x_test)
    eval_model = f1_score(y_test, y_hat, average='weighted')
    eval_list.append(eval_model)
    print('kernel = ', kernel, '\n')

print('evaluations results:')
print(eval_list)
# plotting results:
x = [1, 2, 3, 4]
x_ticks = ['rbf', 'linear', 'polynomial', 'sigmoid']
plt.xticks(x, x_ticks)  # to change 1, 2, 3, 4 => [rbf, linear, polynomial, sigmoid in x-axis]
plt.plot(x, eval_list)
plt.title('SVM Kernel accuracy results on Prediction')
plt.xlabel('Kernel name')
plt.ylabel('F1_Score accuracy')
plt.show()
