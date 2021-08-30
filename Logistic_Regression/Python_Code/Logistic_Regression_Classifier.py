# This is an algorithm for Logistic Regression:
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


data_file = pd.read_excel('telecom_data.xlsx')
data_file = data_file[
    ['tenure', 'age', 'address', 'income', 'education', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
data_file['churn'] = data_file['churn'].astype(int)

x = np.asarray(data_file[['tenure', 'age', 'address', 'income', 'education', 'employ', 'equip']])
y = np.asarray(data_file[['churn']])
y = y.reshape((y.shape[0] * y.shape[1], ))

# Normalizing data:
x_norm = preprocessing.StandardScaler().fit(x).transform(x)

# Splitting data:
x_norm_train, x_norm_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=4)
print('Train set: ', x_norm_train.shape, y_train.shape)
print('Test set: ', x_norm_test.shape, y_test.shape)

model = LogisticRegression(C=0.01, solver='liblinear').fit(x_norm_train, y_train)

y_hat = model.predict(x_norm_test)
y_hat_prob = model.predict_proba(x_norm_test)
print("Jaccard Score: ", jaccard_score(y_test, y_hat, pos_label=0))


# defining a function:
def confusion_matrix_plot(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    this function plots 'Confusion Matrix'.

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without Normalization')

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


# Computation of Confusion Matrix:
eval_3 = confusion_matrix(y_test, y_hat, labels=[1, 0])
np.set_printoptions(precision=2)

plt.figure()
confusion_matrix_plot(eval_3, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion matrix')
plt.show()