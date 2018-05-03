import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# load the breast cancer dataset
cancer = load_breast_cancer()
print(cancer.data)
print(cancer.target)
#print(cancer.data);
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
# fit a Naive Bayes model to the data
GaussNB = GaussianNB()
GaussNB.fit(X_train, y_train)
print('Accuracy of GaussNB , on the training set: {:.3f}'.format(GaussNB.score(X_train, y_train)))
print('Accuracy of GaussNB , on the test set: {:.3f}'.format(GaussNB.score(X_test, y_test)))

# fit a k-nearest neighbor model to the data
# Create two lists for training and test accuracies
training_accuracy = []
test_accuracy = []

# Define a range of 1 to 10 (included) neighbors to be tested
neighbors_settings = range(1,11)

# Loop with the KNN through the different number of neighbors to determine the most appropriate (best)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
    print('Accuracy of KNN n-',end = " ");
    print(n_neighbors,end = " ");
    print(', on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
    print('Accuracy of KNN n-',end = " ");
    print(n_neighbors,end = " ");
    print(', on the test set: {:.3f}'.format(knn.score(X_test, y_test)))

#Perceptron Classifier
n_iter = 40
eta0 = 0.1 #learning train_test_split

ppn = Perceptron(max_iter = n_iter, eta0 = eta0, random_state = 66)
# fit a PPN model to the data
ppn.fit(X_train, y_train)
print('Accuracy of Perceptron , on the training set: {:.3f}'.format(ppn.score(X_train, y_train)))
print('Accuracy of Perceptron , on the test set: {:.3f}'.format(ppn.score(X_test, y_test)))
