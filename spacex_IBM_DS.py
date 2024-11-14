# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Function to plot the confusion matrix
def plot_confusion_matrix(y, y_predict):
    "this function plots the confusion matrix"
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land'])
    ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()

# Load the datasets
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
X = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv")

# Extract the target variable
Y = data['Class'].to_numpy()

# Standardize the feature data
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Logistic Regression with GridSearchCV
parameters = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr = LogisticRegression(max_iter=200)
logreg_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10, scoring='accuracy')
logreg_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for Logistic Regression
print("Logistic Regression - Tuned hyperparameters (best parameters):", logreg_cv.best_params_)
print("Logistic Regression - Best accuracy on validation data:", logreg_cv.best_score_)

# Calculate and print the accuracy on the test data for Logistic Regression
logreg_test_accuracy = logreg_cv.score(X_test, Y_test)
print("Logistic Regression Test Accuracy:", logreg_test_accuracy)

# Plot confusion matrix for Logistic Regression
yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# Support Vector Machine with GridSearchCV
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': np.logspace(-3, 3, 5), 'gamma': np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10, scoring='accuracy')
svm_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for SVM
print("SVM - Tuned hyperparameters (best parameters):", svm_cv.best_params_)
print("SVM - Best accuracy on validation data:", svm_cv.best_score_)

# Calculate and print the accuracy on the test data for SVM
svm_test_accuracy = svm_cv.score(X_test, Y_test)
print("SVM Test Accuracy:", svm_test_accuracy)

# Plot confusion matrix for SVM
yhat = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# Decision Tree with GridSearchCV
parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
              'max_depth': [2*n for n in range(1, 10)], 'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10, scoring='accuracy')
tree_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for Decision Tree
print("Decision Tree - Tuned hyperparameters (best parameters):", tree_cv.best_params_)
print("Decision Tree - Best accuracy on validation data:", tree_cv.best_score_)

# Calculate and print the accuracy on the test data for Decision Tree
tree_test_accuracy = tree_cv.score(X_test, Y_test)
print("Decision Tree Test Accuracy:", tree_test_accuracy)

# Plot confusion matrix for Decision Tree
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# K-Nearest Neighbors with GridSearchCV
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}
KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(estimator=KNN, param_grid=parameters, cv=10, scoring='accuracy')
knn_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for KNN
print("K-Nearest Neighbors - Tuned hyperparameters (best parameters):", knn_cv.best_params_)
print("K-Nearest Neighbors - Best accuracy on validation data:", knn_cv.best_score_)

# Calculate and print the accuracy on the test data for KNN
knn_test_accuracy = knn_cv.score(X_test, Y_test)
print("K-Nearest Neighbors Test Accuracy:", knn_test_accuracy)

# Plot confusion matrix for KNN
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# Determine the best-performing model based on test accuracy
best_model = max(
    ("Logistic Regression", logreg_test_accuracy),
    ("SVM", svm_test_accuracy),
    ("Decision Tree", tree_test_accuracy),
    ("K-Nearest Neighbors", knn_test_accuracy),
    key=lambda x: x[1]
)

print("Best-performing model:", best_model[0], "with accuracy:", best_model[1])
