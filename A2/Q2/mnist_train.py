import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

TRAIN_PATH = "Q2/mnist_train.csv"
TEST_PATH = "Q2/mnist_test.csv"

print("Loading dataset...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y_train = train.iloc[:, 0].values
X_train = train.iloc[:, 1:].values

y_test = test.iloc[:, 0].values
X_test = test.iloc[:, 1:].values


X_train = X_train / 255.0
X_test = X_test / 255.0

print("\nTraining Logistic Regression...")
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
acc_logreg = accuracy_score(y_test, logreg.predict(X_test))
print("Logistic Regression Accuracy:", acc_logreg)

print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
acc_knn = accuracy_score(y_test, knn.predict(X_test))
print("KNN Accuracy:", acc_knn)

print("\nTraining SVM (RBF)...")
svm = SVC(kernel="rbf", gamma="scale", C=5)
svm.fit(X_train[:20000], y_train[:20000])  
acc_svm = accuracy_score(y_test, svm.predict(X_test))
print("SVM Accuracy:", acc_svm)

best = max(
    (acc_logreg, "logreg", logreg),
    (acc_knn, "knn", knn),
    (acc_svm, "svm", svm),
)

acc, name, model = best

print("\nBest model:", name, "| Accuracy:", acc)

with open("best_mnist_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved best_mnist_model.pkl")
