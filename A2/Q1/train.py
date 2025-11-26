import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TRAIN_DIR = "Q1/train"

hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

def load_folder(path, label):
    X, y = [], []
    for fname in os.listdir(path):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(path, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64,64))
        feat = hog.compute(resized).flatten()
        X.append(feat)
        y.append(label)
    return X, y

def load_dataset():
    X, y = [], []
    cat_path = os.path.join(TRAIN_DIR, "Cat")
    dog_path = os.path.join(TRAIN_DIR, "Dog")

    Xc, yc = load_folder(cat_path, 0)
    Xd, yd = load_folder(dog_path, 1)

    X = np.array(Xc + Xd)
    y = np.array(yc + yd)
    return X, y

print("Loading training data...")
X, y = load_dataset()
print("Training samples:", len(y))

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
])

print("Training model...")
model.fit(X, y)

with open("best_cat_dog_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as best_cat_dog_model.pkl")
