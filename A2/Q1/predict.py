import cv2
import numpy as np
import pickle
import sys

hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

with open("best_cat_dog_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(path):
    img = cv2.imread(path)
    if img is None:
        return "Image not found"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64,64))
    feat = hog.compute(resized).reshape(1, -1)
    pred = model.predict(feat)[0]
    return "Cat" if pred == 0 else "Dog"

print(predict(sys.argv[1]))
