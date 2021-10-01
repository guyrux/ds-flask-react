import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42

df = pd.read_csv("./data/raw/diabetes.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

pickle.dump(model, open("./model/example_knn.pkl", "wb"))
