from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def main():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {"model": clf, "X_test": X_test, "y_test": y_test},
        "models/savedmodel.pth"
    )

    print("Model trained and saved to models/savedmodel.pth")

if __name__ == "__main__":
    main()
