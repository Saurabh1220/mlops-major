import joblib
from sklearn.metrics import accuracy_score

def main():
    data = joblib.load("models/savedmodel.pth")
    clf = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Test accuracy:", acc)

if __name__ == "__main__":
    main()
