# digit_recognizer.py

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save_model():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print(f"Accuracy: {model.score(X_test, y_test):.2f}")

    # Save model
    joblib.dump(model, "digit_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
