from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset
digits = datasets.load_digits()

# Show one sample digit
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Label: {digits.target[0]}")
plt.show()

# Prepare data
X = digits.data  # flatten 8x8 image to 64 features
y = digits.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and test accuracy
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Predict a single digit
index = 10
plt.matshow(digits.images[index])
plt.title(f"Actual: {digits.target[index]} | Predicted: {model.predict([digits.data[index]])[0]}")
plt.show()
