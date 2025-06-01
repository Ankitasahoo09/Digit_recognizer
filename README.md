
````markdown
# 🖍️ Handwritten Digit Recognizer (GUI Based)

This project lets you draw digits (0–9) on a canvas and predicts them using a trained Logistic Regression model based on Scikit-learn’s **digits dataset**.

It uses:
- 🧠 Scikit-learn for model training
- 🎨 Tkinter for the GUI
- 🖼️ PIL (Pillow) for image processing
- 💾 Joblib for model saving/loading

---

## 🚀 Demo Preview

https://github.com/your-username/digit-recognizer/assets/your-gif-or-screenshot-link *(Optional)*

---

## 📦 Dependencies

Install with pip:

```bash
pip install scikit-learn matplotlib joblib pillow
````

---

## 🛠️ How to Run
ub.com/your-username/digit-recognizer.git
cd digit-recognizer
```

### 2. Train the Model

This step trains a simple Logistic Regression model on the digits dataset and saves it:

```bash
python digit.py
```

You’ll see an accuracy score printed and `digit_model.pkl` created.

### 3. Run the GUI App

```bash
python app.py
```

