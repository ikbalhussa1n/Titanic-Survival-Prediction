# 🚢 Titanic Survival Prediction (XGBoost + Streamlit App)

A machine learning web application that predicts whether a passenger would have survived the Titanic disaster using a tuned **XGBoost classifier**. The project is deployed as an interactive Streamlit app.

👉 **Live Demo:** https://titanic-survival-prediction-lhsfb8upf4nniuzbjct7wy.streamlit.app/  
👉 **GitHub Repo:** https://github.com/ikbalhussa1n/Titanic-Survival-Prediction  

---

## 📊 Project Overview

This project uses the classic Titanic dataset to build a binary classification model that predicts survival (`0 = No`, `1 = Yes`) based on passenger attributes such as:

- Age
- Sex
- Passenger Class (Pclass)
- Fare
- Family size (SibSp + Parch)
- Embarked location
- Other engineered features

The final model is deployed as an interactive **Streamlit web app** for real-time predictions.

---

## ⚙️ Machine Learning Model

- **Algorithm:** XGBoost Classifier
- **Tuning:** Hyperparameter tuning applied for performance optimization
- **Task:** Binary Classification

---

## 📈 Model Performance

Final tuned model results:

- **Accuracy:** `83.24%`

### Classification Report:

| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|----------|--------|
| 0 (Did not survive) | 0.84 | 0.89 | 0.86 | 105 |
| 1 (Survived) | 0.82 | 0.76 | 0.79 | 74 |

- **Macro Avg F1-score:** 0.82  
- **Weighted Avg F1-score:** 0.83  

---

## 🧠 Key Features

- Cleaned and preprocessed missing values
- Feature engineering (family size, categorical encoding)
- Trained multiple ML models (final: XGBoost)
- Hyperparameter tuning for better generalization
- Real-time predictions via Streamlit UI
- Deployed ML pipeline as a web app

---

## 🖥️ Web App Features

- Interactive input form for passenger details
- Instant survival prediction
- Simple and clean UI using Streamlit
- Fully deployed cloud application


