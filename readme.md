# 🫀 Heart Disease Prediction API

An end-to-end Machine Learning project that predicts the likelihood of heart disease based on patient clinical parameters. Built using Python, Scikit-Learn, and deployed as a RESTful API using FastAPI.

---

## 📌 Features

* **Data Preprocessing & EDA:** Handled missing values, examined class balance, and evaluated   metrics.
* **Model Training:** Trained a `RandomForestClassifier` targeting high **Recall (97%)** to minimize False Negatives in medical diagnosis.
* **Hyperparameter Tuning:** Optimized tree parameters using `GridSearchCV` with cross-validation.
* **Production API:** Served real-time inference via FastAPI with `Pydantic` data validation.
* **Interactive UI:** Built-in Swagger UI documentation for manual endpoint testing.

---

## 🛠️ Tech Stack

* **Language:** Python 3.14
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Model Persistence:** Joblib
* **API Framework:** FastAPI, Uvicorn, Pydantic

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/heart-disease-prediction-api.git](https://github.com/YOUR_USERNAME/heart-disease-prediction-api.git)
cd heart-disease-prediction-api
pip install -r requirements.txt