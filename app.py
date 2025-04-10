import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from dotenv import load_dotenv
import os

load_dotenv()  # This loads your .env variables into os.environ

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(page_title="AI AutoEDA + ML Agent", layout="wide")
st.title("🧠 AI AutoEDA + ML Recommender (Gemini-powered)")

# Gemini API Key (you can load from .env or secrets for production)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# ---------------------------
# CONSTANT PROMPT TEMPLATE
# ---------------------------
def generate_prompt(data_sample, data_types):
    return f"""
You are a data science assistant. A user has uploaded a CSV dataset.
Here is a sample of the data:

{data_sample}

And here are the column names with data types:

{data_types}

Please do the following:
1. Summarize what the dataset is likely about.
2. Identify the most likely target variable.
3. List all numeric and categorical features.
4. Suggest appropriate data cleaning steps.
5. Recommend whether to use a classification, regression, or clustering model and why.
6. List at least 2 models suitable for the task.
7. Suggest if feature scaling (e.g., standardization) is required.
"""

# ---------------------------
# DATA CLEANING FUNCTION
# ---------------------------
def clean_data(df, gemini_text):
    df_cleaned = df.copy()

    # Drop duplicates
    if "drop duplicate" in gemini_text.lower():
        df_cleaned.drop_duplicates(inplace=True)

    # Fill missing values
    if "missing" in gemini_text.lower() or "null" in gemini_text.lower():
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype == "object":
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                else:
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    # Encode categorical features
    if "categorical" in gemini_text.lower() or "encoding" in gemini_text.lower():
        cat_cols = df_cleaned.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df_cleaned[col].nunique() <= 20:
                le = LabelEncoder()
                df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))

    return df_cleaned

# ---------------------------
# DETECT TASK TYPE
# ---------------------------
def detect_task(gemini_text):
    if "classification" in gemini_text.lower():
        return "classification"
    elif "regression" in gemini_text.lower():
        return "regression"
    elif "clustering" in gemini_text.lower():
        return "clustering"
    else:
        return None

# ---------------------------
# UPLOAD SECTION
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    data_sample = df.sample(min(10, len(df))).to_string(index=False)
    data_types = df.dtypes.to_string()

    # Gemini API Call
    with st.spinner("🤖 Analyzing data with Gemini..."):
        prompt = generate_prompt(data_sample, data_types)
        response = model.generate_content(prompt)
        gemini_output = response.text

    st.subheader("🧠 Gemini's Analysis & Suggestions")
    st.markdown(gemini_output)

    # Clean Data
    st.subheader("🧼 Cleaned Dataset (Based on Gemini's Suggestions)")
    df_cleaned = clean_data(df, gemini_output)
    st.dataframe(df_cleaned.head())

    # ML Task
    task_type = detect_task(gemini_output)

    # Target Column
    st.subheader("🎯 Select Target Column")
    target_column = st.selectbox("Choose target variable", df_cleaned.columns)

    if st.button("🚀 Train ML Model"):
        if target_column:
            X = df_cleaned.drop(columns=[target_column])
            y = df_cleaned[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply standardization if Gemini says so
            if "standardization" in gemini_output.lower() or "scaling" in gemini_output.lower():
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            st.subheader("📊 Model Results")

            if task_type == "classification":
                model1 = LogisticRegression(max_iter=500)
                model2 = RandomForestClassifier()

                model1.fit(X_train, y_train)
                model2.fit(X_train, y_train)

                pred1 = model1.predict(X_test)
                pred2 = model2.predict(X_test)

                acc1 = accuracy_score(y_test, pred1)
                acc2 = accuracy_score(y_test, pred2)

                best_model = "Logistic Regression" if acc1 > acc2 else "Random Forest"
                best_acc = max(acc1, acc2)

                st.success(f"✅ Best Model: **{best_model}** with Accuracy: **{best_acc:.2f}**")

            elif task_type == "regression":
                model1 = LinearRegression()
                model2 = RandomForestRegressor()

                model1.fit(X_train, y_train)
                model2.fit(X_train, y_train)

                pred1 = model1.predict(X_test)
                pred2 = model2.predict(X_test)

                rmse1 = mean_squared_error(y_test, pred1, squared=False)
                rmse2 = mean_squared_error(y_test, pred2, squared=False)

                best_model = "Linear Regression" if rmse1 < rmse2 else "Random Forest Regressor"
                best_rmse = min(rmse1, rmse2)

                st.success(f"✅ Best Model: **{best_model}** with RMSE: **{best_rmse:.2f}**")

            else:
                st.warning("⚠️ Clustering not yet implemented.")
        else:
            st.warning("Please select a target column.")
else:
    st.info("📥 Upload a CSV file to get started.")
