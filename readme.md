
# 📊 AI AutoEDA & ML Recommender (Gemini-Powered)

A Streamlit app that automates exploratory data analysis (EDA), data cleaning, and machine learning model selection — powered by Google's Gemini API.

Upload your dataset, and the agent will:
- Understand your data using Gemini
- Clean it automatically
- Detect the task type (classification/regression)
- Apply standardization if needed
- Train multiple models and recommend the best one

---

## 🚀 Demo Preview
> Upload a CSV and get:
- Auto-generated summary from Gemini
- Cleaned dataset
- Best ML model + performance metrics

---

## 📦 Features

✅ Upload CSV  
✅ Gemini-powered dataset summary  
✅ Auto data cleaning  
✅ Task type detection (classification, regression)  
✅ Model training & comparison  
✅ Standardization if needed  
✅ Results displayed instantly via Streamlit

---

## 🧠 Powered By

- [Streamlit](https://streamlit.io/)
- [Google Gemini API](https://makersuite.google.com/app/apikey)
- `scikit-learn` for ML
- `pandas` for data manipulation
- `dotenv` for environment management

---

## 🛠️ Installation & Setup

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Set up environment** (using `conda` or `venv`):

```bash
conda create -n mlagent python=3.10 -y
conda activate mlagent
pip install -r requirements.txt
```

3. **Create `.env` file**:

In the root directory, create a `.env` file and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

4. **Run the app**:

```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
├── app.py
├── .env                 # your Gemini API key (excluded from git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

- This app currently supports classification & regression.
- Clustering functionality is coming soon.
- Your API key is **never shared** or exposed — just make sure to keep `.env` private.
