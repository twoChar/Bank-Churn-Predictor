# Bank Customer Churn Predictor

Predict whether a bank customer will leave and generate smart, personalized retention strategies using Machine Learning and LLMs.

![App Screenshot](https://github.com/user-attachments/assets/56f888ec-994b-41f0-8e52-c0108c79fd6f)

---

## Overview

A Streamlit web application that:
- Uses a combination of machine learning models to predict churn
- Leverages Groq's LLM to provide actionable customer retention strategies
- Runs locally or in the browser with minimal setup

---

## Features

- Churn prediction using an ensemble of 9 models (XGBoost, Random Forest, SVM, etc.)
- SMOTE and feature engineering to improve performance
- Real-time retention suggestions via Groq API
- Interactive dashboard built with Streamlit and Plotly

---

## Live Demo

[Try the app here](https://predictor-bank-churn.streamlit.app/)

---

## Model Performance

| Model Variant               | Accuracy |
|----------------------------|----------|
| XGBoost + SMOTE            | 86%      |
| Feature Engineered XGBoost | 88%      |
| Voting Classifier (9 total)| 87%      |

---

## Tech Stack

- Python 3.12
- Streamlit
- scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- Pandas, NumPy, Plotly
- Groq API

---

## Project Structure

```
bank-churn-predictor/
├── .env.example
├── main.py
├── models/
│   └── xgboost_model.pkl
├── data/
│   └── churn.csv
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- Groq API key (sign up at [groq.com](https://groq.com))
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/bank-churn-predictor.git
cd bank-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY in the .env file

# Run the app
streamlit run main.py
```

Make sure:
- All model files (.pkl) are in the `models/` directory
- The dataset (`churn.csv`) is in the `data/` directory

---

## Future Improvements

- Retrain models using alternative feature engineering
- Add `GradientBoostingClassifier` and `StackingClassifier`
- Explore alternative LLMs and prompt strategies
- Deploy models as standalone APIs
- Test across multiple churn datasets

---

## Author

**Tushar Suredia** – [GitHub @twoChar](https://github.com/twoChar)

If you find this useful, feel free to star the repo or contribute!
