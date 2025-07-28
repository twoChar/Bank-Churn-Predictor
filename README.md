# Bank Customer Churn Predictor 🏦

![image](https://github.com/user-attachments/assets/56f888ec-994b-41f0-8e52-c0108c79fd6f)

## Overview
An end-to-end machine learning application that predicts customer churn probability and generates personalized engagement strategies using ML models and LLMs.

## 🔥 Features
- **Multi-Model Prediction**: Ensemble of 9 different ML models including XGBoost, Random Forest, and SVM
- **Interactive Dashboard**: Built with Streamlit for easy data input and visualization
- **AI-Driven Engagement**: Personalized customer retention strategies using Groq's LLM
- **Advanced ML Techniques**: 
  - SMOTE for handling imbalanced data
  - Feature engineering for improved accuracy
  - Voting classifier for ensemble predictions

## 🚀 Live Demo
Try out the live application here: [Bank Churn Predictor](https://bank-churn-predictor.streamlit.app)

## 💻 Tech Stack
- Python 3.12
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Groq API
- Plotly

## 📊 Model Performance
- XGBoost with SMOTE: 86% accuracy
- Feature-engineered model: 88% accuracy
- Voting Classifier: 87% accuracy

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bank-churn-predictor.git
cd bank-churn-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create a .env file in the root directory and add:
GROQ_API_KEY=your_api_key_here
```

4. **Run the application**
```bash
streamlit run main.py
```

### Prerequisites
- Python 3.11 or higher
- Groq API key (get it from [Groq's website](https://groq.com))
- Git

### Note
Make sure all model files (`.pkl`) are in the `models/` directory and the dataset (`churn.csv`) is in the `data/` directory before running the application.

## 📁 Project Structure
```
bank-churn-predictor/
├── main.py                     # Main Streamlit application
├── utils.py                    # Utility functions and visualizations
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables (Groq API key)
├── models/                     # Trained ML models
│   ├── xgb_model.pkl          # XGBoost model
│   ├── nb_model.pkl           # Naive Bayes model
│   ├── rf_model.pkl           # Random Forest model
│   ├── dt_model.pkl           # Decision Tree model
│   ├── svm_model.pkl          # SVM model
│   ├── knn_model.pkl          # K-Nearest Neighbors model
│   ├── voting_clf.pkl         # Voting Classifier model
│   ├── xgboost-SMOTE.pkl      # XGBoost with SMOTE
│   └── xgboost-featureEngineered.pkl  # Feature-engineered XGBoost
├── data/                       # Dataset directory
│   └── churn.csv              # Bank customer dataset
├── notebooks/                  # Jupyter notebooks
│   └── model_training.ipynb    # Model training and analysis
└── README.md                   # Project documentation
```

## 🔮 Future Improvements
- [ ] Retrain models with different feature engineering approaches
- [ ] Implement GradientBoostingClassifier and StackingClassifier
- [ ] Explore different LLM options and prompting techniques
- [ ] Deploy ML models as separate API endpoints
- [ ] Test models on different churn datasets

## 📝 Blog Post
Read about the development process and technical details in my blog post: [How I Built a System to Predict and Prevent Bank Customer Churn Using ML, LLMs, and Streamlit](https://imalexwang.substack.com/p/how-i-built-a-system-to-predict-and)

## 📫 Contact
- LinkedIn: [Alex Wang](https://www.linkedin.com/in/alexwang-/)
- Twitter: [@imalexwang](https://x.com/imalexwang)
- Blog: [solo diaries](https://imalexwang.substack.com/)
# Bank-Churn-Predictor
# Bank-Churn-Predictor
# Bank-Churn-Predictor
# Bank-Churn-Predictor
# Bank-Churn-Predictor
# Bank-Churn-Predictor
