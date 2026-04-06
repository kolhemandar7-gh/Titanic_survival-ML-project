🚢 Titanic Survival Prediction Web App

A complete Machine Learning + Flask Web Application that predicts whether a passenger would survive the Titanic disaster based on input features like age, class, gender, and more.

📌 Project Overview

This project uses the famous Titanic dataset to:

Perform Data Analysis & Visualization
Apply Multiple ML Algorithms
Compare model performance
Deploy the best model using a Flask Web App

The final model allows users to input passenger details through a web interface and get real-time survival predictions.

⚙️ Tech Stack
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Flask
HTML + CSS (Vintage UI Design)
📂 Project Structure
📁 Titanic-Survival-Predictor
│
├── titanic.py              # ML pipeline (OOP based)
├── titanic.csv             # Dataset
├── DT_model.pkl            # Saved Decision Tree model
├── app.py                  # Flask application
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/ (optional)
│
└── README.md               # Project documentation
🔄 Machine Learning Workflow
Data Loading
Data Understanding
Handling Missing Values
Exploratory Data Analysis
Data Visualization
Data Preprocessing
Encoding categorical variables
Train-Test Split
Model Training
Logistic Regression
KNN
Decision Tree ✅ (Final Model)
Naive Bayes
SVM
Model Evaluation
Model Saving (.pkl)
📊 Model Performance

The project compares multiple models and selects:

✅ Decision Tree Classifier (used for deployment)

🌐 Web Application Features
User-friendly Passenger Form
Vintage Titanic-themed UI
Real-time prediction:
🟢 Survived
🔴 Did not survive
Clean and responsive design
🧠 Input Features
Passenger Class (1, 2, 3)
Sex (Male/Female)
Age
Siblings/Spouses aboard
Parents/Children aboard
Port of Embarkation
📈 Future Improvements
Add more feature engineering
Use advanced models (Random Forest, XGBoost)
Deploy on cloud (Heroku / Render / AWS)
Add user authentication
