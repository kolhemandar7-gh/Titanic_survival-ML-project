from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved Decision Tree model
model = joblib.load("DT_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        embarked = int(request.form['embarked'])

        # Arrange features for prediction (matching the training column order)
        # Order: Pclass, Sex, Age, SibSp, Parch, Embarked
        features = np.array([[pclass, sex, age, sibsp, parch, embarked]])
        
        prediction = model.predict(features)
        
        result = "Survived" if prediction[0] == 1 else "Did not survive"
        color = "green" if prediction[0] == 1 else "red"

        return render_template('index.html', prediction_text=result, text_color=color)

if __name__ == "__main__":
    app.run(debug=True)