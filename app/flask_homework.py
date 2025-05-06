from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__, template_folder='.')  # Look in current directory

# Load model and label encoder
with open('/workspaces/flask_assignment/models/diabetes_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature names in correct order (same as training)
features = ['glucose', 'bloodpressure', 'insulin', 'bmi', 'age', 'diabetespedigreefunction']

# Outcome class mapping (0=No Diabetes, 1=Diabetes)
class_dict = {'0': 'No Diabetes', '1': 'Diabetes'}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form values
        input_data = {
            'glucose': float(request.form['glucose']),
            'bloodpressure': float(request.form['bloodpressure']),
            'insulin': float(request.form['insulin']),
            'bmi': float(request.form['bmi']),
            'age': float(request.form['age']),
            'diabetespedigreefunction': float(request.form['diabetespedigreefunction'])
        }
        
        # Convert to DataFrame with correct feature order
        data = pd.DataFrame([input_data])[features]
        
        # Make prediction
        pred = model.predict(data)[0]
        prediction = class_dict[str(pred)]
        
    return render_template('entry.html',  # Fixed quote
                         prediction=prediction,
                         features=features)

if __name__ == '__main__':
    app.run(debug=True)