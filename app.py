from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('save_model12')

# Create LabelEncoder instance and fit it
le = LabelEncoder()
le.fit(['M', 'F', 'I'])  

# Define the feature columns
feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 'Whole weight.2', 'Shell weight']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from the form
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        
       
        df = df.rename(columns={
            'Sex': 'Sex',
            'Length': 'Length',
            'Diameter': 'Diameter',
            'Height': 'Height',
            'WholeWeight': 'Whole weight',
            'WholeWeight1': 'Whole weight.1',
            'WholeWeight2': 'Whole weight.2',
            'ShellWeight': 'Shell weight'
        })
        
        # Encode categorical variables
        if 'Sex' in df.columns:
            df['Sex'] = le.transform(df['Sex'])

        # Ensure all required columns are present and correctly ordered
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Assign default value if column is missing
        
        # Predict for loaded model
        prediction = model.predict(df[feature_columns])
        
        # Return prediction as JSON
        return jsonify({'Rings': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
