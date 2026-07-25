from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
# Ensure 'insaurance_model.pkl' is in the same directory as this script
model = pickle.load(open('insaurance_model.pkl', 'rb'))

# HTML and CSS Template embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Premium Predictor</title>
    <style>
        :root {
            --primary-color: #4A90E2;
            --secondary-color: #f5f7fa;
            --text-color: #333;
            --border-color: #e1e4e8;
            --success-color: #2ecc71;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #c3cfe2 0%, #c3cfe2 100%);
            color: var(--text-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 500px;
        }

        h2 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 25px;
            font-size: 24px;
        }

        .form-group {
            margin-bottom: 18px;
        }

        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 14px;
        }

        input[type="number"],
        input[type="text"],
        select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 5px rgba(74, 144, 226, 0.3);
        }

        button {
            width: 100%;
            padding: 12px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s;
            margin-top: 10px;
        }

        button:hover {
            background-color: #357ABD;
        }

        .result-box {
            margin-top: 25px;
            padding: 15px;
            background-color: var(--secondary-color);
            border-left: 5px solid var(--success-color);
            border-radius: 4px;
            text-align: center;
        }

        .result-box h3 {
            color: var(--success-color);
            font-size: 20px;
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>Health Insurance Predictor</h2>
        <form action="/predict" method="post">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required placeholder="e.g., 35">
            </div>

            <div class="form-group">
                <label for="sex">Sex</label>
                <select id="sex" name="sex" required>
                    <option value="" disabled selected>Select Gender</option>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <div class="form-group">
                <label for="bmi">BMI</label>
                <input type="number" step="0.01" id="bmi" name="bmi" required placeholder="e.g., 25.5">
            </div>

            <div class="form-group">
                <label for="children">Number of Children</label>
                <input type="number" id="children" name="children" required placeholder="e.g., 2">
            </div>

            <div class="form-group">
                <label for="smoker">Smoker?</label>
                <select id="smoker" name="smoker" required>
                    <option value="" disabled selected>Select Status</option>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="region">Region</label>
                <select id="region" name="region" required>
                    <option value="" disabled selected>Select Region</option>
                    <option value="0">Southwest</option>
                    <option value="1">Southeast</option>
                    <option value="2">Northwest</option>
                    <option value="3">Northeast</option>
                </select>
            </div>

            <button type="submit">Estimate Charges</button>
        </form>

        {% if prediction_text %}
        <div class="result-box">
            <p>Estimated Insurance Cost:</p>
            <h3>{{ prediction_text }}</h3>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Arrange features in the exact order the model expects: 
        # ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        final_features = np.array([[age, sex, bmi, children, smoker, region]])

        # Make prediction
        prediction = model.predict(final_features)
        
        # Format the output as currency
        output = round(prediction[0], 2)
        result_text = f"${output:,.2f}"

    except Exception as e:
        result_text = f"Error processing request: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=result_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
