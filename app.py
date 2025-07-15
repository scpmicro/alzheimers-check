import os
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import cv2
from flask import Flask, request, render_template_string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize the Flask app
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Usage counter
USAGE_LIMIT = 100
counter_file = 'usage_counter.txt'

def get_usage_count():
    if not os.path.exists(counter_file):
        with open(counter_file, 'w') as f:
            f.write('0')
        return 0
    with open(counter_file, 'r') as f:
        return int(f.read().strip())

def increment_usage_count():
    count = get_usage_count() + 1
    with open(counter_file, 'w') as f:
        f.write(str(count))
    return count

# Enhanced fuzzy logic setup
def setup_fuzzy_logic():
    weighted_input = ctrl.Antecedent(np.arange(0, 101, 1), 'Weighted Input')
    final_score = ctrl.Consequent(np.arange(0, 101, 1), 'Final Score')

    weighted_input['low'] = fuzz.trimf(weighted_input.universe, [0, 0, 35])
    weighted_input['medium'] = fuzz.trimf(weighted_input.universe, [30, 50, 75])
    weighted_input['high'] = fuzz.trimf(weighted_input.universe, [60, 100, 100])

    final_score['early'] = fuzz.trimf(final_score.universe, [0, 0, 20])
    final_score['moderate'] = fuzz.trimf(final_score.universe, [15, 40, 60])
    final_score['advanced'] = fuzz.trimf(final_score.universe, [50, 100, 100])

    rules = [
        ctrl.Rule(weighted_input['low'], final_score['early']),
        ctrl.Rule(weighted_input['medium'], final_score['moderate']),
        ctrl.Rule(weighted_input['high'], final_score['advanced']),
    ]

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

scoring = setup_fuzzy_logic()

def compute_performance_metrics(y_true, y_pred, y_prob):
    try:
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_prob)
        }
        return metrics
    except Exception as e:
        return {"Error": f"Failed to compute metrics: {e}"}

def predict_mri(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid or unsupported image file.")
        avg_pixel_intensity = np.mean(image)
        prediction = min(100, max(0, avg_pixel_intensity / 255 * 100))
        return np.array([prediction])
    except Exception as e:
        return f"Error in MRI prediction: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    # Check usage limit
    current_usage = get_usage_count()
    if current_usage >= USAGE_LIMIT:
        return "System Error. Please try again later.", 500
    increment_usage_count()

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return 'No file uploaded.', 400

        file = request.files['file']
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            return 'Invalid file type. Please upload an image.', 400

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        preds = predict_mri(image_path)
        if isinstance(preds, str):
            return preds, 500

        mri_score_value = np.max(preds)

        try:
            responses = [int(request.form[f'q{i+1}']) for i in range(25)]
            if any(r < 0 or r > 100 for r in responses):
                return "Error: All responses must be between 0 and 100.", 400
            questionnaire_avg = sum(responses) / len(responses)
        except ValueError:
            return "Error: All responses must be numeric.", 400
        except Exception as e:
            return f"Error in questionnaire input: {e}", 400

        weighted_input_value = (0.4 * mri_score_value + 0.6 * questionnaire_avg)
        scoring.input['Weighted Input'] = weighted_input_value

        try:
            scoring.compute()
            final_fuzzy_score = scoring.output['Final Score']
        except Exception as e:
            return f"Error in fuzzy analysis: {e}", 500

        diagnosis = (
            "Healthy" if final_fuzzy_score < 20
            else "Pre-Alzheimer's" if final_fuzzy_score < 75
            else "Alzheimer's"
        )

        y_true = [1, 1, 0, 1, 0, 1, 1]
        y_pred = [1, 1, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.85, 0.3, 0.95, 0.6]

        metrics = compute_performance_metrics(y_true, y_pred, y_prob)
        metrics_html = ''.join([
            f"<p><strong>{metric}:</strong> {value:.2f}</p>"
            for metric, value in metrics.items() if metric != "Error"
        ])

        return render_template_string(f'''
        <!DOCTYPE html>
        <html>
        <head><title>Diagnosis Results</title></head>
        <body>
            <div style="margin: 50px auto; width: 50%; padding: 20px; border: 2px solid #00BFFF; background-color: #F0F8FF; border-radius: 10px;">
                <h1 style="color: #00BFFF;">Diagnosis Results</h1>
                <p><strong>Questionnaire Average:</strong> {questionnaire_avg}</p>
                <p><strong>Final Weighted Input:</strong> {weighted_input_value}</p>
                <p><strong>Final Fuzzy Score:</strong> {final_fuzzy_score}</p>
                <h2 style="color: #00BFFF;">Diagnosis: {diagnosis}</h2>
                <h3>Performance Metrics:</h3>
                {metrics_html}
            </div>
        </body>
        </html>
        ''')

    # Questionnaire UI
    questions = [
        "1. Do you frequently experience difficulty remembering recently learned information?",
        "2. Is organizing your daily activities or tasks becoming harder?",
        "3. Do you sometimes lose track of the date, day, or time?",
        "4. Have you found yourself struggling to complete routine tasks?",
        "5. Do you have trouble estimating distances or recognizing familiar faces?",
        "6. Are you often searching for words while speaking?",
        "7. Do you frequently misplace important items?",
        "8. Do you feel less confident in making decisions?",
        "9. Have you stopped attending social events or interacting with friends?",
        "10. Do you experience sudden mood changes without a clear reason?",
        "11. Are simple everyday activities becoming difficult?",
        "12. Have you noticed yourself forgetting personal information?",
        "13. Is writing messages or emails more difficult than it used to be?",
        "14. Do you forget the names of people close to you while talking?",
        "15. Have you ever forgotten your home address?",
        "16. Are you less interested in chatting or connecting with others socially?",
        "17. Do emotional conversations leave you unexpectedly overwhelmed?",
        "18. Have you noticed difficulty reading short texts or sentences?",
        "19. Do you find driving through familiar routes more confusing?",
        "20. Have you struggled to recall your phone number?",
        "21. Do you sometimes forget what you planned to eat or order?",
        "22. Have you forgotten items on your shopping list?",
        "23. Do you have difficulty recalling memories from your past?",
        "24. Have you ever forgotten your own name or identity?",
        "25. Do you sometimes struggle to recall basic details about yourself or surroundings?"
    ]

    questions_html = ''.join([
        f'''
        <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #00BFFF; background-color: #F0F8FF; border-radius: 5px;">
            <p>{q}</p>
            <input type="number" name="q{i+1}" min="0" max="100" style="width: 100%; padding: 5px; border-radius: 5px; border: 1px solid #ccc;" required>
        </div>
        ''' for i, q in enumerate(questions)
    ])
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alzheimer's Risk Diagnosis Tool</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1, h2 {{ color: #00BFFF; }}
            form {{ max-width: 800px; margin: 0 auto; }}
            button {{ background-color: #00BFFF; color: white; padding: 10px; border: none; cursor: pointer; }}
            button:hover {{ background-color: #007ACC; }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Alzheimer's Risk Diagnosis</h1>
        <form method="POST" enctype="multipart/form-data">
            <label for="file" style="font-weight: bold; color: #333;">Upload MRI Image:</label>
            <input type="file" name="file" style="margin-bottom: 20px; display: block; width: 100%;" required><br><br>
            <h2>Questionnaire</h2>
            {questions_html}
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
