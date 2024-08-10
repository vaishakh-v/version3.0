from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_responses = [int(request.form[f'question_{i+1}']) for i in range(total_questions)]
        result = classify_mental_disorder('path_to_your_questions_file.csv', user_responses)
        return render_template('result.html', result=result)
    return render_template('index.html')

def classify_mental_disorder(csv_file_path, user_responses):
    df = pd.read_csv(csv_file_path)
    disorder_scores = {
        "Depression": 0,
        "Anxiety": 0,
        "ADHD": 0,
        "PTSD": 0,
        "Psychosis & Schizophrenia": 0,
        "Bipolar": 0
    }
    
    for i, row in df.iterrows():
        disorder = row['Mental_Disorder']
        response_value = user_responses[i]
        disorder_scores[disorder] += response_value
    
    predicted_disorder = max(disorder_scores, key=disorder_scores.get)
    
    return predicted_disorder

if __name__ == '__main__':
    total_questions = 70  # Set this to the number of questions in your CSV file
    app.run(debug=True)
