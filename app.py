from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the model and preprocessing tools
model = load_model('models/question_selector_model.h5')
with open('models/le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('models/le_location.pkl', 'rb') as f:
    le_location = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = le_gender.transform([request.form['gender']])[0]
        location = le_location.transform([request.form['location']])[0]
        last_response = int(request.form['last_response'])
        
        user_data = np.array([[age, gender, location, last_response]])
        user_data = scaler.transform(user_data)
        
        predictions = model.predict(user_data)
        selected_questions = select_relevant_questions(predictions)
        
        return render_template('questions.htm', selected_questions=selected_questions)
    
    return render_template('index.htm')

@app.route('/result', methods=['POST'])
def result():
    user_responses = [int(request.form[f'question_{i+1}']) for i in range(20)]
    result = classify_mental_disorder('data/question_final.csv', user_responses)
    return render_template('result.htm', result=result)

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
        if i >= len(user_responses):
            break
        
        disorder = row['Mental_Disorder']
        response_value = user_responses[i]
        disorder_scores[disorder] += response_value
    
    predicted_disorder = max(disorder_scores, key=disorder_scores.get)
    
    return predicted_disorder

def select_relevant_questions(predictions):
    top_disorders = np.argsort(predictions[0])[-2:]  # Select top 2 disorders
    df = pd.read_csv('data/questions_data.csv')
    selected_questions = df[df['Mental_Disorder'].isin(top_disorders)].head(20)
    return selected_questions.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)
