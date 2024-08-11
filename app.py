from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        location = request.form['location']
        trauma = int(request.form['trauma'])
        
        # Placeholder for machine learning model to select relevant questions based on the user's parameters
        selected_questions = select_relevant_questions(age, gender, location, trauma)
        
        return render_template('questions.htm', selected_questions=selected_questions)

    return render_template('index.htm')

@app.route('/result', methods=['POST'])
def result():
    user_responses = [int(request.form[f'question_{i+1}']) for i in range(len(request.form)-1)] # Adjusted to dynamically count questions
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

def select_relevant_questions(age, gender, location, trauma):
    # Implement machine learning model to return a subset of questions based on user parameters
    # For now, this is a placeholder that selects all questions
    df = pd.read_csv('data/question_final.csv')
    return df.to_dict(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
