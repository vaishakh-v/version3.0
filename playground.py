import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and Preprocess the Data
data = pd.read_csv('data/question_final.csv')

# Encode the mental disorder labels
label_encoder = LabelEncoder()
data['Mental_Disorder_Encoded'] = label_encoder.fit_transform(data['Mental_Disorder'])

# Extract feature names (this should match the columns used for training)
feature_names = [col for col in data.columns if col not in ['Question', 'Mental_Disorder', 'Mental_Disorder_Encoded']]

# Separate features and target
X = data[feature_names]
y = data['Mental_Disorder_Encoded']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to ask questions and get responses
def ask_question(question, choices):
    print(question)
    for i, choice in enumerate(choices):
        print(f"{i}: {choice}")
    answer = int(input("Your response (enter the number): "))
    return answer

# Function to select random questions from different disorders
def select_random_questions(data, num_questions=10):
    questions = []
    selected_disorders = set()
    
    while len(questions) < num_questions:
        potential_question = data.sample(1).iloc[0]
        disorder = potential_question['Mental_Disorder']
        
        if disorder not in selected_disorders or len(selected_disorders) == len(data['Mental_Disorder'].unique()):
            questions.append(potential_question)
            selected_disorders.add(disorder)
    
    return questions

# Predict the disorder based on user responses
def predict_disorder(model, questions):
    responses = {}
    
    for question in questions:
        response = ask_question(question['Question'], ['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'])
        responses[question.name] = response
    
    # Align responses with the feature names expected by the model
    response_data = pd.DataFrame([responses], columns=feature_names)
    
    # Predict using the trained model
    prediction = model.predict(response_data)
    disorder = label_encoder.inverse_transform(prediction)[0]
    return disorder

# Main Function to Perform Adaptive Prediction
def adaptive_predict_disorder(model, data):
    total_questions = 0
    max_questions = 15
    while total_questions < max_questions:
        questions = select_random_questions(data, num_questions=10)
        disorder = predict_disorder(model, questions)
        print(f"\nPredicted Mental Disorder: {disorder}")

        total_questions += len(questions)
        
        if total_questions >= 15:
            print("The model is unable to confidently determine the disorder. Please provide more information.")
            continue_more = input("Would you like to answer 10 more questions? (yes/no): ").strip().lower()
            if continue_more == 'yes':
                max_questions += 10
            else:
                print("Thank you for participating. Please consult a professional for further evaluation.")
                break
        else:
            break

# Example of running the adaptive prediction
adaptive_predict_disorder(model, data)
