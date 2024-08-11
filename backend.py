import pandas as pd

def classify_mental_disorder(csv_file_path, user_responses):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Print the column names to verify the correct names
    print("Columns in CSV:", df.columns)
    
    # Check if 'Mental_Disorder' column exists
    if 'Mental_Disorder' not in df.columns:
        raise KeyError("'Mental_Disorder' column is not found in the CSV file.")
    
    # Initialize a dictionary to keep track of scores for each disorder
    disorder_scores = {
        "Depression": 0,
        "Anxiety": 0,
        "ADHD": 0,
        "PTSD": 0,
        "Psychosis & Schizophrenia": 0,
        "Bipolar": 0
    }
    
    # Ensure that we only process as many rows as we have responses for
    for i, row in df.iterrows():
        if i >= len(user_responses):
            print("Warning: More questions than user responses. Stopping at available responses.")
            break
        
        disorder = row['Mental_Disorder']
        response_value = user_responses[i]
        
        if disorder in disorder_scores:
            disorder_scores[disorder] += response_value
    
    # Determine the disorder with the highest score
    predicted_disorder = max(disorder_scores, key=disorder_scores.get)
    
    return predicted_disorder

# Example usage (replace with actual user inputs):
user_responses = [2, 4, 3, 1]  # This should match the numerical response options in your CSV
csv_file_path = "data/question_final.csv"  # Replace with the actual path

try:
    result = classify_mental_disorder(csv_file_path, user_responses)
    print(f"The user is most likely experiencing: {result}")
except KeyError as e:
    print(e)
except IndexError as e:
    print(f"Error: {e}. Please ensure the number of responses matches the number of questions.")
