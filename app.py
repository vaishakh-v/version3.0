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
    for i, row in enumerate(df.iterrows()):
        if i >= len(user_responses):
            print("Warning: More questions than user responses. Stopping at available responses.")
            break
        
        disorder = row[1]['Mental_Disorder']
        response_value = user_responses[i]
        
        # Update the score for the corresponding mental disorder
        disorder_scores[disorder] += response_value
    
    # Determine the disorder with the highest score
    predicted_disorder = max(disorder_scores, key=disorder_scores.get)
    
    return predicted_disorder

# Example usage (replace with actual user inputs):
user_responses = [3, 2, 1, 4, 0, 3, 2, 1, 3, 4, 2, 3, 1, 2, 3, 0, 1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1]  
csv_file_path = "question_final.csv"

try:
    result = classify_mental_disorder(csv_file_path, user_responses)
    print(f"The user is most likely experiencing: {result}")
except KeyError as e:
    print(e)
except IndexError as e:
    print(f"Error: {e}. Please ensure the number of responses matches the number of questions.")
