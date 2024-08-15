from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Assuming you have a DataFrame df with 78 questions as features and multiple disorders as labels
X = df.iloc[:, :-n_disorders]  # Features (78 questions)
y = df.iloc[:, -n_disorders:]  # Labels (disorders)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate mutual information for each question with respect to each disorder
mi_scores = np.array([mutual_info_classif(X_train, y_train[col], random_state=42) for col in y_train.columns])

# Average the mutual information scores across all disorders to get a global importance score
average_mi_scores = np.mean(mi_scores, axis=0)

# Select the top questions based on the average mutual information scores
selected_questions = np.argsort(average_mi_scores)[-20:]  # Selecting top 20 questions

# Filter the training and testing sets to include only the selected questions
X_train_reduced = X_train.iloc[:, selected_questions]
X_test_reduced = X_test.iloc[:, selected_questions]

# Train a Random Forest model with the reduced set of questions
model_reduced = RandomForestClassifier(random_state=42)
model_reduced.fit(X_train_reduced, y_train)

# Evaluate the performance
accuracy = model_reduced.score(X_test_reduced, y_test)
print("Accuracy with reduced questions:", accuracy)

# Output the selected questions
selected_questions_info = X.columns[selected_questions]
print("Selected questions based on mutual information and feature importance:")
for question in selected_questions_info:
    print(question)
