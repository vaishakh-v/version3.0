import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the data
questions_df = pd.read_csv('data/questions_data.csv')
responses_df = pd.read_csv('data/user_responses.csv')

# Merge the dataframes on 'Question_ID'
data = pd.merge(responses_df, questions_df, on='Question_ID')

# Encode categorical variables (Gender, Location, Mental_Disorder)
le_gender = LabelEncoder()
le_location = LabelEncoder()
le_disorder = LabelEncoder()

data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Location'] = le_location.fit_transform(data['Location'])
data['Mental_Disorder'] = le_disorder.fit_transform(data['Mental_Disorder'])

# Save the encoders for later use in prediction
with open('models/le_gender.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

with open('models/le_location.pkl', 'wb') as f:
    pickle.dump(le_location, f)

with open('models/le_disorder.pkl', 'wb') as f:
    pickle.dump(le_disorder, f)

# Features and labels
X = data[['Age', 'Gender', 'Location', 'Response']].values
y = data['Mental_Disorder'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('models/question_selector_model.h5')
print("Model saved as 'models/question_selector_model.h5'.")
