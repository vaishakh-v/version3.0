from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the saved model
model = load_model('models/question_selector_model.h5')

# Load the encoders and scaler
with open('models/le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('models/le_location.pkl', 'rb') as f:
    le_location = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example new user data
# Format: [Age, Gender, Location, Last_Response]
# Ensure that Gender and Location are encoded in the same way as during training
new_user_data = np.array([[30, 1, 2, 3]])  # Example: Age=30, Gender=1, Location=2, Response=3

# Standardize the new user data
new_user_data = scaler.transform(new_user_data)

# Predict relevant mental disorder
predictions = model.predict(new_user_data)
predicted_disorder = np.argmax(predictions, axis=1)

# Decode the predicted disorder back to its original label
disorder_name = le_disorder.inverse_transform(predicted_disorder)

print(f"Predicted mental disorder: {disorder_name[0]}")
