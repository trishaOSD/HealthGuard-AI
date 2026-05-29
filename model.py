import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and labels
X = data[["Fever", "Cough", "Fatigue", "Difficulty Breathing"]]
y = data["Disease"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Better AI Model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)

# Accuracy check
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("AI Accuracy:", round(accuracy * 100, 2), "%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Prediction function
def predict_disease(features):

    prediction = model.predict([features])

    probabilities = model.predict_proba([features])

    confidence = max(probabilities[0]) * 100

    disease = encoder.inverse_transform(prediction)

    return disease[0], round(confidence, 2)