import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Features
X = data[["Fever", "Cough", "Fatigue", "Difficulty Breathing"]]

# Labels
y = data["Disease"]

# Encode disease names
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("AI Accuracy:", round(accuracy * 100, 2), "%")

# Save trained model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")

# Prediction function
def predict_disease(features):

    prediction = model.predict([features])

    probabilities = model.predict_proba([features])

    confidence = max(probabilities[0]) * 100

    disease = encoder.inverse_transform(prediction)

    return disease[0], round(confidence, 2)