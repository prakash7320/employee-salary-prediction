
import pandas as pd
import joblib
# Load the saved model
model = joblib.load("salary_model.pkl")

# Load Test Data
data_df = pd.read_csv("test_data.csv")

# Select features
x = data_df[
    [
        "Department",
        "Experience_Years",
        "Education_Level",
        "Age",
        "Gender",
        "City"
    ]
].copy()


# Load encoders
label_encoders = joblib.load("label_encoders.pkl")


# Convert categorical columns
for column, encoder in label_encoders.items():
    x[column] = encoder.transform(x[column])


# Predict
y_pred = model.predict(x)


# Print Prediction
print("Predictions:", y_pred)

