import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load Dataset
df = pd.read_csv("employee_salary.csv")

print("Original Data Loaded:", df.shape)


# Data Cleaning
df = df.drop_duplicates()

df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

print("After Cleaning:", df.shape)


# Create Salary Category
def salary_category(salary):
    if salary < 75000:
        return "Low"
    elif salary <= 110000:
        return "Medium"
    else:
        return "High"


df["salary_category"] = df["Monthly_Salary"].apply(
    salary_category
)


# Select Features
features = [
    "Department",
    "Experience_Years",
    "Education_Level",
    "Age",
    "Gender",
    "City"
]

X = df[features]
y = df["salary_category"]


# Convert Categorical Data
label_encoders = {}

for column in X.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(X[column])

    label_encoders[column] = encoder


# Convert Target
target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(
    "Decision Tree Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)


# Save Model
joblib.dump(model, "salary_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("salary_model.pkl created successfully!")
print("label_encoders.pkl created successfully!")
