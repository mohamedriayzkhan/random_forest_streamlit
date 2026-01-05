import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("data/dataset.csv")

# 🔧 Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Drop ID column
df.drop(columns=["loan_id"], inplace=True)

# Target column
TARGET_COLUMN = "loan_status"

# Encode categorical columns
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest model
model = Pipeline([
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Random Forest loan status model trained and saved as model.pkl")
