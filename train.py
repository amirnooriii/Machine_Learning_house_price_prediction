import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import storage
import os
import glob

# === Load environment variables ===
BUCKET_NAME = "data-analytics-prod-2847-kw-region"

# === Step 1: Find the latest CSV in the repo's data folder ===
csv_files = glob.glob("data/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files found in data directory.")

latest_csv = max(csv_files, key=os.path.getctime)
print(f"📂 Using data file: {latest_csv}")

# === Step 2: Load and preprocess data ===
df = pd.read_csv(latest_csv)

# Simple example: Assume 'target' column exists
# first column for sqare footage, second for number of bathrooms, third for price
X = df.drop(df.columns[2], axis=1)
y = df[df.columns[2]]

# Optional: split to check accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 3: Train model ===
model = LinearRegression(max_iter=500)
model.fit(X_train, y_train)

# === Step 4: Evaluate model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {acc:.4f}")

# === Step 5: Save model locally ===
os.makedirs("model", exist_ok=True)
model_path = "model/model.joblib"
joblib.dump(model, model_path)
print(f"💾 Model saved locally at {model_path}")

# === Step 6: Upload model to GCS ===
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob("model/model.joblib")
blob.upload_from_filename(model_path)

print(f"☁️ Model uploaded to gs://{BUCKET_NAME}/model/model.joblib")
