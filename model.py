import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("transaction_data.csv")

# Step 1: Convert 'Fraud Flag' to 0/1
df['Fraud Flag'] = df['Fraud Flag'].map({True: 1, False: 0})

# Step 2: Encode categorical columns
encoders = {}
for col in ['Transaction Type', 'Transaction Status', 'Device Used', 'Network Slice ID']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Step 3: Select features
features = ['Transaction Amount', 'Transaction Type', 'Transaction Status',
            'Device Used', 'Network Slice ID', 'Latency (ms)',
            'Slice Bandwidth (Mbps)', 'PIN Code']
X = df[features]
y = df['Fraud Flag']

# Step 4: Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 8: Save model, scaler, encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
