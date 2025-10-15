import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("dataset/Coffee_sales.csv")

# 2. Pilih fitur & target (tambahkan Month_name)
X = df[['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday', 'Month_name']].copy()
y = df['coffee_name']

# 3. Encode kategori
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        X.loc[:, col] = encoders[col].fit_transform(X[col])

label_y = LabelEncoder()
y = label_y.fit_transform(y)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Induktif (RandomForest)
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # bantu kalau data tidak seimbang
    random_state=42
)
model.fit(X_train, y_train)

new_data = pd.DataFrame([{
    "hour_of_day": 9,
    "cash_type": "card",
    "Time_of_Day": "Morning",
    "Weekday": "Mon",
    "Month_name": "Jan"
}])

# Encode dengan encoder yang sudah dipakai
for col in new_data.columns:
    if col in encoders:
        new_data.loc[:, col] = encoders[col].transform(new_data[col])

# Prediksi
y_new_pred = model.predict(new_data)
coffee_pred = label_y.inverse_transform(y_new_pred)

print("\n=== Prediksi Data Baru ===")
print(new_data)
print("Rekomendasi kopi:", coffee_pred[0])