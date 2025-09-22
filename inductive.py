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

# 6. Evaluasi
y_pred = model.predict(X_test)
print("=== Hasil Induksi (ML) ===")
print("Akurasi:", model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_y.classes_))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_y.classes_, yticklabels=label_y.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Coffee Prediction")
plt.show()