from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("dataset/Coffee_sales.csv")

# Encode categorical columns dulu
X = df[['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday']].copy()
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        X.loc[:, col] = encoders[col].fit_transform(X[col])

label_y = LabelEncoder()
y = label_y.fit_transform(df['coffee_name'])

# CBR dengan KNN (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Contoh kasus baru
new_case = pd.DataFrame([[10, "card", "Morning", "Fri"]],
                        columns=['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday'])

# Encode kasus baru
for col in new_case.columns:
    if col in encoders:
        new_case.loc[:, col] = encoders[col].transform(new_case[col])

# Inference
pred = knn.predict(new_case)[0]
print("CBR Prediction:", label_y.inverse_transform([pred])[0])
