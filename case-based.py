from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- CBR CYCLE: Retrieve, Reuse, Revise, Retain ---

# Load dataset
csv_file = "dataset/Coffee_sales.csv"
df = pd.read_csv(csv_file)

# Encode categorical columns
X = df[['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday']].copy()
print(X.shape)
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])

label_y = LabelEncoder()
y = label_y.fit_transform(df['coffee_name'])
X['solution'] = y   # simpan solusi ke dalam basis kasus

# CBR RETRIEVE: pakai NearestNeighbors (mirip KNN, tapi untuk cari tetangga)
nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
nn.fit(X.drop(columns=['solution']))

# Kasus baru
new_case = pd.DataFrame([[10, "card", "Morning", "Fri"]],
                        columns=['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday'])

# Encode new case
for col in new_case.columns:
    if col in encoders:
        new_case[col] = encoders[col].transform(new_case[col])

# Retrieve tetangga terdekat
distances, indices = nn.kneighbors(new_case)

neighbors = X.iloc[indices[0]]
suggested_solutions = neighbors['solution'].value_counts().idxmax()
prediction = label_y.inverse_transform([suggested_solutions])[0]

print("CBR Prediction:", prediction)

# --- CBR REVISE ---
# (contoh sederhana: minta feedback user)
feedback = input(f"Apakah prediksi '{prediction}' benar? (y/n): ")
if feedback.lower() == 'n':
    correct_solution = input("Masukkan solusi yang benar: ")
    correct_solution_enc = label_y.transform([correct_solution])[0]
    # REVISE hasil
    prediction = correct_solution

# --- CBR RETAIN ---
# simpan kasus baru + solusi (baik prediksi yang benar, atau hasil revisi)
new_case['solution'] = label_y.transform([prediction])[0]
X = pd.concat([X, new_case], ignore_index=True)

print("Basis kasus sekarang punya:", len(X), "kasus")