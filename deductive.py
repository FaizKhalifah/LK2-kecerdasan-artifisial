# Faiz
import pandas as pd

# 1. Load dataset
df = pd.read_csv("dataset/Coffee_sales.csv")

# 2. Pilih fitur & target
X = df[['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday']].copy()
y = df['coffee_name']

def deduce_coffee(hour_of_day, cash_type, time_of_day, weekday):
    if hour_of_day < 12:
        return "Espresso"
    elif time_of_day == "Afternoon" and cash_type == "Card":
        return "Latte"
    elif weekday in ["Saturday", "Sunday"]:
        return "Cappuccino"
    else:
        return "Americano"

# Contoh
print("Hasil deduksi : ")
print(deduce_coffee(9, "Cash", "Morning", "Monday"))   # Espresso
print(deduce_coffee(15, "Card", "Afternoon", "Tuesday")) # Latte