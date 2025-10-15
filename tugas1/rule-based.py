import pandas as pd

df = pd.read_csv("dataset/Coffee_sales.csv")

def rule_based_inference(row):
    hour = row['hour_of_day']
    cash = row['cash_type']
    time_day = row['Time_of_Day']
    weekday = row['Weekday']

    if hour < 12:
        return "Espresso"
    elif time_day == "Afternoon" and cash == "Card":
        return "Latte"
    elif weekday in ["Saturday", "Sunday"]:
        return "Cappuccino"
    else:
        return "Americano"

# Contoh inference ke seluruh dataset
df['RBS_prediction'] = df.apply(rule_based_inference, axis=1)
print(df[['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday', 'RBS_prediction']].head())
