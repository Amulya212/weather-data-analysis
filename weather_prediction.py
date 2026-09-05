import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# WEATHER DATA ANALYSIS & PREDICTION
# -------------------------------

# 1. Load the dataset
data = pd.read_csv("weather_data.csv")

print("\n===== WEATHER DATA =====")
print(data.head())

print("\n===== BASIC INFORMATION =====")
print(data.describe())

# 2. Average weather conditions
print("\n===== AVERAGES =====")
print("Average Temperature:", round(data["Temperature_C"].mean(), 2), "°C")
print("Average Humidity:", round(data["Humidity_Percent"].mean(), 2), "%")
print("Average Rainfall:", round(data["Rainfall_mm"].mean(), 2), "mm")
print("Average Wind Speed:", round(data["Wind_Speed_kmh"].mean(), 2), "km/h")

# 3. Temperature trend
plt.figure(figsize=(9, 5))
plt.plot(data["Day"], data["Temperature_C"], marker="o")
plt.title("Temperature Trend")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Humidity vs Rainfall
plt.figure(figsize=(9, 5))
plt.scatter(data["Humidity_Percent"], data["Rainfall_mm"])
plt.title("Humidity vs Rainfall")
plt.xlabel("Humidity (%)")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Machine Learning: predict temperature from other weather values
X = data[["Humidity_Percent", "Rainfall_mm", "Wind_Speed_kmh"]]
y = data["Temperature_C"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n===== MODEL RESULTS =====")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# 6. User input for prediction
print("\n===== WEATHER PREDICTION =====")
try:
    humidity = float(input("Enter humidity (%): "))
    rainfall = float(input("Enter rainfall (mm): "))
    wind = float(input("Enter wind speed (km/h): "))

    new_weather = pd.DataFrame({
        "Humidity_Percent": [humidity],
        "Rainfall_mm": [rainfall],
        "Wind_Speed_kmh": [wind]
    })

    predicted_temp = model.predict(new_weather)[0]
    print("\nPredicted Temperature:", round(predicted_temp, 2), "°C")

except ValueError:
    print("Please enter numbers only.")

print("\nProject completed successfully!")
