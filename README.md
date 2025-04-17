import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("FuelConsumption.csv")

# Check column names to match actual ones
print("Columns:", df.columns)

# Adjust column names (change these if your dataset uses different headers)
cyl_col = 'Cylinders' if 'Cylinders' in df.columns else df.columns[df.columns.str.contains('cyl', case=False)][0]
co2_col = 'CO2 Emissions' if 'CO2 Emissions' in df.columns else df.columns[df.columns.str.contains('co2', case=False)][0]
eng_col = 'Engine Size' if 'Engine Size' in df.columns else df.columns[df.columns.str.contains('engine', case=False)][0]
fuel_col = 'Fuel Consumption Comb (mpg)' if 'Fuel Consumption Comb (mpg)' in df.columns else df.columns[df.columns.str.contains('fuel', case=False) & df.columns.str.contains('comb', case=False)][0]

# Q1: Scatter plot - Cylinders vs CO2 Emissions
plt.scatter(df[cyl_col], df[co2_col], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 Emissions')
plt.show()

# Q2: Scatter plot - Cylinders & Engine Size vs CO2 Emissions
plt.scatter(df[cyl_col], df[co2_col], color='blue', label='Cylinders')
plt.scatter(df[eng_col], df[co2_col], color='red', label='Engine Size')
plt.xlabel('Feature Value')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders & Engine Size vs CO2 Emissions')
plt.legend()
plt.show()

# Q3: Scatter plot - Cylinders, Engine Size, Fuel Consumption vs CO2 Emissions
plt.scatter(df[cyl_col], df[co2_col], color='blue', label='Cylinders')
plt.scatter(df[eng_col], df[co2_col], color='red', label='Engine Size')
plt.scatter(df[fuel_col], df[co2_col], color='green', label='Fuel Consumption')
plt.xlabel('Feature Value')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders, Engine Size & Fuel Consumption vs CO2 Emissions')
plt.legend()
plt.show()

# Q4: Model with Cylinders
X = df[[cyl_col]]
y = df[co2_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model_cyl = LinearRegression()
model_cyl.fit(X_train, y_train)
pred_cyl = model_cyl.predict(X_test)
print("Accuracy with Cylinders:", r2_score(y_test, pred_cyl))

# Q5: Model with Fuel Consumption
X = df[[fuel_col]]
y = df[co2_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)
pred_fuel = model_fuel.predict(X_test)
print("Accuracy with Fuel Consumption:", r2_score(y_test, pred_fuel))

# Q6: Different test ratios
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(df[[fuel_col]], df[co2_col], test_size=ratio, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = r2_score(y_test, y_pred)
    print(f"Test ratio {ratio}: Accuracy = {acc}")



\![Screenshot 2025-04-17 154317](https://github.com/user-attachments/assets/b2980510-4350-42eb-a7a3-8dc30b0fc72d)

![Screenshot 2025-04-17 154332](https://github.com/user-attachments/assets/93f41d87-aab0-4ff1-940f-983cbd3b0386)

![Screenshot 2025-04-17 154345](https://github.com/user-attachments/assets/2eddb68f-35a3-4755-b9b4-f676cbb3f3ca)


