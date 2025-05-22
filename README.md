# SalaryPredictionProject

# salary_prediction.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Salary_Data.csv")
print("Dataset Preview:\n", df)

# Visualize the relationship
sns.jointplot(x="Years of Experience", y="Salary", data=df)
plt.title("Experience vs Salary")
plt.show()

# Prepare input and output
x = df[["Years of Experience"]]  # Feature (2D for ML)
y = df["Salary"]                 # Target (Label)

print("\nFeature (x):\n", x)
print("\nLabel (y):\n", y)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nTesting *************")
print(x_test)
print(y_test)

print("\nTraining *************")
print(x_train)
print(y_train)
