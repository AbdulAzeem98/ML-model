import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('Housing.csv')  # Ensure you have a CSV file with housing data

# Define features (X) and target (y)
X = data[['area', 'bedrooms', 'bathrooms','location_score']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as house_price_model.pkl")
