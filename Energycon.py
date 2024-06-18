from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
data = pd.read_csv(r'D:\Chrome Downloads\Energy_consumption.csv')
# Convert the 'Timestamp' to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extracting additional time features
data['Hour'] = data['Timestamp'].dt.hour
data['Month'] = data['Timestamp'].dt.month

# Encoding categorical variables
categorical_features = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Selecting features and target
X = data_encoded.drop(['Timestamp', 'EnergyConsumption'], axis=1)
y = data_encoded['EnergyConsumption']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining models to train
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Training and evaluating models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'RMSE': rmse, 'R2': r2}

print(results)
