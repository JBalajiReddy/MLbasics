import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Load the dataset
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"
df = pd.read_csv(URL)
print(df)
# Plot the data
df.plot.scatter(x="Horsepower", y="MPG")
plt.show()

# Prepare the features and target
target = df["MPG"]
features = df[["Horsepower", "Weight"]]

# Create and fit the model
lr = LinearRegression()
lr.fit(features, target)

# Model score
score = lr.score(features, target)
print(f"Model score: {score}")

# Make a prediction
prediction = lr.predict([[100, 2000]])
print(f"Prediction for Horsepower=100 and Weight=2000: {prediction}")
