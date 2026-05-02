import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data.csv")

X = data[['Hours']]
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Actual:", y_test.values)
print("Predicted:", predictions)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Accuracy:", r2 * 100, "%")

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Student Score Predictor")
plt.show()