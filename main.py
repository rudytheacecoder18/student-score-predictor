import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[['Hours']]
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Actual:", y_test.values)
print("Predicted:", predictions)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Student Score Predictor")
plt.show()