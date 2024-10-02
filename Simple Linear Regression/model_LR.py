from cleaning_csv import dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

x = dataset[['open', 'high', 'low', 'volume']]
y = dataset[['close']]

x_train, x_temp, y_train, y_temp =train_test_split(x, y, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)

print(f'Training set size: {x_train.shape[0]} samples')
print(f'Test set size: {x_test.shape[0]} samples')
print(f'Validation set size: {x_val.shape[0]} samples')

model = LinearRegression()
model.fit(x_train, y_train)

train_prediction = model.predict(x_train)
test_predictions = model.predict(x_test)

plt.scatter(y_test, test_predictions)
plt.xlabel('Actual Prediction')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(y_test, test_predictions)
plt.plot(y_test, test_predictions,color='blue', linewidth=0.5)
plt.title('Actual vs Predicted Prices (Training Evaluation)')
plt.xlabel('Actual Prediction')
plt.ylabel('Predicted Prices')
plt.show()

with open("linear_regression.pkl", 'wb') as f:
    pickle.dump(model, f)

