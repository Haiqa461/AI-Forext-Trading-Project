from scaling import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# building model
model = Sequential()
model.add(LSTM(units=20, return_sequences=True, input_shape=(x_train.shape[1], 6)))
model.add(LSTM(units=20))
# the dense layer is 1 due to only close price is predicting if the open, high, low, close is predicting then it would be 4.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=120, batch_size=32, validation_data=(x_test, y_test))
mse = model.evaluate(x_test, y_test)
print(f'MSE: {mse}')

train_mse = model.evaluate(x_train, y_train)
test_mse = model.evaluate(x_test, y_test)


train_accuracy = 1 - (train_mse / (np.max(y_train) - np.min(y_train)))
test_accuracy = 1 - (test_mse / (np.max(y_test) - np.min(y_test)))

print(f'Training MSE: {train_mse}')
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing MSE: {test_mse}')
print(f'Testing Accuracy: {test_accuracy:.2f}')

plt.plot(y_test)
plt.plot(model.predict(x_test))
plt.show()

history = model.fit(x_train, y_train, epochs=120, batch_size=32, validation_data=(x_test, y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model.save('LSTM_close.keras')