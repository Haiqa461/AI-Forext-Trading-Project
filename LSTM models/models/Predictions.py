import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained models
model_close = load_model('LSTM_close.keras')
model_open = load_model('LSTM_open.keras')
model_high = load_model('LSTM_high.keras')
model_low = load_model('LSTM_low.keras')

# Define the number of timestamps (from 8:00 AM to 4:00 PM with 30-minute intervals)
num_timestamps = 17

# Create an input sequence of zeros with shape (17, 30, 6)
input_sequence = np.zeros((num_timestamps, 30, 6))

# Generate predicted values for each model
predicted_close = model_close.predict(input_sequence)
predicted_open = model_open.predict(input_sequence)
predicted_high = model_high.predict(input_sequence)
predicted_low = model_low.predict(input_sequence)

# Create timestamps from 8:00 AM to 4:00 PM with 30-minute intervals
timestamps = pd.date_range(start='2024-09-29 08:00:00', end='2024-09-29 16:00:00', freq='30min')

# Plot the predicted values
plt.plot(timestamps, predicted_close[:, 0], label='Close', color='black')
plt.plot(timestamps, predicted_open[:, 0], label='Open', color='blue')
plt.plot(timestamps, predicted_high[:, 0], label='High', color='red')
plt.plot(timestamps, predicted_low[:, 0], label='Low', color='green')

# Add title and labels
plt.title('Predicted EUR/USD Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')

# Add legend
plt.legend()

# Show the plot
plt.show()