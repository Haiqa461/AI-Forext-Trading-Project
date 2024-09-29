import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
df = pd.read_csv(r'C:\Users\haiqa\OneDrive\Documents\Projects with Ahmed\Forex trading Project\AI-Forext-Trading-Project\Datasets\30min\September 2024 30min dataset.csv', sep=',', header=None)
df.columns = ['date', 'open', 'high', 'low', 'close']
df['date']= pd.to_datetime(df['date'])
df['date_only']= df['date'].dt.date
df['time_only']= df['date'].dt.time
df.drop('date', axis=1, inplace=True)
df.rename(columns={'date_only': 'date', 'time_only': 'time'}, inplace=True)

# Convert date and time to ordinal and float values
df['date'] = df['date'].apply(lambda x: x.toordinal())
df['time'] = df['time'].apply(lambda x: x.hour + x.minute/60)

# Scale data
scaler = MinMaxScaler()
df[['date', 'time']] = scaler.fit_transform(df[['date', 'time']])
df[['open', 'high', 'low', 'close']] = scaler.fit_transform(df[['open', 'high', 'low', 'close']])

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create sequences
def create_sequences(df, seq_len):
    x, y_open, y_high, y_low, y_close = [], [], [], [], []
    for i in range(len(df) - seq_len):
        x.append(df[['open', 'high', 'low', 'close', 'date', 'time']].iloc[i:i+seq_len].values)
        y_open.append(df['open'].iloc[i+seq_len])
        y_high.append(df['high'].iloc[i+seq_len])
        y_low.append(df['low'].iloc[i+seq_len])
        y_close.append(df['close'].iloc[i+seq_len])
    return np.array(x), np.array(y_open), np.array(y_high), np.array(y_low), np.array(y_close)

seq_len = 30
x_train, y_open_train, y_high_train, y_low_train, y_close_train = create_sequences(train_df, seq_len)
x_test, y_open_test, y_high_test, y_low_test, y_close_test = create_sequences(test_df, seq_len)

# Reshape data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))

# Build model
model = Sequential()
model.add(LSTM(units=20, return_sequences=True, input_shape=(x_train.shape[1], 6)))
model.add(LSTM(units=20))
model.add(Dense(4))  # output 4 values: open, high, low, close
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(x_train, [y_open_train, y_high_train, y_low_train, y_close_train], epochs=120, batch_size=32, validation_data=(x_test, [y_open_test, y_high_test, y_low_test, y_close_test]))

# Make predictions
y_pred_open, y_pred_high, y_pred_low, y_pred_close = model.predict(x_test)

# Convert ordinal dates back to datetime format
def convert_ordinal_to_datetime(ordinal_dates):
    return pd.to_datetime([datetime.date.fromordinal(int(x)) for x in ordinal_dates])

y_pred_dates = convert_ordinal_to_datetime(y_pred_open[:, -1])  # assuming the last column is the date

# Plot graph
fig, ax = plt.subplots()
ax.plot(y_pred_dates, y_pred_open, label='Open', color='b')
ax.plot(y_pred_dates, y_pred_high, label='High', color='g')
ax.plot(y_pred_dates, y_pred_low, label='Low', color='r')
ax.plot(y_pred_dates, y_pred_close, label='Close', color='y')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend(loc='upper left')
plt.show()