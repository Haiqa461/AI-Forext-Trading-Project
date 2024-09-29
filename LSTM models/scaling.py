from data_cleaning import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

scaler = MinMaxScaler()
df[['date', 'time']] = scaler.fit_transform(df[['date', 'time']])
df[['open', 'high', 'low', 'close']] = scaler.fit_transform(df[['open', 'high', 'low', 'close']])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# sequence creation
def create_sequences(df, seq_len):
    x, y = [], []
    for i in range(len(df) - seq_len):
        x.append(df[['open', 'high', 'low', 'close', 'date', 'time']].iloc[i:i+seq_len].values)
        y.append(df['close'].iloc[i+seq_len])
    return np.array(x), np.array(y)


seq_len = 30
x_train, y_train = create_sequences(train_df, seq_len)
x_test, y_test = create_sequences(test_df, seq_len)

# reshaping of the data for the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))
print(x_train.shape)