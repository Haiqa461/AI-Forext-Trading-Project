from LSTM_model import *
new_df = pd.DataFrame({
    'date': ['2024-08-18 22:00:00', '2024-08-18 22:30:00', '2024-08-18 23:00:00', '2024-08-18 23:30:00',
             '2024-08-19 00:00:00', '2024-08-19 00:30:00', '2024-08-19 01:00:00', '2024-08-19 01:30:00',
             '2024-08-19 02:00:00', '2024-08-19 02:30:00', '2024-08-19 03:00:00', '2024-08-19 03:30:00',
             '2024-08-19 04:00:00', '2024-08-19 04:30:00'],
    'open': [1.10266, 1.10256, 1.10254, 1.10235, 1.1025, 1.10246, 1.10305, 1.10312,
             1.10312, 1.10302, 1.10309, 1.10368, 1.10392, 1.1039],
    'high': [1.10344, 1.10272, 1.10263, 1.10261, 1.10274, 1.10363, 1.10315, 1.10336,
             1.10315, 1.1032, 1.10397, 1.10405, 1.10402, 1.10419],
    'low': [1.10252, 1.10246, 1.1023, 1.10228, 1.10234, 1.10246, 1.10278, 1.10289,
            1.103, 1.1029, 1.10301, 1.10368, 1.10367, 1.1038],
    'close': [1.10256, 1.10255, 1.10235, 1.10248, 1.10245, 1.10304, 1.1031, 1.10312,
              1.10302, 1.10308, 1.10368, 1.10392, 1.1039, 1.10398]
})

new_df['date'] = pd.to_datetime(new_df['date'])
new_df['date_only'] = new_df['date'].dt.date
new_df['time_only'] = new_df['date'].dt.time
new_df.drop('date', axis=1, inplace=True)
new_df.rename(columns={'date_only': 'date', 'time_only': 'time'}, inplace=True)

new_df['date'] = new_df['date'].apply(lambda x: x.toordinal())
new_df['time'] = new_df['time'].apply(lambda x: x.hour + x.minute/60)

scaler = MinMaxScaler()
new_df[['date', 'time']] = scaler.fit_transform(new_df[['date', 'time']])
new_df[['open', 'high', 'low', 'close']] = scaler.fit_transform(new_df[['open', 'high', 'low', 'close']])

new_x, _ = create_sequences(new_df, seq_len=30)

# Reshape the data for the LSTM model
new_x = np.reshape(new_x, (new_x.shape[0], new_x.shape[1], 6))
predictions = model.predict(new_x)
top_10_predictions = predictions[:10]
print(top_10_predictions)