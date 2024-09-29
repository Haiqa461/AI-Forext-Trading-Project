import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\haiqa\OneDrive\Documents\Projects with Ahmed\Forex trading Project\AI-Forext-Trading-Project\Datasets\30min\September 2024 30min dataset.csv', sep=',', header=None)
df.columns = ['date', 'open', 'high', 'low', 'close']
df['date']= pd.to_datetime(df['date'])
df['date_only']= df['date'].dt.date
df['time_only']= df['date'].dt.time
df.drop('date', axis=1, inplace=True)
df.rename(columns={'date_only': 'date', 'time_only': 'time'}, inplace=True)

# Convert date and time to ordinal values
df['date'] = df['date'].apply(lambda x: x.toordinal())
df['time'] = df['time'].apply(lambda x: x.hour + x.minute/60)

# Plot the dataset
plt.figure(figsize=(12,6))
plt.plot(df['date'] + df['time'], df['open'], label='Open', color='blue')
plt.plot(df['date'] + df['time'], df['high'], label='High', color='red')
plt.plot(df['date'] + df['time'], df['low'], label='Low', color='green')
plt.plot(df['date'] + df['time'], df['close'], label='Close', color='black')

# Add title and labels
plt.title('EUR/USD Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')

# Add legend
plt.legend()

# Show the plot
plt.show()