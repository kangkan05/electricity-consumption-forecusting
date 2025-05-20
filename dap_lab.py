import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("/Users/kangkanjyotiborah/Downloads/DAP lab.csv")
df = df.drop_duplicates()
df = df.dropna(thresh=len(df.columns) - 2)

df['wdir'] = df['wdir'].fillna(method='ffill')
df['moving_avg_3'] = df['moving_avg_3'].fillna(method='bfill')
df['datetime'] = pd.to_datetime(df['datetime'])

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

q_low = df['power_demand'].quantile(0.01)
q_high = df['power_demand'].quantile(0.99)

df = df[(df['power_demand'] >= q_low) & (df['power_demand'] <= q_high)]

scaler = MinMaxScaler()
color_to_normalize = ['power_demand', 'temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']

print(df.info())
print(df.describe())