import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/superstore.csv', encoding='latin1')

# Prepare data for Prophet
df['Order Date'] = pd.to_datetime(df['Order Date'])
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Prophet needs columns named 'ds' and 'y'
daily_sales = daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Initialize and fit model
model = Prophet()
model.fit(daily_sales)

# Make future dataframe for 90 days
future = model.make_future_dataframe(periods=90)

# Predict sales
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title('Sales Forecast')
plt.show()
