from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
import warnings


def parser(x):
    year = int(x)
    return pd.to_datetime(str(year))

# Read the data with the appropriate date format
series = read_csv('ev-sales.csv', header=0, parse_dates=[0], index_col=0, date_format='%Y')


# Suppress specific warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Train the ARIMA model using the entire dataset
model = ARIMA(series, order=(4, 1, 1))
model_fit = model.fit()

# Forecast for the years 2023 to 2050
forecast_years = pd.date_range(start='2023-01-01', end='2050-01-01', freq='AS-JAN')
forecast = model_fit.forecast(steps=len(forecast_years))  # Generate the forecast

# Plot the forecast
pyplot.plot(series.index, series.values, label='Historical Data')
pyplot.plot(forecast_years, forecast, color='red', label='Forecast')
pyplot.xlabel('Time')
pyplot.ylabel('EVSales')
pyplot.legend()
pyplot.show()
