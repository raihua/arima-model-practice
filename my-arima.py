from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_data(file_path):
    series = read_csv(file_path, header=0, parse_dates=[0], index_col=0, date_format='%Y')
    return series

def train_arima_model(data, order=(4, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def generate_forecast(model_fit, start_date, end_date):
    forecast_years = pd.date_range(start=start_date, end=end_date, freq='AS-JAN')
    forecast = model_fit.forecast(steps=len(forecast_years))
    return forecast

def plot_forecast(data, forecast_years, forecast):
    plt.plot(data.index, data.values, label='Historical Data')
    plt.plot(forecast_years, forecast, color='red', label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('EVSales')
    plt.legend()
    plt.show()

def main():
    file_path = 'ev-sales.csv'
    start_date = '2023-01-01'
    end_date = '2050-01-01'
    
    data = load_data(file_path)
    model_fit = train_arima_model(data['Sales'])
    forecast = generate_forecast(model_fit, start_date, end_date)
    forecast_years = pd.date_range(start=start_date, end=end_date, freq='AS-JAN')
    
    # Create a new DataFrame with forecasted values and switch the columns
    forecast_data = pd.DataFrame({'Year': forecast_years, 'Sales': forecast})
    
    # Combine the original data and forecast data
    combined_data = pd.concat([data, forecast_data.set_index('Year')]).sort_index()
    
    # Export the combined data to a CSV file
    combined_data.to_csv('extended_ev_sales.csv')

    plot_forecast(data, forecast_years, forecast)

if __name__ == "__main__":
    main()
