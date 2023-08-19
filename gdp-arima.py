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


def append_forecast_to_data(data, forecast, forecast_years):
    forecast_data = pd.DataFrame({'Forecast': forecast}, index=forecast_years)
    combined_data = pd.concat([data, forecast_data])
    return combined_data


def export_to_csv(data, file_path):
    data.to_csv(file_path)


def main():
    file_path = 'ev-sales.csv'
    start_date = '2023-01-01'
    end_date = '2050-01-01'
    
    data = load_data(file_path)
    model_fit = train_arima_model(data)
    forecast = generate_forecast(model_fit, start_date, end_date)
    forecast_years = pd.date_range(start=start_date, end=end_date, freq='AS-JAN')
    
    combined_data = append_forecast_to_data(data, forecast, forecast_years)
    export_to_csv(combined_data, 'combined_ev_sales.csv')


if __name__ == "__main__":
    main()
