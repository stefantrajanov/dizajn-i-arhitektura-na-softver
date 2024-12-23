from fastapi import APIRouter
import pandas as pd
import numpy as np

router = APIRouter()
data = pd.read_csv('app/data-formatted.csv')

# def string_to_float(column):
#     if column == 'DATE':
#         return data[column]
#     if column == 'TOTAL REVENUE IN DENARS':
#         data[column] = data['TOTAL REVENUE IN DENARS'].str.replace('.', '').astype(float)
#         return data[column]
#     try:
#         data[column] = data[column].str.replace(',', '.')
#         data[column] = data[column].str.replace('.', '', 1)
#         data[column] = data[column].astype(float)
#     except:
#         return data[column]
#     return data[column]
# data.apply(lambda x: string_to_float(x.name))

# Function to calculate technical indicators
def calculate_technical_indicators(data, column="PRICE OF LAST TRANSACTION"):
    # Ensure data is sorted by date
    data = data.sort_values(by="DATE").reset_index(drop=True)

    # Oscillators
    delta = data[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    data["EMA12"] = data[column].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data[column].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA12"] - data["EMA26"]

    # Stochastic Oscillator
    data["L14"] = data[column].rolling(window=14).min()
    data["H14"] = data[column].rolling(window=14).max()
    data["Stochastic"] = (data[column] - data["L14"]) / (data["H14"] - data["L14"]) * 100

    # Williams %R
    data["Williams %R"] = (data["H14"] - data[column]) / (data["H14"] - data["L14"]) * -100

    # Rate of Change
    data["ROC"] = data[column].pct_change(periods=12)

    # Moving Averages
    for window in [10, 20, 50]:
        data[f"SMA{window}"] = data[column].rolling(window=window).mean()
        data[f"EMA{window}"] = data[column].ewm(span=window, adjust=False).mean()

    # Meters
    oscillators_meter = "STRONG BUY" if data["RSI"].iloc[-1] > 70 else "NEUTRAL"
    moving_averages_meter = "STRONG BUY" if data[f"SMA10"].iloc[-1] < data[column].iloc[-1] else "SELL"

    return data, oscillators_meter, moving_averages_meter

# Example API route
@router.get("/stock-data/{ticker}")
async def get_stock_data(ticker: str):
    # Replace this with your actual data loading logic
    # Assuming `data` is a DataFrame loaded from a CSV or database
    stock_data = data[data["COMPANY"] == ticker]  # Filter data for the ticker

    if stock_data.empty:
        return {"error": "Ticker not found"}

    # Calculate indicators and meters
    stock_data, oscillators_meter, moving_averages_meter = calculate_technical_indicators(stock_data)
    stock_data.fillna(0, inplace=True)
    # Construct response
    response = {
        "Ticker": ticker,
        "Company Name": stock_data["COMPANY"].iloc[0],
        "Current Price": stock_data[stock_data["DATE"].max() == stock_data["DATE"]]["PRICE OF LAST TRANSACTION"].iloc[0],
        "MAX Price": stock_data["MAX"].max(),
        "MIN Price": stock_data["MIN"].min(),
        "Volume": stock_data["QUANTITY"].sum(),
        "REVENUE": stock_data["REVENUE IN BEST DENARS"].sum(),
        "AVERAGE PRICE": stock_data[stock_data["DATE"].max() == stock_data["DATE"]]["AVERAGE PRICE"].iloc[0],
        "Oscillators": {
            "RSI": stock_data["RSI"].iloc[-1],
            "MACD": stock_data["MACD"].iloc[-1],
            "Stochastic Oscillator": stock_data["Stochastic"].iloc[-1],
            "Williams %R": stock_data["Williams %R"].iloc[-1],
            "Rate of Change": stock_data["ROC"].iloc[-1],
            "METER": oscillators_meter,
        },
        "Moving Averages": {
            "SMA10": stock_data["SMA10"].iloc[-1],
            "EMA10": stock_data["EMA10"].iloc[-1],
            "SMA20": stock_data["SMA20"].iloc[-1],
            "EMA20": stock_data["EMA20"].iloc[-1],
            "SMA50": stock_data["SMA50"].iloc[-1],
            "METER": moving_averages_meter,
        },
    }

    return response