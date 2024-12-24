from fastapi import APIRouter
import pandas as pd
import numpy as np

router = APIRouter()
data = pd.read_csv('app/data-formatted.csv')

# Function to resample data for timeframes
def resample_data(data, timeframe):
    data["DATE"] = pd.to_datetime(data["DATE"])  # Ensure DATE is in datetime format
    data = data.set_index("DATE")  # Set DATE as the index

    # Select only numeric columns for resampling
    numeric_columns = data.select_dtypes(include=["number"]).columns
    non_numeric_columns = data.select_dtypes(exclude=["number"]).columns

    if timeframe == "1D":
        resampled_data = data[numeric_columns].asfreq("D").fillna(method="ffill")
    elif timeframe == "1W":
        resampled_data = data[numeric_columns].resample("W").mean().fillna(0)
    elif timeframe == "1M":
        resampled_data = data[numeric_columns].resample("M").mean().fillna(0)
    else:
        raise ValueError("Invalid timeframe. Choose '1D', '1W', or '1M'.")

    # Reset the index to bring DATE back as a column
    resampled_data = resampled_data.reset_index()

    # Reattach non-numeric columns (e.g., COMPANY)
    if not non_numeric_columns.empty:
        non_numeric_data = data[non_numeric_columns].reset_index().drop_duplicates(subset="DATE")
        resampled_data = resampled_data.merge(non_numeric_data, on="DATE", how="left")

    return resampled_data

# Function to calculate technical indicators
def calculate_technical_indicators(data, column="PRICE OF LAST TRANSACTION"):
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


@router.get("/stock-data/{ticker}")
async def get_stock_data(ticker: str):
    print(f"Fetching data for ticker: {ticker}")

    stock_data = data[data["COMPANY"] == ticker]
    if stock_data.empty:
        print("No data found for the given ticker.")
        return {"error": "Ticker not found"}

    # Process for each timeframe
    timeframes = ["1D", "1W", "1M"]
    timeframe_results = {}
    for timeframe in timeframes:
        try:
            resampled_data = resample_data(stock_data, timeframe)

            # Replace NaN/Inf/-Inf in resampled data
            resampled_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

            indicators_data, oscillators_meter, moving_averages_meter = calculate_technical_indicators(resampled_data)

            # Replace NaN/Inf/-Inf in indicators_data
            indicators_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

            timeframe_results[timeframe] = {
                "GraphData": indicators_data[["DATE", "PRICE OF LAST TRANSACTION"]].to_dict(orient="records"),
                "Oscillators": {
                    "RSI": indicators_data["RSI"].iloc[-1],
                    "MACD": indicators_data["MACD"].iloc[-1],
                    "Stochastic Oscillator": indicators_data["Stochastic"].iloc[-1],
                    "Williams %R": indicators_data["Williams %R"].iloc[-1],
                    "Rate of Change": indicators_data["ROC"].iloc[-1],
                    "METER": oscillators_meter,
                },
                "Moving Averages": {
                    "SMA10": indicators_data["SMA10"].iloc[-1],
                    "EMA10": indicators_data["EMA10"].iloc[-1],
                    "SMA20": indicators_data["SMA20"].iloc[-1],
                    "EMA20": indicators_data["EMA20"].iloc[-1],
                    "SMA50": indicators_data["SMA50"].iloc[-1],
                    "METER": moving_averages_meter,
                },
            }
        except Exception as e:
            print(f"Error processing timeframe {timeframe}: {e}")
            timeframe_results[timeframe] = {"error": str(e)}

    # Replace NaN/Inf/-Inf in stock_data
    stock_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # Construct response
    response = {
        "Ticker": ticker,
        "Company Name": stock_data["COMPANY"].iloc[0],
        "Current Price": stock_data["PRICE OF LAST TRANSACTION"].iloc[-1],
        "MAX Price": stock_data["PRICE OF LAST TRANSACTION"].max(),
        "MIN Price": stock_data["PRICE OF LAST TRANSACTION"].min(),
        "Volume": stock_data["QUANTITY"].sum() if "QUANTITY" in stock_data.columns else None,
        "REVENUE": stock_data[
            "REVENUE IN BEST DENARS"].sum() if "REVENUE IN BEST DENARS" in stock_data.columns else None,
        "AVERAGE PRICE": stock_data["PRICE OF LAST TRANSACTION"].mean(),
        "Timeframes": timeframe_results,
    }

    # Ensure JSON compliance
    response = {
        key: (float(value) if isinstance(value, (np.float64, np.int64)) else value)
        for key, value in response.items()
    }

    return response