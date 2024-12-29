from fastapi import APIRouter
import pandas as pd
import numpy as np
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import logging
from transformers import logging as transformers_logging
router = APIRouter()
data = pd.read_csv('app/data-formatted.csv')
model = load_model('ai-models/lstm_model.h5')

def getBerzaNews(symbol):
    url = f'https://www.mse.mk/mk/symbol/{symbol}'
    response = requests.get(url)
    content = BeautifulSoup(response.text, 'lxml')

    # finding links to news
    aElements = content.find_all('a', href=True)
    newsLinks = [link['href'] for link in aElements if link['href'].startswith('/mk/news')]

    news = []
    for link in newsLinks:
        response = requests.get(f'https://www.mse.mk{link}')
        content = BeautifulSoup(response.text, 'lxml')
        try:
            # print(content.find(id='content').text)
            # print('-----------------------------------')
            news.append(content.find(id='content').text)
        except Exception as e:
            continue

    return news


# Load a multilingual model
def analyzeSentiment(symbol):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)  # Use TF equivalent class

    # Create a pipeline
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, framework="tf")

    # Analyze sentiment
    texts = getBerzaNews(symbol)

    scores = []
    for text in texts:
        try:
            result = sentiment_analysis(text)
            # print(f"Sentiment: {result[0]['score']}")
            scores.append(result[0]['score'])
        except Exception as e:
            continue

    if not scores:
        return 'No news for ' + symbol

    avg = sum(scores) / len(scores)
    if avg > 0.55:
        return 'Buy'
    else:
        return 'Sell'


def preprocess_and_predict(input_data, model_path):
    input_data = input_data.drop(columns=['COMPANY', 'DATE', 'PRICE OF LAST TRANSACTION'])
    """
    Preprocess input data and predict the price using a pre-trained LSTM model.

    Args:
        input_data (pd.DataFrame): Input dataset for prediction.
        model_path (str): Path to the trained model file.

    Returns:
        np.array: Predicted prices.
    """
    # Load the pre-trained model
    timesteps = model.input_shape[1]
    features = model.input_shape[2]

    # Ensure input_data has the correct number of features
    input_data = input_data.iloc[:, :features]

    # Handle missing values and normalize
    input_data = input_data.fillna(0)
    max_value = input_data.max().max()
    input_data_normalized = input_data / max_value

    # Check if there are enough rows for timesteps
    if len(input_data) < timesteps:
        raise ValueError(f"Input data must have at least {timesteps} rows for prediction.")

    # Reshape the data
    input_data_reshaped = np.array([input_data_normalized.values[-timesteps:]])
    input_data_reshaped = input_data_reshaped.reshape(1, timesteps, features)

    # Predict
    predictions = model.predict(input_data_reshaped)

    # Denormalize predictions if necessary
    predictions_denormalized = predictions * max_value

    return predictions_denormalized

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

            price_prediction = preprocess_and_predict(indicators_data, 'ai-models/lstm_model.h5')

            market_news_evaluation = analyzeSentiment(ticker)


            timeframe_results[timeframe] = {
                "Price Prediction": price_prediction[0][0],
                "Market News Evaluation": market_news_evaluation,
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

    # reverse stock data
    stock_data = stock_data.iloc[::-1]

    # Construct response
    response = {
        "Ticker": ticker,
        "Company Name": stock_data["COMPANY"].iloc[0],
        "Current Price": stock_data["AVERAGE PRICE"].iloc[-1],
        "MAX Price": stock_data["PRICE OF LAST TRANSACTION"].max(),
        "MIN Price": stock_data["PRICE OF LAST TRANSACTION"].min(),
        "Volume": stock_data["QUANTITY"].sum() if "QUANTITY" in stock_data.columns else None,
        "REVENUE": stock_data[
            "REVENUE IN BEST DENARS"].sum() if "REVENUE IN BEST DENARS" in stock_data.columns else None,
        "AVERAGE PRICE": stock_data["AVERAGE PRICE"].iloc[-1],
        "Timeframes": timeframe_results,
    }

    # Ensure JSON compliance
    response = {
        key: (float(value) if isinstance(value, (np.float64, np.int64)) else value)
        for key, value in response.items()
    }

    return response
