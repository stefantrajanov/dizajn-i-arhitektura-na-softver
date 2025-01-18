from collections import Counter
from fastapi import APIRouter
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()
HF_API_KEY = os.getenv("HF_API_KEY")

from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd


def getLatestStatistics(symbol):
    all_company_data = []
    file_path = 'output_file_update.csv'

    url = f'https://www.mse.mk/mk/stats/symbolhistory/{symbol}'
    response = requests.get(url)
    content = BeautifulSoup(response.text, 'html.parser')
    table_of_data = content.select("#resultsTable tbody tr")

    for row in table_of_data:
        data = row.text.split('\n')
        data.pop()
        data.remove('')
        if data.__contains__(''):
            continue
        data.insert(0, symbol)
        all_company_data.append(data)

    dataframe = pd.DataFrame(all_company_data, columns=['COMPANY', 'DATE', 'PRICE OF LAST TRANSACTION', 'MAX', 'MIN', 'AVERAGE PRICE', '% PERCENT', 'QUANTITY', 'REVENUE IN BEST DENARS', 'TOTAL REVENUE IN DENARS'])

    def string_to_float(column):
        if column == 'DATE':
            return dataframe[column]
        if column == 'TOTAL REVENUE IN DENARS':
            dataframe[column] = dataframe[column].str.replace('.', '').astype(float)
            return dataframe[column]
        if column == 'COMPANY':
            return dataframe[column]
        try:
            # Replace commas with dots, then remove extra dots and convert to float
            dataframe[column] = dataframe[column].str.replace(',', '.').str.replace('.', '', 1)
            dataframe[column] = dataframe[column].astype(float)
        except Exception as e:
            print(f"Error processing column {column}: {e}")
        return dataframe[column]

    # Apply the transformation to each column
    for col in dataframe.columns:
        dataframe[col] = string_to_float(col)

    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])

    current_data = pd.read_csv('app/data-formatted.csv')
    current_data['DATE'] = pd.to_datetime(current_data['DATE'])

    merged_df = pd.concat([current_data, dataframe], ignore_index=True)
    updated_df = merged_df.drop_duplicates(keep='first')

    updated_df.to_csv('app/data-formatted.csv', index=False)

    return updated_df


def getBerzaNews(symbol):
    url = f'https://www.mse.mk/en/symbol/{symbol}'
    response = requests.get(url)
    content = BeautifulSoup(response.text, 'html.parser')

    # finding links to news
    aElements = content.find_all('a', href=True)
    newsLinks = [link['href'] for link in aElements if link['href'].startswith('/en/news')]

    news = []
    for link in newsLinks:
        response = requests.get(f'https://www.mse.mk{link}')
        content = BeautifulSoup(response.text, 'html.parser')
        try:
            # print(content.find(id='content').text)
            # print('-----------------------------------')
            news.append(content.find(id='content').text)
        except Exception as e:
            continue

    return news


# Load a multilingual model
def analyzeSentiment(symbol):
    API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(text):
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return None
        return response.json()

    def getMax(result):
        # Ensure the response has the expected structure
        if not result or not isinstance(result, list):
            return "neutral"  # Default to neutral if the response is invalid

        result = result[0] if isinstance(result[0], list) else result
        max_label = max(result, key=lambda x: x['score'])['label']
        return max_label

    # Fetch news articles
    texts = getBerzaNews(symbol)
    if not texts:
        return 'No news for ' + symbol

    # Collect sentiment labels
    sentiment_labels = []
    for text in texts:
        try:
            result = query(text)  # Query the Hugging Face API
            if result:
                sentiment_label = getMax(result)  # Get the label
                sentiment_labels.append(sentiment_label)
        except Exception as e:
            print(f"Error processing text: {e}")
            continue

    if not sentiment_labels:
        return 'No news for ' + symbol

    # Count occurrences of each sentiment
    sentiment_counts = Counter(sentiment_labels)

    # Find the sentiment with the most occurrences
    most_common = sentiment_counts.most_common()

    if len(most_common) == 0:
        return "neutral"  # Default to neutral if no sentiments found

    # Check for ties
    max_count = most_common[0][1]
    top_sentiments = [sentiment for sentiment, count in most_common if count == max_count]

    if len(top_sentiments) > 1:
        return "neutral"  # Return neutral in case of ties
    return top_sentiments[0]

def predict_future_price(input_data):
    input_data = input_data.drop(columns=['DATE'])
    data_to_dictionary = input_data.to_dict(orient='list')
    
    url = 'https://stefan155-das-lstm-model-api.hf.space/predict/'
    
    payload = {
        "input_data": data_to_dictionary,
    }
    
    response = requests.post(url, json=payload)
    return response.json()['prediction']


# Function to resample data for timeframes
def resample_data(data, timeframe):
    data["DATE"] = pd.to_datetime(data["DATE"])  # Ensure DATE is in datetime format

    data = data.drop_duplicates(subset="DATE", keep="first")  # Drop duplicate dates

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

    print(f"Resampled data for {timeframe} timeframe:")
    print(resampled_data)

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
    latest_data = getLatestStatistics(ticker)
    stock_data = latest_data[latest_data["COMPANY"] == ticker]
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

            price_prediction = predict_future_price(indicators_data)
            market_news_evaluation = analyzeSentiment(ticker)
            # market_news_evaluation = 'neutral'


            timeframe_results[timeframe] = {
                "Price Prediction": price_prediction,
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
