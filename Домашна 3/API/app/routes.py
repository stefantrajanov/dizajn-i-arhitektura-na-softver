from fastapi import APIRouter

router = APIRouter()

@router.get("/stock-data")
async def get_stock_data():
    # Example stock data (replace with real logic later)
    return {
        "ticker": "TSLA",
        "prices": [100, 102, 105, 108, 110],
        "volumes": [5000, 6000, 5500, 6200, 5800]
    }

@router.post("/analyze")
async def analyze(data: dict):
    # Example analysis logic
    prices = data.get("prices", [])
    average_price = sum(prices) / len(prices) if prices else 0
    return {"average_price": average_price}