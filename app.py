
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from collections import defaultdict

app = FastAPI()

class LogEvent(BaseModel):
    userId: str
    event: str
    productId: str
    timestamp: str
    b2bUnit: str

@app.post("/logEvent")
def log_event(event: LogEvent):
    import pandas as pd

    # Load existing logs or create empty DataFrame if missing
    try:
        df = pd.read_csv("logs.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["userId", "productId", "rating", "b2bUnit"])

    # Append new entry with a fixed rating (implicit binary rating = 1)
    df = df._append({
        "userId": event.userId,
        "productId": event.productId,
        "rating": 1,
        "b2bUnit": event.b2bUnit
    }, ignore_index=True)

    # Save logs
    df.to_csv("logs.csv", index=False)

    # Update products.csv if productId not already there
    products = pd.read_csv("products.csv")
    if str(event.productId) not in products["productId"].astype(str).values:
        products = products._append({
            "productId": event.productId,
            "name": "Unknown",
            "category": "Unknown"
        }, ignore_index=True)
        products.to_csv("products.csv", index=False)

    return {"status": "event logged"}

@app.get("/recommendations")
def recommend(userId: str):
    logs = pd.read_csv("logs.csv")
    products = pd.read_csv("products.csv")

    # Get current user's b2bUnit
    user_row = logs[logs["userId"] == userId]
    if user_row.empty:
        return {"error": "User not found"}

    user_unit = user_row["b2bUnit"].iloc[0]

    # Filter logs to that unit
    unit_logs = logs[logs["b2bUnit"] == user_unit]

    # Use only required columns
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(unit_logs[["userId", "productId", "rating"]], reader)
    trainset = data.build_full_trainset()

    # Fit model
    algo = KNNBasic(sim_options={"user_based": True})
    algo.fit(trainset)

    # Build testset of unseen items
    seen_items = set(logs[logs["userId"] == userId]["productId"].values)
    all_items = trainset.all_items()
    raw_items = [trainset.to_raw_iid(i) for i in all_items if trainset.to_raw_iid(i) not in seen_items]
    testset = [(userId, iid, 0) for iid in raw_items]

    # Predict and sort
    predictions = algo.test(testset)
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)
    recommended_ids = [pred.iid for pred in top_n[:5]]

    # Join with product data
    recommendations = products[products["productId"].isin([int(i) for i in recommended_ids])]
    return recommendations[["productId", "name", "category"]].to_dict(orient="records")

@app.get("/")
def home():
    return {"message": "AI Engine is live"}
