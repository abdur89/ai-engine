
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

@app.post("/logEvent")
def log_event(event: LogEvent):
    df = pd.read_csv("logs.csv")
    df = df._append({"userId": event.userId, "productId": event.productId, "rating": 1}, ignore_index=True)
    df.to_csv("logs.csv", index=False)

    product_df = pd.read_csv("products.csv")
    if event.productId not in product_df["productId"].values:
        product_df = product_df._append({
            "productId": event.productId,
            "name": "Unknown",
            "category": "Unknown"
        }, ignore_index=True)
        product_df.to_csv("products.csv", index=False)

    return {"status": "logged"}

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
