
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

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(logs, reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={"user_based": True})
    algo.fit(trainset)

    all_items = trainset.all_items()
    raw_items = [trainset.to_raw_iid(i) for i in all_items]
    seen = set(logs[logs["userId"] == userId]["productId"].values)
    testset = [(userId, iid, 0) for iid in raw_items if iid not in seen]

    predictions = algo.test(testset)
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid, est))
    top_n[userId].sort(key=lambda x: x[1], reverse=True)

    rec_ids = [iid for iid, _ in top_n[userId][:3]]
    return {
        "userId": userId,
        "recommendations": products[products["productId"].isin(rec_ids)][["productId", "name"]].to_dict(orient="records")
    }

@app.get("/")
def home():
    return {"message": "AI Engine is live"}
