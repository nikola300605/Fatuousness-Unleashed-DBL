from pymongo import MongoClient
from tqdm import tqdm
import json

client = MongoClient("mongodb://localhost:27017")
db = client["twitter_db"]
collection = db["conversations"]

output = []

# No .count_documents({}) to avoid slowdown
for convo in tqdm(collection.find({}), desc="Processing conversations"):
    thread = convo.get("thread", [])
    if not thread:
        continue

    thread.sort(key=lambda x: x.get("timestamp", ""))
    first_tweet = thread[0]["text"]
    all_texts = [msg["text"] for msg in thread]

    output.append({
        "conversation_id": str(convo["_id"]),
        "airline": convo.get("airline", None),
        "first_tweet": first_tweet,
        "full_convo": all_texts,
        "participants": convo.get("participants", []),
        "length": convo.get("length", len(all_texts)),
        "start_sent": convo.get("computed_metrics", {}).get("start_sent", None),
        "end_sent": convo.get("computed_metrics", {}).get("end_sent", None),
        "delta_sent": convo.get("computed_metrics", {}).get("delta_sent", None),
        "evolution_score": convo.get("computed_metrics", {}).get("evolution_score", None)
    })

with open("../conversation_data_cleaned.json", "w") as f:
    json.dump(output, f, indent=2)