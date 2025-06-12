from pymongo import MongoClient
import json

output = []

for convo in collection.find({}):
    thread = convo.get("thread", [])
    if not thread:
        continue

    thread.sort(key=lambda x: x.get("timestamp", ""))
    first_tweet = thread[0]["text"]
    all_texts = [msg["text"] for msg in thread]

    output.append({
        
    })