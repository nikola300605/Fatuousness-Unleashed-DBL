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
        "converstion_id": str(convo["_id"]),
        "airline": convo.get("airline", None),
        "first_tweet": first_tweet,
        "full_convo": all_texts,
        "participants": convo.get("participants",[]),
        "length": convo.get("length", len(all_texts)),
        "start_sent": convo.get("computed_metrics", {}).get("start_sent",None),
        "end_sent": convo.get("computed_metrics", {}).get("end_sent",None),
        "delta_sent": convo.get("computed_metrics", {}).get("delta_sent",None)
    })

with open("conversation_data_cleaned.json","w") as f:
    json.dump(output, f, indent=2)
