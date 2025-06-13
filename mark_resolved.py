import json

with open("conversation_data_with_topics.json") as f:
    data = json.load(f)


for convo in data:
    start = convo.get("start_sent")
    delta = convo.get("delta_sent")

    convo["resolved"] = bool(start is not None and delta is not None and start < 0.2 and delta > 0.1)


with open("conversation_data_resolved.json","w") as f:
    json.dump(data, f, indent=2)