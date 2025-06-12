import json
import re

with open("conversation_data_cleaned.json") as f:
    data = json.load(f)

topics = {
    "delay": ["delay", "late", "cancel", "reschedule"],
    "luggage": ["luggage", "baggage", "lost bag", "missing bag"],
    "booking": ["booking", "ticket", "seat", "reservation"],
    "customer_service": ["rude", "support", "agent", "help", "service"]
}

def classify_topic(text):
    text = text.lower()
    for topic, keywords in topics.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape + r"\b", text):
                return topic
    return "other"

for convo in data:
    convo["topic"] = classify_topic(convo["first_tweet"])

with open("conversation_data_with_topics.json", "w") as f:
    json.dump(data, f, indent=2)