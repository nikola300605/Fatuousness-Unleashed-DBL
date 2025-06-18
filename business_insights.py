import pandas as pd
import json
import re
from spicy.stats import chi2_contingency

with open("convesation_data_resolved.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

def classify_topic(text):
    text = text.lower()
    if re.search(r"delay|late|cancel|reschedule",text):
        return "delay"
    elif re.search(r"luggage|bag|lost|baggage",text):
        return "luggage"
    elif re.search(r"booking|ticket|seat|check[- ]in",text):
        return "booking"
    elif re.search(r"rude|staff|customer service|support|agent",text):
        return "customer_service"
    else:
        return "other"
    
df["topic"] = df["first_tweet"].apply(classify_topic)

df["resolved"] = df["delta_sent"] > 0