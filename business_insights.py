import pandas as pd
import json
import re
from scipy.stats import chi2_contingency

with open("conversation_data_resolved.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

sample_df = df.sample(n=1000, random_state=42)

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

print("\nTopic Frequency:")
print(df["topic"].value_counts())

print("\nResolution Rate per Topic")
print(df.groupby("topic")["resolved"].mean())

print("\nAvg Senitment Change per Topic:")
print(df.groupby("topic")["delta_sent"].mean())

print("\nAirline Resolution Rates:")
print(df.groupby("airline")["resolved"].mean())

contingency = pd.crosstab(df["topic"], df["resolved"])
chi2, p, dof, expected = chi2_contingency(contingency)
print("\nChi-Square Test:")
print("Chi2=", chi2)
print("p-value", p)