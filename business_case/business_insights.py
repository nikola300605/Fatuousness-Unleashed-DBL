import pandas as pd
import json
import re
from scipy.stats import chi2_contingency
import numpy as np

with open("conversation_data_resolved.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

sample_df = df.sample(n=1000, random_state=42)
crosstab = pd.crosstab(df["topic"], df["resolved"])
crosstab_sample = pd.crosstab(sample_df["topic"], sample_df["resolved"])

print("\nTopic Frequency:")
print(df["topic"].value_counts())

print("\nResolution Rate per Topic")
print(df.groupby("topic")["resolved"].mean())

print("\nAvg Senitment Change per Topic:")
print(df.groupby("topic")["delta_sent"].mean())

print("\nAirline Resolution Rates:")
print(df.groupby("airline")["resolved"].mean())

contingency = crosstab_sample
chi2, p, dof, expected = chi2_contingency(contingency)
print("\nChi-Square Test:")
print("Chi2=", chi2)
print(f"p-value = {p:.4f}")

n = contingency.to_numpy().sum()
phi2 = chi2 / n
r, k = contingency.shape
cramer_v = np.sqrt(phi2 / min(k - 1, r - 1))
print(f"Cramer's V = {cramer_v:.4f}")

#There is a weak association between the topic of the complaint and whether it was resolved (by Cramer's V).
