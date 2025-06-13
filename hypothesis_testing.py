import pandas as pd
import json
from scipy.stats import chi2_contingency

with open("conversation_data_resolved.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

crosstab = pd.crosstab(df["topic"], df["resolved"])
print("Contingency Table:\n", crosstab)

chi2, p, dof, expected = chi2_contingency(crosstab)

print(f"\nChi2 = {chi2:.2f}, p-value = {p:.4f}")
if p < 0.05:
    print("Significant difference in resolution rates between topics.")
else:
    print("No significant difference between topics")