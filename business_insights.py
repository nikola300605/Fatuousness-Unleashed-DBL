import pandas as pd
import json
import re
from spicy.stats import chi2_contingency

with open("convesation_data_resolved.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

def classify_topic(text):
    text = text.lower()
    