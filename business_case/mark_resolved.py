import json
from tqdm import tqdm  # Add tqdm for progress bar

with open("conversation_data_hybrid_conservative.json") as f:
    data = json.load(f)

# Use tqdm to wrap the loop
for convo in tqdm(data, desc="Evaluating resolution status"):
    start = convo.get("start_sent")
    delta = convo.get("delta_sent")
    evolution_score = convo.get("evolution_score")

    convo["resolved"] = bool(
        evolution_score is not None and
        evolution_score > 0.2
    )

with open("conversation_data_resolved.json", "w") as f:
    json.dump(data, f, indent=2)
