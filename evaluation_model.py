from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os

os.environ['HF_DATASETS_CACHE'] = './hf_cache'

# Step 1: Load the Pretrained Model and Tokenizer
model_name = "tabularisai/multilingual-sentiment-analysis"

tokenizer = DistilBertTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = DistilBertForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

dataset = load_dataset(
    "carblacac/twitter-sentiment-analysis",
    trust_remote_code=True,
    cache_dir="./hf_cache",
)
test_data = dataset["test"]

# Step 3: Predict Labels Using the Model
pred_labels = []
true_labels = []

label_map = {
    "NEGATIVE": 0,
    "POSITIVE": 1
}

print("Running predictions...")

for item in tqdm(test_data):
    text = item["text"]
    true_label = item["feeling"]

    prediction = nlp(text)[0]['label']
    pred_label = label_map[prediction]

    pred_labels.append(pred_label)
    true_labels.append(true_label)

# Step 4: Compute and Print Accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Accuracy: {accuracy:.2f}")

# Optional: Detailed Performance Breakdown
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"]))