from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

tokenizer = DistilBertTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
model = DistilBertForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

dataset = load_dataset(
    "carblacac/twitter-sentiment-analysis",
    trust_remote_code=True,
    cache_dir="./hf_cache",
)
test_data = dataset["test"]

texts = [item['text'] for item in test_data]
true_labels = [item['feeling'] for item in test_data]

# Batch prediction for speed
results = nlp(texts, batch_size=32)
predictions = [result['label'] for result in results]

label_map = {
    'NEGATIVE': 0,
    'POSITIVE': 1
}
pred_labels = [label_map[pred] for pred in predictions]

# Calculate accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {accuracy:.4f}")