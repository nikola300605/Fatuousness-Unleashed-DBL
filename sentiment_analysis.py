from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from bson import ObjectId
from pymongo_interface import get_documents_batch
from processing_data import process_batch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import evaluate
import matplotlib.pyplot as plt


tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=None)

def get_sentiment(batch):
    all_texts = []
    tweet_refs = []

    for conv in batch:
        for tweet in conv['thread']:
            text = tweet.get('extended_tweet', {}).get('extended_text') or tweet.get('text')
            if not text:
                continue
            all_texts.append(text)
            tweet_refs.append(tweet)

    # Single batched sentiment call
    results = nlp(all_texts, batch_size=32)  # You can tune batch_size

    # Assign results back to tweets
    for tweet, result in zip(tweet_refs, results):
        tweet['sentiment'] = {'label': result[0]['label'], 'score': round(result[0]['score'], 2)} # New model gives a list of all labels and their scores, so I just pick the top one

    return batch
            
""" sentence = "I love this product! It's amazing and works perfectly."
result = nlp(sentence)

# Print the result
print(result) """


""" df = process_batch(new_batch)
scores = df['sentiment.label'].value_counts().plot(kind='bar')
plt.show() """

# Total number of conversations
# Average conversation length (tweet number)
# Mean sentiment score
# Sentiment distribution
# Average response time
# Average response time per airline
# Trends over time  

def test_model_with_running(data_csv):
    pass