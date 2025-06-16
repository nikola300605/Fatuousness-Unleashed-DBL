from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import torch
from bson import ObjectId
from pymongo_interface import get_documents_batch



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

def test_model_without_running(data_csv):
    df = pd.read_csv(data_csv, names=["conversation_id","tweet_id", "true_tweet_sentiment", "true_evolution_category"])
    tweet_ids = df['tweet_id'].tolist()
    conversation_ids = df['conversation_id'].tolist()

    object_ids = [ObjectId(cid) for cid in conversation_ids]
    query = {"_id": {"$in": object_ids}}
    projection = {
        '_id': True,
        'conversation_id': True,
        'thread': True,
        'computed_metrics.evolution_category': True
    }
    cursor = get_documents_batch(query=query, projection = projection, collection='conversations')
    data= []
    for batch in cursor:
        data.extend(batch)

