from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
from pymongo_interface import get_documents_batch
from processing_data import process_batch

import seaborn as sns
import matplotlib.pyplot as plt


tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

for batch in get_documents_batch(batch_size=1000, collection='tweets_try'):
    texts = [tweet.get("text", "") for tweet in batch if tweet.get("text")]

    if not texts:
        print("No text found in the batch.")
        continue

    results = nlp(texts)

    new_batch = []
    for tweet, result in zip(batch, results):
        tweet['sentiment'] = {'label': result['label'], 'score': round(result['score'], 2)}
        new_batch.append(tweet)

    break  

df = process_batch(new_batch)
scores = df['sentiment.label'].value_counts().plot(kind='bar')
plt.show()

# Total number of conversations
# Average conversation length (tweet number)
# Mean sentiment score
# Sentiment distribution
# Average response time
# Average response time per airline
# Trends over time  