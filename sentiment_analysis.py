from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
from pymongo_interface import get_documents_batch

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

for batch in get_documents_batch(batch_size=1000, collection='tweets_try'):
    texts = [tweet.get("text", "") for tweet in batch if tweet.get("text")]

    if not texts:
        print("No text found in the batch.")
        continue

    results = nlp(texts)

   
    for tweet, result in zip(batch, results):
        tweet['sentiment'] = {'label': result['label'], 'score': round(result['score'], 2)}
        print(f"Tweet: {tweet.get('text', '')[:20]}...\n→ Sentiment: {result}, Score: {result['score']:.2f}\n")
        print(tweet)
        print("\n \n")
        


    break  


