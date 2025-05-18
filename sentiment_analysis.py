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

def get_sentiment(batch):
    for conv in batch:
        thread_texts = []
        twitter_refs = []
        for tweet in conv['thread']:

            text = tweet.get('text') if 'text' in tweet else None
            if not text:
                print("No text found in the tweet.")
                continue

            thread_texts.append(text)
            twitter_refs.append(tweet)

        results = nlp(thread_texts)
        for tweet,result in zip(twitter_refs, results):
            tweet['sentiment'] = {'label': result['label'], 'score': round(result['score'], 2)}
    
    return batch
            

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