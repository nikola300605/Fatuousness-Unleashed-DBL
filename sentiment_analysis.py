from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import torch



tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

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
        tweet['sentiment'] = {'label': result['label'], 'score': round(result['score'], 2)}

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