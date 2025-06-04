from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import torch



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

tweet1 = "There is, I believe laws on how animals are transported. I am defeated, Your planes are filthy, your staff appears incapable of doing their jobs, and your customer service is non-existent."
tweet2= "@British_Airways @Juggler90 Looks ok to me!"
tweet3 = "@British_Airways I’ve just checked in online, it’s a American Airlines flight through BA. I’m travelling with my mum and she’s 80. I’m not sure if she’ll be able to walk the distance to the gate. Can you help us? Thanks."
result = nlp([tweet1, tweet2, tweet3])
results = []
for i in range(len(result)):
    final = {
                'predicted_label': result[i][0]['label'],  # Top prediction
                'scores': {
                    'negative': round(next(r['score'] for r in result[i] if r['label'] == 'negative'), 2),
                    'neutral': round(next(r['score'] for r in result[i] if r['label'] == 'neutral'), 2),
                    'positive': round(next(r['score'] for r in result[i] if r['label'] == 'positive'), 2)
                }
            }
    results.append(final)

print("".join(f"result: {res}\n" for res in results))