from pymongo import MongoClient
from dotenv import load_dotenv
from dateutil import parser
import pandas as pd
import os
from time import time
from pymongo_interface import save_conversations_to_mongo
from datetime import datetime
import numpy as np
from bson import ObjectId

load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))
db = client.twitter_db
collection = db.conversations

SENTIMENT_MAP = {
    "positive": 1,
    "neutral": 0.1,
    "negative": -1
}
def get_sentiment_value(label, confidence):
    if label == "positive":
        return 1  # 0-1 range
    elif label == "negative":
        return -1  # -1-0 range
    elif label == "neutral":
        # Map confidence to (-0.3, 0.3) range
        return (confidence - 0.5) * 0.6

def load_and_prepare_data():
    ''' Loads tweets from the MongoDB 'conversations'.

        Retrieves conversation ID, airline name, tweet author, sentiment label, and sentiment score.
        For each tweet, determines whether the author is a 'user' or 'support' (based on screen name vs. airline),
        and calculates a sentiment score by mapping the sentiment label to a numeric value.

        Returns: pd.DataFrame: A DataFrame where each row represents a tweet with metadata and weighted sentiment.
    '''

    print("Loading conversation data from MongoDB...")
    cursor = collection.find({}, {
        "_id": 1,
        "airline": 1,
        "thread.user.screen_name": 1,
        "thread.sentiment.label": 1,
        "thread.sentiment.score": 1
    })

    rows = []

    for doc in cursor:
        airline = doc.get("airline")
        thread = doc.get("thread", [])
        if not isinstance(thread, list):
            continue

        for i, tweet in enumerate(thread):
            screen_name = tweet.get("user", {}).get("screen_name")
            sentiment = tweet.get("sentiment", {})
            label = sentiment.get("label", "").lower()
            score = sentiment.get("score")

            if label in SENTIMENT_MAP and score is not None and screen_name and airline:
                role = "support" if screen_name.lower() == airline.lower() else "user"
                numeric_value = get_sentiment_value(label, score)

                rows.append({
                    "conversation_id": str(doc["_id"]),
                    "airline": airline,
                    "screen_name": screen_name,
                    "role": role,
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "sentiment_numerical": numeric_value,
                    "tweet_index": i
                })

    print(f"Loaded {len(rows)} tweets.")
    return pd.DataFrame(rows)

def exponential_decay_weight(min_weight, decay_rate, tweet_index, max_index):
    '''Calculates a weight based on exponential decay for tweet position'''

    position = tweet_index / max_index  # Normalized position from 0 (start) to 1 (end)
    return  max(min_weight, decay_rate ** (1 - position)) 

def calculate_role_weights(role, sentiment_label):
    '''Calculates role-based weights dependent on the role in conversation and sentiment of the tweet.'''

    ROLE_LABEL_WEIGHT = {
    ('user', 'negative'): 1.2,
    ('user', 'positive'): 1.0,
    ('user', 'neutral'): 0.4,
    ('support', 'negative'): 0.6,
    ('support', 'positive'): 0.3,
    ('support', 'neutral'): 0.2
    }

    role_label_key = (role, sentiment_label)
    role_weight = ROLE_LABEL_WEIGHT.get(role_label_key, 0.7)

    return role_weight


def compute_conversation_score(group: pd.DataFrame):
    '''Analyzes sentiment evolution with proper positional weighting'''
    airline = group['airline'].iloc[0]
    group = group.sort_values(by='tweet_index')
    
    total_weighted_sentiment = 0
    max_possible_weight = 0
    
    for _, row in group.iterrows():
        # Get base components
        base_sentiment = row['sentiment_numerical']
        confidence = row['sentiment_score']
        role_weight = calculate_role_weights(row['role'], row['sentiment_label'])
        
        # Calculate positional weight (using tweet_index instead of i)
        positional_idx = row['tweet_index']  # Using the stored index
        positional_weight = exponential_decay_weight(0.2, 0.5, positional_idx, len(group) - 1)
        
        # Calculate contribution
        weight = confidence * role_weight * positional_weight
        sentiment_contribution = base_sentiment * weight
        
        max_possible_weight += weight
        total_weighted_sentiment += sentiment_contribution
    
    # Normalize to [-1, 1] range
    conv_sentiment = total_weighted_sentiment / max_possible_weight if max_possible_weight > 0 else 0
    return conv_sentiment

def compute_trend_score(group: pd.DataFrame):
    sorted_group = group.sort_values(by='tweet_index')
    n = len(sorted_group)

    if n > 1:
        sorted_group['position'] = sorted_group['tweet_index'] / (n - 1)
    else:
        sorted_group['position'] = 0.5
    
    sorted_group['positional_weight'] = 0.5 ** (n-1-sorted_group['tweet_index'])
    role_weights = {'user': 1.0, 'support': 0.7}  
    sorted_group['role_weight'] = sorted_group['role'].map(role_weights)

    sorted_group['weight'] = (
    sorted_group['sentiment_numerical'] * 
    sorted_group['role_weight'] * 
    sorted_group['positional_weight'])

    total_weight = sorted_group['weight'].sum()
    if total_weight > 0:
        y_w = (sorted_group['sentiment_numerical'] * sorted_group['weight']).sum() / total_weight
        x_w = (sorted_group['position'] * sorted_group['weight']).sum() / total_weight
    else:
        # Fallback to simple unweighted average
        y_w = sorted_group['sentiment_numerical'].mean()
        x_w = sorted_group['position'].mean()

    numerator = 0
    denominator = 0

    for _, row in sorted_group.iterrows():
        x_diff = row['position'] - x_w
        y_diff = row['sentiment_numerical'] - y_w
        weighted_diff = row['weight'] * x_diff * y_diff
        numerator += weighted_diff

        x_sq_diff = row['weight']*(x_diff ** 2)
        denominator += x_sq_diff
    
    trend_slope = numerator / denominator if denominator != 0 else 0
    return trend_slope
def compute_delta_score(group: pd.DataFrame):
    sorted_group = group.sort_values(by='tweet_index')

    max_index = sorted_group['tweet_index'].max()
    if max_index == 0:
        return 1
    
    start_sent_unweighted = sorted_group.iloc[0]['sentiment_numerical']
    start_score = sorted_group.iloc[0]['sentiment_score']

    end_sent_unweighted = sorted_group[sorted_group['role'] == 'user'].iloc[-1]['sentiment_numerical']
    end_score = sorted_group[sorted_group['role'] == 'user'].iloc[-1]['sentiment_score']

    start_sent = start_sent_unweighted * start_score
    end_sent = end_sent_unweighted * end_score
    delta_sent = ((end_sent) - (start_sent)) / 2

    return pd.Series({
        'start_sent': start_sent,
        'end_sent':   end_sent,
        'delta_sent': delta_sent
    })

def categorize_behavior(row):
    delta = row['delta_sent']
    if delta > 0.3:
        return 'Improving'
    elif delta < -0.3:
        return 'Worsening'
    else:
        return 'Stable'


def store_results_to_mongodb(df_convo: pd.DataFrame, collection):
    '''Updates each conversation document with precomputed metrics'''
    print("Storing results to MongoDB...")

    updates_count = 0

    for _, row in df_convo.iterrows():
        try:
            conversation_id = ObjectId(row['conversation_id'])  # Convert back to ObjectId -> wasn't storing because id is a string
        except Exception as e:
            print(f"Skipping invalid ID: {row['conversation_id']}")
            continue
        update_data = {
            'conversation_score': row['conversation_score'],
            'delta_sent': row['delta_sent'],
            'evolution_score': row['evolution_score'],
            'evolution_category': row['evolution_category'],
            'conversation_trajectory': row['conversation_trajectory'],
            # Optional: 'computed_at': datetime.now()
        }

        result = collection.update_one(
            {'_id': conversation_id},
            {'$set': {'computed_metrics': update_data}}
        )

        if result.modified_count > 0:
            updates_count += 1

    print(f"Updated {updates_count} conversation documents with computed metrics.")
    return True



if __name__ == "__main__":
    start_time = time()

    # Load tweet-level data
    df = load_and_prepare_data()

    # Compute conversation-level scores
    scores_conv_sc = df.groupby('conversation_id', group_keys=False)[df.columns]\
                   .apply(compute_conversation_score)\
                   .reset_index(name='conversation_score')

    scores_delta = df.groupby('conversation_id', group_keys=False)[df.columns]\
                 .apply(compute_delta_score)\
                 .reset_index()
    #changed as to not give depriciation warning

    # Merge convo-level scores into a single conversation-level DataFrame
    df_convo = scores_conv_sc.merge(scores_delta, on='conversation_id')
    df_convo['evolution_score'] = df_convo['conversation_score'] * 0.7 + df_convo['delta_sent'] * 0.3

    bins = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    df_convo['evolution_category'] = pd.cut(df_convo['evolution_score'], bins=bins, labels=labels)

    df_convo['conversation_trajectory'] = df_convo.apply(categorize_behavior, axis=1)

    # Store conversation-level metrics into MongoDB
    store_results_to_mongodb(df_convo, collection)

    end_time = time()
    print(f"Data processing completed in {end_time - start_time:.2f} seconds.")
    print(f"Results stored in MongoDB. Processed {df_convo.shape[0]} conversations.")
    print(df_convo.head(10))