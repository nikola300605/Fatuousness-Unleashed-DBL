from pymongo import MongoClient
from dotenv import load_dotenv
from dateutil import parser
import pandas as pd
import os

load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))
db = client.twitter_db
collection = db.conversations

SENTIMENT_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

def load_and_prepare_data():
    ''' Loads tweets from the MongoDB 'conversations'.

        Retrieves conversation ID, airline name, tweet author, sentiment label, and sentiment score.
        For each tweet, determines whether the author is a 'user' or 'support' (based on screen name vs. airline),
        and calculates a weighted sentiment score by mapping the sentiment label to a numeric value and 
        multiplying it by the model's confidence score.

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

        for tweet in thread:
            screen_name = tweet.get("user", {}).get("screen_name")
            sentiment = tweet.get("sentiment", {})
            label = sentiment.get("label", "").lower()
            score = sentiment.get("score")

            if label in SENTIMENT_MAP and score is not None and screen_name and airline:
                role = "support" if screen_name.lower() == airline.lower() else "user"
                numeric_value = SENTIMENT_MAP[label] * score

                rows.append({
                    "conversation_id": str(doc["_id"]),
                    "airline": airline,
                    "screen_name": screen_name,
                    "role": role,
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "weighted_sentiment": numeric_value
                })

    print(f"Loaded {len(rows)} tweets.")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(df.head())
