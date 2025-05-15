import os
from pymongo import MongoClient
from dotenv import load_dotenv


# Initialize MongoDB connection


def get_documents_batch(batch_size=1000, collection='tweets_try'):
    """
    Generator to yield batches of size 1000 documents from MongoDB.
    This function uses a projection to exclude certain fields from the documents. Change the projection as needed.
    """

    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    client = MongoClient(database_url, connect=True, maxPoolSize=50)
    db = client.twitter_db
    collection = db[collection]
    projection = {
        '_id': False,
        'timestamp_ms' : False,
        'id_str': False,
        'display_text_range': False,
        'in_reply_to_status_id_str': False,
        'quote_count': False,
        'reply_count': False,
        'retweet_count': False,
        'favorite_count': False,
        'entities.hashtags': False,
        'entities.urls': False,
        'place': False,
        'user.id_str': False,
        'user.location': False,
        'user.default_profile': False,
        'user.followers_count': False,
        'retweeted_status._id': False,
        'retweeted_status.timestamp_ms' : False,
        'retweeted_status.id_str': False,
        'retweeted_status.display_text_range': False,
        'retweeted_status.in_reply_to_status_id_str': False,
        'retweeted_status.quote_count': False,
        'retweeted_status.reply_count': False,
        'retweeted_status.retweet_count': False,
        'retweeted_status.favorite_count': False,
        'retweeted_status.entities.hashtags': False,
        'retweeted_status.entities.urls': False,
        'retweeted_status.place': False,
        'quoted_status._id': False,
        'quoted_status.timestamp_ms' : False,
        'quoted_status.id_str': False,
        'quoted_status.display_text_range': False,
        'quoted_status.in_reply_to_status_id_str': False,
        'quoted_status.quote_count': False,
        'quoted_status.reply_count': False,
        'quoted_status.retweet_count': False,
        'quoted_status.favorite_count': False,
        'quoted_status.entities.hashtags': False,
        'quoted_status.entities.urls': False,
        'quoted_status.place': False
    }
    cursor = collection.find({}, projection=projection, batch_size=batch_size, allow_disk_use=True)
    batch = []

    for doc in cursor:
        batch.append(doc)
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch

def send_to_mongo(batch, collection='tweets'):
    """Sends a batch of documents to MongoDB."""

    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    client = MongoClient(database_url, connect=True, maxPoolSize=50)
    db = client.twitter_db
    collection = db[collection]
    collection.insert_many(batch)