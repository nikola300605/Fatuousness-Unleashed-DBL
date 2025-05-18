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
        'created_at': True,
        'id': True,
        'text': True,
        'user': True,
        'in_reply_to_status_id': True,
        'entities': True,   
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

def save_conversations_to_mongo(conversations, collection_name='conversation_threads'):
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    client = MongoClient(database_url, connect=True, maxPoolSize=50)
    db = client.twitter_db
    collection = db[collection_name]

    if conversations:
        collection.insert_many(conversations)
        print(f"Saved {len(conversations)} conversation threads to MongoDB collection '{collection_name}'")
    else:
        print("No conversations to save.")
