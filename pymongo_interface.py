import os
from pymongo import MongoClient
from dotenv import load_dotenv


# Initialize MongoDB connection
load_dotenv()
database_url = os.getenv("DATABASE_URL")
client = MongoClient(database_url, connect=True, maxPoolSize=50)
db = client.twitter_db

def get_documents_batch(batch_size=1000, collection='tweets_try', query = {}, projection = {
        '_id': False,
        'created_at': True,
        'id': True,
        'user': True,
        'in_reply_to_status_id': True,
        'entities': True,
        'lang': True,
        'extended_tweet:': True,
        'text': True
    }):
    """
    Generator to yield batches of size 1000 documents from MongoDB.
    This function uses a projection to exclude certain fields from the documents. Change the projection as needed.
    """


    collection = db[collection]
    cursor = collection.find(query, projection=projection, batch_size=batch_size, allow_disk_use=True)
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

def update_one_document(collection_name: str, data: dict, method: str):

    """
    Updates one document in MongoDB.
    
    :param collection_name: The name of the collection you want to update.
    :param data: A dict with fields to update. Must include '_id' (ObjectId).
    :param method: Update type. Currently only supports 'set'.
    
    :return: Number of documents modified.
    """

    collection = db[collection_name]
    if method == 'set':
        _id = data['_id']
        data.pop('_id')
        result = collection.update_one(
            {'_id': _id},
            {
                '$set': data
            }
        )
    
    return result.modified_count