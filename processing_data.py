import os
import pandas as pd
from bson import json_util, ObjectId
import json
from pandas import json_normalize
from pymongo_interface import get_mongo_collection
from dateutil import parser

def process_batch(batch):
    sanitised = json.loads(json_util.dumps(batch))
    normalised = json_normalize(sanitised)
    df = pd.DataFrame(normalised)

    return df

#Defining Airline IDs for checking the match of the Tweet

AIRLINE_IDS = {
    56377143, 106062176, 18332190, 22536055, 124476322, 26223583,
    2182373406, 38676903, 1542862735, 253340062, 218730857, 45621423,
    20626359
}

def mine_conversations():
    collection = get_mongo_collection()
    tweets = list(collection.find({}))
    tweet_by_id = {}

    for tweet in tweets:
        if "id" in tweet:
            tweet_by_id[tweet["id"]] = tweet

    conversations = []
    skipped = 0