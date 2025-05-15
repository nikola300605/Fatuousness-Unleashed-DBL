import os
import pandas as pd
from bson import json_util, ObjectId
import json
from pandas import json_normalize
from pymongo_interface import get_documents_batch
from dateutil import parser
from collections import deque

def process_batch(batch):
    sanitised = json.loads(json_util.dumps(batch))
    normalised = json_normalize(sanitised)
    df = pd.DataFrame(normalised)

    return df

#Defining Airline IDs for checking the match of the Tweet.

AIRLINE_IDS = {
    56377143, 106062176, 18332190, 22536055, 124476322, 26223583,
    2182373406, 38676903, 1542862735, 253340062, 218730857, 45621423,
    20626359
}

#Creating dictionary that have key as id, value as tweet.
def mine_conversations():
    collection = 'tweets_try'
    tweet_by_id = {}
    conversations = []
    skipped = 0

    #First pass: build the ID
    print(f'Indexing tweets by id...')
    for batch in get_documents_batch(collection=collection):
        for tweet in batch:
            if "id" in tweet:
                tweet_by_id[tweet["id"]] = tweet

   
    #Second pass: iterate again and build conversations
    print("Mining conversations...")
    for batch in get_documents_batch(collection=collection):
        for tweet in batch:
            conversation_thread = deque([tweet])
            current_tweet = tweet

        while True:
            parent_id = current_tweet.get('in_reply_to_status_id')
            if not parent_id:
                break

            parent = tweet_by_id.get(parent_id)
            if not parent:
                skipped += 1
                break

            try:
                t_parent = parser.parse(parent['created_at'])
                t_current = parser.parse(current_tweet['created_at'])
                if abs((t_current - t_parent).days) > 7:
                    break
            except Exception as e:
                skipped += 1
                break
            
            user_parent = parent["user"]["id"]
            user_current = current_tweet["user"]["id"]
            if (user_parent in AIRLINE_IDS) == (user_current in AIRLINE_IDS):
                break

            conversation_thread.appendleft(parent)
            current_tweet = parent
            
        if len(conversation_thread) >= 2:
            conversations.append(list(conversation_thread))

    print(f'Found {len(conversations)} valid conversations.')
    print(f'Skipped {skipped} tweets due to  issues.')
    return conversations

