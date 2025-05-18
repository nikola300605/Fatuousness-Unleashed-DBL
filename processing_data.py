import os
import pandas as pd
from bson import json_util
import json
from pandas import json_normalize
from pymongo_interface import get_documents_batch
from dateutil import parser
from collections import deque
import traceback
import time

# Flatten a batch of tweets into a DataFrame
def process_batch(batch):
    sanitised = json.loads(json_util.dumps(batch))
    normalised = json_normalize(sanitised)
    df = pd.DataFrame(normalised)
    return df

# Set of known airline user IDs
AIRLINE_IDS = {
    56377143, 106062176, 18332190, 22536055, 124476322, 26223583,
    2182373406, 38676903, 1542862735, 253340062, 218730857, 45621423,
    20626359
}


def create_id_map():
    # First pass: build the ID map
    print(f'Indexing tweets by id...')
    # First pass: Collect IDs of tweets that are replied to
    reply_ids = set()
    print("Identifying potential parents...")
    for batch in get_documents_batch(collection='tweets_try'):
        for tweet in batch:
            if 'deleted' in tweet:
                continue
            if 'in_reply_to_status_id' in tweet and tweet['in_reply_to_status_id'] is not None:  # Fixed condition
                reply_ids.add(tweet['in_reply_to_status_id'])
        print(f"Batch {len(batch)} done")

    tweet_by_id = {}
    print("Caching parents...")
    for batch in get_documents_batch(collection='tweets_try'):
        for tweet in batch:
            if 'deleted' in tweet:
                continue
            if 'id' in tweet and tweet['id'] in reply_ids:  # Only store if this tweet is a parent
                tweet_by_id[tweet['id']] = tweet
        # REMOVED the break statement here - this was causing the main issue
    
    print(f"Identified {len(reply_ids)} potential parent tweets, cached {len(tweet_by_id)}")
    return tweet_by_id

# Reconstruct conversation threads between customers and airlines
def mine_conversations():
    collection = 'tweets_try'
    skipped = 0
    conversation_batch = []
    
    tweet_by_id = create_id_map()

    # Second pass: iterate again and build conversations
    print("Mining conversations...")
    for batch in get_documents_batch(collection=collection):
        for tweet in batch:
            try:
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
                    except Exception:
                        skipped += 1
                        break

                    user_parent = parent["user"]["id"]
                    user_current = current_tweet["user"]["id"]
                    if (user_parent in AIRLINE_IDS) == (user_current in AIRLINE_IDS):
                        break

                    conversation_thread.appendleft(parent)
                    current_tweet = parent
                if len(conversation_thread) >= 2:

                    if (tweet["user"]["id"] not in AIRLINE_IDS and
                        "entities" in tweet and
                        any(mention["id"] in AIRLINE_IDS for mention in tweet["entities"]["user_mentions"])
                        ):

                        participants = [t["user"]["screen_name"] for t in conversation_thread]
                        airline_user = next(
                            (t["user"]["screen_name"] for t in conversation_thread if t["user"]["id"] in AIRLINE_IDS),
                            None
                        )
                        conversation = {
                            "length": len(conversation_thread),
                            "participants": participants,
                            "airline": airline_user,
                            "thread": list(conversation_thread)
                        }

                        conversation_batch.append(conversation)
                        if len(conversation_batch) >= 1000:
                            yield conversation_batch
                            conversation_batch.clear()
            except Exception as e:
                skipped += 1
                print(f"Error processing tweet {tweet['id']}: {e}")
                traceback.print_exc()

    if conversation_batch:
        yield conversation_batch

    print(f'Skipped {skipped} tweets due to issues.')
