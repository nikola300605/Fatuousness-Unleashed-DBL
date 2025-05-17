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

# Reconstruct conversation threads between customers and airlines
def mine_conversations():
    collection = 'tweets_try'
    tweet_by_id = {}
    conversations = []
    skipped = 0

    try:
        print(f"[{time.strftime('%X')}] Starting: Indexing tweets by ID...")

        for batch_num, batch in enumerate(get_documents_batch(collection=collection), start=1):
            for tweet in batch:
                if "id" in tweet:
                    tweet_by_id[tweet["id"]] = tweet
            print(f"Indexed batch {batch_num} (total IDs so far: {len(tweet_by_id)})")

        print(f"[{time.strftime('%X')}] Finished indexing. Total tweets indexed: {len(tweet_by_id)}")

    except Exception as e:
        print("Error while indexing tweets:")
        traceback.print_exc()
        return []

    try:
        print(f"[{time.strftime('%X')}] Starting: Mining conversations...")

        for batch_num, batch in enumerate(get_documents_batch(collection=collection), start=1):
            print(f"Processing batch {batch_num}...")

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
                        except Exception as parse_error:
                            skipped += 1
                            break

                        user_parent = parent["user"]["id"]
                        user_current = current_tweet["user"]["id"]
                        if (user_parent in AIRLINE_IDS) == (user_current in AIRLINE_IDS):
                            break

                        conversation_thread.appendleft(parent)
                        current_tweet = parent

                    if len(conversation_thread) >= 2:
                        participants = [t["user"]["screen_name"] for t in conversation_thread]
                        airline_user = next(
                            (t["user"]["screen_name"] for t in conversation_thread if t["user"]["id"] in AIRLINE_IDS),
                            None
                        )
                        conversations.append({
                            "length": len(conversation_thread),
                            "participants": participants,
                            "airline": airline_user,
                            "thread": list(conversation_thread)
                        })

                except Exception as convo_error:
                    skipped += 1
                    print("[!] Skipped a conversation due to error:")
                    traceback.print_exc()

        print(f"[{time.strftime('%X')}] Finished mining conversations.")
        print(f"Found {len(conversations)} valid conversations.")
        print(f"Skipped {skipped} tweets due to issues.")

    except Exception as e:
        print("Error during mining process:")
        traceback.print_exc()

    return conversations
