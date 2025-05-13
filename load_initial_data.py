from pymongo import MongoClient
import ijson
import os
from decimal import Decimal
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

load_dotenv()
database_url = os.getenv("DATABASE_URL")

connection_string = database_url
json_dir = './cleaned_tweets_json/'

def convert_decimals(obj):
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def clean_data(data: dict, depth=0, max_depth=3) -> dict:
    if not isinstance(data, dict):
        return None

    # Stop recursion if max depth is reached
    if depth > max_depth:
        return data

    # Define columns to purge
    purge_user_columns = [
        "url", "description", "translator_type", "friends_count", "listed_count",
        "favourites_count", "statuses_count", "created_at", "utc_offset", "time_zone",
        "lang", "contributors_enabled", "is_translator", "profile_background_color",
        "profile_background_image_url", "profile_background_image_url_https", "profile_background_tile",
        "profile_link_color", "profile_sidebar_border_color", "profile_sidebar_fill_color",
        "profile_text_color", "profile_use_background_image", "profile_image_url",
        "profile_image_url_https", "profile_banner_url", "default_profile_image",
        "following", "follow_request_sent", "notifications", "protected", "is_translation_enabled"
    ]

    purge_place_columns = [
        "bounding_box", "attributes", "country", "place_type", "name", "url"
    ]

    purge_entities_columns = ['symbols', 'polls']

    purge_other_columns = [
        "geo", "coordinates", "favorited", "retweeted", "possibly_sensitive",
        "filter_level", "contributors", "truncated", "extended_entities",
        "quoted_status_permalink"
    ]

    # Recursively clean nested tweet fields
    for key in ['retweeted_status', 'quoted_status']:
        if isinstance(data.get(key), dict):
            data[key] = clean_data(data[key], depth=depth + 1, max_depth=max_depth)

    # Remove top-level fields
    for field in purge_other_columns:
        data.pop(field, None)

    # Clean nested 'user'
    if isinstance(data.get('user'), dict):
        for field in purge_user_columns:
            data['user'].pop(field, None)

    # Clean nested 'place'
    if isinstance(data.get('place'), dict):
        for field in purge_place_columns:
            data['place'].pop(field, None)

    # Clean nested 'entities'
    if isinstance(data.get('entities'), dict):
        for field in purge_entities_columns:
            data['entities'].pop(field, None)

    return data  

def clean_batch(batch:list) -> list:
    return[clean_data(doc) for doc in batch if clean_data(doc) is not None]

def conv_dec_batch(batch:list) -> list:
    return [convert_decimals(doc) for doc in batch if convert_decimals(doc) is not None]

def load_files(files):
    client = MongoClient(connection_string)
    db = client.twitter_db
    collection = db.tweets_try
    BATCH_SIZE = 2500
    batch = []

    for file_path in files:
        try:    
            with open(file=file_path, mode='r',encoding='utf-8') as file:
                objects = ijson.items(file, '', multiple_values=True)
                for obj in objects:
                        batch.append(obj)

                        if len(batch) == BATCH_SIZE:
                            cleaned_batch = conv_dec_batch(clean_batch(batch))
                            collection.insert_many(cleaned_batch)
                            batch.clear()
                        
                        print(f"Inserted batch of {BATCH_SIZE} tweets from {file_path}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")        
        if batch:
            cleaned_batch = conv_dec_batch(clean_batch(batch))
            collection.insert_many(cleaned_batch)
            print(f"Inserted remaining {len(batch)} tweets from {file_path}")
            batch.clear()
                


def assign_files_among_workers(files, num_workers):
    return [files[i::num_workers] for i in range(num_workers)]


def main():
    json_files = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(json_dir, file_name)
            json_files.append(file_path)

    NUM_WORKERS = max(1, os.cpu_count()-1)
    file_chunks = assign_files_among_workers(json_files, NUM_WORKERS)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(load_files, file_chunks)


if __name__ == '__main__':
    start = time.time()
    main()            
    end = time.time()  
    print(f"\nTotal execution time: {end - start:.2f} seconds")
    print("All tweets loaded")
