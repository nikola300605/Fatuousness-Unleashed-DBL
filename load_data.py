
from pymongo import MongoClient
import ijson
import os
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv("DATABASE_URL")

connection_string = database_url
client = MongoClient(connection_string) #left the connect like that, should fix (env or smth)
db = client.twitter_db
collection = db.tweets
print(client.list_database_names())
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


def clean_data(data: dict) -> dict:
   

    purge_user_columns: list[str] = [
    "url", "description", "translator_type", "friends_count", "listed_count",
    "favourites_count", "statuses_count", "created_at", "utc_offset", "time_zone",
    "lang", "contributors_enabled", "is_translator", "profile_background_color",
    "profile_background_image_url", "profile_background_image_url_https", "profile_background_tile",
    "profile_link_color", "profile_sidebar_border_color", "profile_sidebar_fill_color",
    "profile_text_color", "profile_use_background_image", "profile_image_url",
    "profile_image_url_https", "profile_banner_url", "default_profile_image",
    "following", "follow_request_sent", "notifications", "protected", "is_translation_enabled", ]
    
    purge_place_columns: list[str] = ["bounding_box", "attributes", "country", "place_type", "name", "url"]

    purge_entities_columns: list[str] = ['symbols', 'polls']

    purge_other_columns: list[str] = ["geo", "coordinates", "favorited", "retweeted", "possibly_sensitive", "filter_level", "contributors", "truncated", "extended_entities"]

    for field in purge_other_columns:
        if field in data.keys():
            data.pop(field)

    for field in purge_user_columns:
        if 'user' in data.keys():
            if data['user'] != None:
                if field in data["user"].keys():
                    data['user'].pop(field)
    
    for field in purge_place_columns:
        if 'place' in data.keys():
            if data['place'] != None:
                if field in data['place'].keys():
                    data['place'].pop(field)
    
    for field in purge_entities_columns:
        if 'entities' in data.keys():
            if data['entities'] != None:
                if field in data['entities'].keys():
                    data['entities'].pop(field)
    return data  

for file_name in os.listdir(json_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(json_dir, file_name)
        with open(file=file_path, encoding='utf-8') as file:
            objects = ijson.items(file, '', multiple_values=True)
            for obj in objects:
                try:
                    obj = clean_data(obj)
                    obj = convert_decimals(obj)
                    collection.insert_one(obj)
                    print(f"Inserted tweet with id: {obj['id']}")
                except Exception as e:
                    print(f"Error: {e}")
                
                

print("All tweets loaded")
