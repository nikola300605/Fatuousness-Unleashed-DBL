
from pymongo import MongoClient
import json
import ijson
import os

connection_string = "mongodb+srv://nikolacupic555:hopacupa@tweet-cluster.ruxakzs.mongodb.net/?retryWrites=true&w=majority&appName=tweet-cluster"
client = MongoClient(connection_string) #left the connect like that, should fix (env or smth)

db = client.twitter_db
collection = db.tweets
print(client.list_database_names())
json_dir = './tweets_json/'

def clean_data(data: dict) -> dict:
   

    purge_user_columns: list[str] = [
    "url", "description", "translator_type", "friends_count", "listed_count",
    "favourites_count", "statuses_count", "created_at", "utc_offset", "time_zone",
    "lang", "contributors_enabled", "is_translator", "profile_background_color",
    "profile_background_image_url", "profile_background_image_url_https", "profile_background_tile",
    "profile_link_color", "profile_sidebar_border_color", "profile_sidebar_fill_color",
    "profile_text_color", "profile_use_background_image", "profile_image_url",
    "profile_image_url_https", "profile_banner_url", "default_profile", "default_profile_image",
    "following", "follow_request_sent", "notifications", "protected"]
    
    purge_place_columns: list[str] = ["bounding_box", "attributes"]

    purge_other_columns: list[str] = ["geo", "coordinates", "favorite_count", "favorited", "retweeted", "possibly_sensitive", "filter_level", "contributors"]

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
                if field in data['user'].keys():
                    data['place'].pop(field)
    return data  
    
for file_name in os.listdir(json_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(json_dir, file_name)
        with open(file=file_path, encoding='utf-8') as file:
            objects = ijson.items(file, '', multiple_values=True)
            for obj in objects:
                clean_data(obj)
                print(obj)
                

print("All tweets loaded")

