from bson import json_util, ObjectId
import json
from pandas import json_normalize
from pymongo_interface import get_documents_batch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
""" from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d """

def process_batch(batch):
    sanitised = json.loads(json_util.dumps(batch))
    normalised = json_normalize(sanitised)
    df = pd.DataFrame(normalised)

    return df

def get_top_languages(df):
    """ Get the top 10 languages used in the tweets """
    lang_counts = df['lang'].value_counts().head(10)
    lang_full_df = lang_counts.reset_index() 
    return lang_full_df 

def main():
    i = 0
    for batch in get_documents_batch():
        df = process_batch(batch)
        print(df.info())

        if i == 5:
            break
        i += 1
    
    # Uncomment the following lines to enable multiprocessing
    """ MAX_WORKERS = os.cpu_count() - 1 
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        i = 0
        for batch in get_documents_batch():
            futures.append(executor.submit(process_batch, batch))
            if i == 10:
                break
            i += 1
        
        for future in as_completed(futures):
            result = future.result()

            print(result.info()) """
    ############################################################################

    """ # Add new columns for analysis
    full_df['created_at'] = pd.to_datetime(full_df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    full_df['date'] = full_df['created_at'].dt.date
    full_df['weekday'] = full_df['created_at'].dt.day_name()

    # Airline mentions analysis
    airlines_user_ids = {
        56377143: 'KLM',
        106062176: 'AirFrance',
        18332190: 'British Airways',
        22536055: 'AmericanAir',
        124476322: 'Lufthansa',
        26223583: 'AirBerlin',
        2182373406: 'AirBerlin assist',
        38676903: 'easyJet',
        1542862735: 'RyanAir',
        253340062: 'SingaporeAir',
        218730857: 'Qantas',
        45621423: 'EtihadAirways',
        20626359: 'VirginAtlantic',
    }

    airlines_mentions = Counter()

    full_df['airline_mentioned'] = False
    full_df['airline_m'] = None
    for idx, row in full_df.iterrows():
        airline_mentioned = False
        airline_name = None

        if row.get('in_reply_to_user_id') is not None and not pd.isna(row['in_reply_to_user_id']):
            user_id = int(row['in_reply_to_user_id'])
            if user_id in airlines_user_ids:
                airline_name = airlines_user_ids[user_id]
                airlines_mentions[airline_name] += 1
                airline_mentioned = True
                continue

        if isinstance(row.get('retweeted_status'), dict):
            retweeted_status = row['retweeted_status']
            retweeted_uid = retweeted_status['user'].get('id')
            if retweeted_uid in airlines_user_ids:
                airline_name = airlines_user_ids[retweeted_uid]
                airlines_mentions[airline_name] += 1
                airline_mentioned = True
                continue
        
        if row.get('entities') is not None:
            entities = row['entities']
            if isinstance(entities, dict):
                user_mentions = entities['user_mentions']
                for mention in user_mentions:
                    user_id = mention['id']
                    if user_id in airlines_user_ids:
                        airline_name = airlines_user_ids[user_id]
                        airlines_mentions[airline_name] += 1
                        airline_mentioned = True
                        break
        
        full_df.at[idx, 'airline_mentioned'] = airline_mentioned
        full_df.at[idx, 'airline_m'] = airline_name

    # Plotting
    full_df_mentions = pd.DataFrame(airlines_mentions.items(), columns=['Airline', 'Mentions'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=full_df_mentions, x='Airline', y='Mentions')
    plt.title("Number of Tweet Mentions per Airline")
    plt.xlabel("Airline")
    plt.ylabel("Mentions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    reply_counts = pd.Series({
        'Reply': full_df['in_reply_to_status_id'].notnull().sum(),
        'Non-Reply': full_df['in_reply_to_status_id'].isnull().sum()
    })
    reply_full_df = reply_counts.reset_index()
    reply_full_df.columns = ['Type', 'Count']

    plt.figure(figsize=(6, 4))
    sns.barplot(data=reply_full_df, x='Type', y='Count', palette='Set2')
    plt.title('Number of Reply vs. Non-Reply Tweets')
    plt.ylabel('Tweet Count')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()

    lang_counts = full_df['lang'].value_counts().head(10)
    lang_full_df = lang_counts.reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=lang_full_df, x='lang', y='count', palette='Set2')
    plt.title('Top 10 languages used in Tweets')
    plt.ylabel('Tweet Count')
    plt.xlabel('Languages')
    plt.tight_layout()
    plt.show()

    daily_tweet_counts = full_df.groupby('date').size().reset_index(name='tweet_count')
    daily_tweet_counts['smoothed'] = gaussian_filter1d(daily_tweet_counts['tweet_count'], sigma=2)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_tweet_counts, x='date', y='tweet_count', label='Smoothed', color='red')
    plt.title('Smoothed Tweet Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Tweet Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
 """
if __name__ == "__main__":
    main()
