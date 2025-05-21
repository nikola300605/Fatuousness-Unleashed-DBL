from pymongo import MongoClient
from dotenv import load_dotenv
from dateutil import parser
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
#styles
sns.set_style("whitegrid")
custom_palette = [
    "#4CB5AE", "#F67280", "#F8B195",
    "#FBC687", "#6C5B7B", "#355C7D", "#C06C84"
]
sns.set_palette(custom_palette)


#load environment variables and connect to Mongo
print("Loading environment variables and connecting to MongoDB...")
load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))
db = client.twitter_db
collection = db.conversations

BATCH_SIZE = 1000

def get_conversations_batch(batch_size=BATCH_SIZE):
    cursor = collection.find({}, batch_size=batch_size)
    batch = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

#stats accumulator
print("Initializing accumulators...")
stats = {
    'total_conversations': 0,
    'total_length': 0,
    'sentiment_scores': [],
    'sentiment_labels': [],
    'response_times': [],
    'airline_times': defaultdict(list),
    'daily_sentiments': defaultdict(list),
    'convo_lengths': [],
    'airline_counts': defaultdict(int),
    'first_tweet_dates': []
}

#process batches
print("Starting batch processing...")
batch_number = 1
for batch in get_conversations_batch():
    print(f"Processing batch #{batch_number} with {len(batch)} conversations...")
    for conv in batch:
        stats['total_conversations'] += 1
        conv_length = conv.get('length', 0)
        stats['total_length'] += conv_length
        stats['convo_lengths'].append(conv_length)
        
        thread = conv.get('thread', [])
        airline = conv.get('airline')
        if airline:
            stats['airline_counts'][airline] += 1

        for i, tweet in enumerate(thread):
            #track first tweet dates for time analysis
            if i == 0:
                try:
                    stats['first_tweet_dates'].append(parser.parse(tweet['created_at']))
                except:
                    pass
            
            sentiment = tweet.get('sentiment', {})
            score = sentiment.get('score')
            label = sentiment.get('label')

            if score is not None:
                stats['sentiment_scores'].append(score)
                try:
                    date = parser.parse(tweet['created_at']).date()
                    stats['daily_sentiments'][date].append(score)
                except Exception:
                    continue
            if label:
                stats['sentiment_labels'].append(label.lower())

            #response time calculation
            if i > 0:
                try:
                    t1 = parser.parse(thread[i - 1]['created_at'])
                    t2 = parser.parse(tweet['created_at'])
                    delta = (t2 - t1).total_seconds() / 60
                    stats['response_times'].append(delta)
                    if airline:
                        stats['airline_times'][airline].append(delta)
                except Exception:
                    continue
    batch_number += 1

#terminal output of stats, do a table later
print("\n=== Key Statistics ===")
print(f"Total conversations: {stats['total_conversations']}")
print(f"Average conversation length: {stats['total_length']/stats['total_conversations']:.2f} tweets")
if stats['sentiment_scores']:
    print(f"Mean sentiment score: {np.mean(stats['sentiment_scores']):.2f}")
if stats['response_times']:
    print(f"Average response time: {np.mean(stats['response_times']):.2f} minutes")


#airline summary table, do a table later
airline_data = [
    [airline, stats['airline_counts'][airline], f"{np.mean(stats['airline_times'][airline]):.1f} mins"]
    for airline in stats['airline_counts'] if stats['airline_times'][airline]
]

if airline_data:
    fig, ax = plt.subplots(figsize=(10, len(airline_data) * 0.5 + 1))
    ax.axis('off')
    ax.table(cellText=airline_data, colLabels=["Airline", "Conversations", "Avg Response Time"], loc='center', cellLoc='center')
    ax.set_title('Airline Conversation Stats', pad=10)
    plt.savefig("plots/airline_summary_table.png", bbox_inches='tight')
    plt.show()


#vsualization
print("\nGenerating visualizations...")
os.makedirs("plots", exist_ok=True)

# Summary Table
summary_data = [
    ["Total Conversations", stats['total_conversations']],
    ["Avg Convo Length", f"{stats['total_length']/stats['total_conversations']:.2f}"],
    ["Mean Sentiment", f"{np.mean(stats['sentiment_scores']):.2f}" if stats['sentiment_scores'] else "N/A"],
    ["Avg Response Time", f"{np.mean(stats['response_times']):.2f} mins" if stats['response_times'] else "N/A"]
]
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
ax.table(cellText=summary_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
ax.set_title('Key Statistics', pad=10)
plt.savefig("plots/summary_table.png")
plt.show()

# Conversation Length Distribution (Clipped)
clipped_lengths = [x for x in stats['convo_lengths'] if x <= 10]
plt.figure(figsize=(8, 4))
sns.histplot(clipped_lengths, bins=10, kde=False, color=custom_palette[0])
plt.title("Conversation Length Distribution (Clipped at 10)", fontsize=14, fontweight="bold", color="#355C7D")
plt.xlabel("Number of Tweets")
plt.tight_layout()
plt.savefig("plots/convo_length_clipped_distribution.png")
plt.show()

# Sentiment Label Distribution with Percentages
label_counts = Counter(stats['sentiment_labels'])
labels = list(label_counts.keys())
values = list(label_counts.values())
total = sum(values)
colors = [custom_palette[i % len(custom_palette)] for i in range(len(values))]

plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=values, palette=colors)
plt.title("Sentiment Label Distribution", fontsize=14, fontweight="bold", color="#355C7D")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
for i, v in enumerate(values):
    percent = f"{(v / total) * 100:.1f}%"
    plt.text(i, v + max(values) * 0.01, percent, ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("plots/sentiment_label_distribution.png")
plt.show()

# Sentiment Confidence Distribution
if stats['sentiment_scores']:
    plt.figure(figsize=(8, 4))
    sns.histplot(stats['sentiment_scores'], bins=30, kde=True, color=custom_palette[2])
    plt.title("Sentiment Confidence Distribution", fontsize=14, fontweight="bold", color="#355C7D")
    plt.xlabel("Confidence")
    plt.axvline(np.mean(stats['sentiment_scores']), color='r', linestyle='--', label=f'Mean Confidence: {np.mean(stats["sentiment_scores"]):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sentiment_score_distribution.png")
    plt.show()

# Response Time Distribution (Clipped)
clipped_response_times = [x for x in stats['response_times'] if x <= 500]
plt.figure(figsize=(8, 4))
sns.histplot(clipped_response_times, bins=50, kde=False, color=custom_palette[1])
plt.title("Response Time Distribution (Clipped at 500 mins)", fontsize=14, fontweight="bold", color="#355C7D")
plt.xlabel("Minutes")
plt.tight_layout()
plt.savefig("plots/response_time_clipped_distribution.png")
plt.show()

# Airline Avg Response Times
airline_avg_times = {k: np.mean(v) for k, v in stats['airline_times'].items() if v}
if airline_avg_times:
    colors = [custom_palette[i % len(custom_palette)] for i in range(len(airline_avg_times))]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(airline_avg_times.keys()), y=list(airline_avg_times.values()), palette=colors)
    plt.title("Average Response Time by Airline", fontsize=14, fontweight="bold", color="#355C7D")
    plt.ylabel("Minutes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/airline_response_times.png")
    plt.show()

# Airline Conversation Counts
if stats['airline_counts']:
    colors = [custom_palette[i % len(custom_palette)] for i in range(len(stats['airline_counts']))]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(stats['airline_counts'].keys()), y=list(stats['airline_counts'].values()), palette=colors)
    plt.title("Conversation Count per Airline", fontsize=14, fontweight="bold", color="#355C7D")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/airline_conversation_counts.png")
    plt.show()

# Daily Avg Sentiment
if stats['daily_sentiments']:
    df_trend = pd.DataFrame([
        {"date": d, "avg_sentiment": sum(scores)/len(scores), "count": len(scores)}
        for d, scores in stats['daily_sentiments'].items()
    ]).sort_values("date")
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=df_trend, x="date", y="avg_sentiment", color=custom_palette[4])
    plt.title("Daily Average Sentiment", fontsize=14, fontweight="bold", color="#355C7D")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/daily_avg_sentiment.png")
    plt.show()

# Conversation Lifespan Distribution
print("Calculating conversation lifespans...")
lifespans = []
for batch in get_conversations_batch():
    for conv in batch:
        thread = conv.get('thread', [])
        if len(thread) >= 2:
            try:
                t_start = parser.parse(thread[0]['created_at'])
                t_end = parser.parse(thread[-1]['created_at'])
                delta_minutes = (t_end - t_start).total_seconds() / 60
                if delta_minutes >= 0:
                    lifespans.append(delta_minutes)
            except:
                continue

if lifespans:
    clipped_lifespans = [x for x in lifespans if x <= 1440]
    plt.figure(figsize=(8, 4))
    sns.histplot(clipped_lifespans, bins=50, color=custom_palette[3])
    plt.title("Conversation Lifespan Distribution (Clipped at 1 day)", fontsize=14, fontweight="bold", color="#355C7D")
    plt.xlabel("Lifespan (minutes)")
    plt.tight_layout()
    plt.savefig("plots/convo_lifespan_distribution.png")
    plt.show()
