from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SentimentVisualizer:
    def __init__(self, sample_size=None):
        load_dotenv()
        self.client = MongoClient(os.getenv("DATABASE_URL"))
        self.db = self.client.twitter_db
        self.collection = self.db.conversations

        plt.style.use('default')
        sns.set_palette("husl")

        self.scores_df = None
        self.df_merged = None
        self.sample_size = sample_size

        self.save_dir = os.path.join(os.path.dirname(__file__), "plots", "sentiment_evo")
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data_from_mongodb(self):
        print("Loading data from MongoDB...")

        convo_cursor = self.collection.aggregate([
            {"$sample": {"size": self.sample_size if self.sample_size else 1000000}},  # fallback large number
            {"$project": {
                "airline": 1,
                "thread.created_at": 1,
                "thread.user.screen_name": 1,
                "thread.sentiment.label": 1,
                "thread.sentiment.score": 1,
                "computed_metrics": 1
            }}
        ])

        tweet_rows = []
        convo_metrics = []

        for doc in convo_cursor:
            convo_id = str(doc.get("_id"))
            airline = doc.get("airline")
            thread = doc.get("thread", [])
            computed = doc.get("computed_metrics", {})

            if not isinstance(thread, list) or not computed:
                continue

            convo_metrics.append({
                "conversation_id": convo_id,
                "airline": airline,
                "conversation_score": computed.get("conversation_score"),
                "start_sent": computed.get("start_sent"),
                "end_sent": computed.get("end_sent"),
                "delta_sent": computed.get("delta_sent"),
                "evolution_score": computed.get("evolution_score"),
                "evolution_category": computed.get("evolution_category"),
                "conversation_trajectory": computed.get("conversation_trajectory"),
                "total_tweets": len(thread)
            })

            for i, tweet in enumerate(thread):
                created_at = tweet.get("created_at")
                if isinstance(created_at, str):
                    try:
                        created_at = pd.to_datetime(created_at)
                    except:
                        created_at = None

                screen_name = tweet.get("user", {}).get("screen_name")
                sentiment = tweet.get("sentiment", {})
                label = sentiment.get("label", "").lower()
                score = sentiment.get("score")

                if label and score is not None and screen_name and airline:
                    role = "support" if screen_name.lower() == airline.lower() else "user"
                    tweet_rows.append({
                        "conversation_id": convo_id,
                        "airline": airline,
                        "screen_name": screen_name,
                        "role": role,
                        "sentiment_label": label,
                        "sentiment_score": score,
                        "tweet_index": i,
                        "created_at": created_at
                    })

        self.scores_df = pd.DataFrame(convo_metrics)
        self.df_merged = pd.DataFrame(tweet_rows)

        if self.sample_size and self.sample_size < len(self.scores_df):
            self.scores_df = self.scores_df.sample(n=self.sample_size, random_state=42)
            self.df_merged = self.df_merged[self.df_merged['conversation_id'].isin(self.scores_df['conversation_id'])]

        print(f"Loaded {len(self.scores_df)} conversations and {len(self.df_merged)} tweets")

    def save_plot(self, fig, name):
        path = os.path.join(self.save_dir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {path}")

    def plot_evolution_score_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.scores_df, x='airline', y='evolution_score', ax=ax)
        ax.set_title('Evolution Score Distribution by Airline')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        self.save_plot(fig, "evolution_score_distribution_by_airline")

    def plot_evolution_category_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = self.scores_df['evolution_category'].value_counts()
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(counts)))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors)
        ax.set_title("Evolution Category Distribution")
        self.save_plot(fig, "evolution_category_distribution")

    def plot_conversation_trajectory_by_airline(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        tab = pd.crosstab(self.scores_df['airline'], self.scores_df['conversation_trajectory'])
        tab.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title("Conversation Trajectory by Airline")
        ax.legend(title="Trajectory", bbox_to_anchor=(1.05, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        self.save_plot(fig, "conversation_trajectory_by_airline")

    def plot_conversation_score_vs_delta_sent(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.scores_df['conversation_score'], self.scores_df['delta_sent'], alpha=0.5)
        ax.set_title("Conversation Score vs Delta Sentiment")
        ax.set_xlabel("Conversation Score")
        ax.set_ylabel("Delta Sentiment")
        corr = self.scores_df['conversation_score'].corr(self.scores_df['delta_sent'])
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
        self.save_plot(fig, "conversation_score_vs_delta_sent")

    def plot_sentiment_distribution_by_role(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        tab = self.df_merged.groupby(['role', 'sentiment_label']).size().unstack(fill_value=0)
        tab.plot(kind='bar', ax=ax)
        ax.set_title("Sentiment Distribution by Role")
        ax.set_ylabel("Tweet Count")
        self.save_plot(fig, "sentiment_distribution_by_role")

    def plot_average_evolution_score_by_airline(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_scores = self.scores_df.groupby('airline')['evolution_score'].mean().sort_values()
        avg_scores.plot(kind='barh', ax=ax)
        ax.set_title("Average Evolution Score by Airline")
        ax.set_xlabel("Score")
        self.save_plot(fig, "average_evolution_score_by_airline")

    def plot_conversation_length_vs_evolution_score(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.scores_df['total_tweets'], self.scores_df['evolution_score'], alpha=0.5)
        ax.set_title("Conversation Length vs Evolution Score")
        ax.set_xlabel("Total Tweets")
        ax.set_ylabel("Evolution Score")
        self.save_plot(fig, "conversation_length_vs_evolution_score")

    def plot_sentiment_heatmap(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        pivot = self.df_merged.pivot_table(
            values='sentiment_score',
            index='airline',
            columns='sentiment_label',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, cmap='RdYlBu_r', center=0.5, ax=ax)
        ax.set_title("Average Sentiment Score by Airline & Label")
        self.save_plot(fig, "heatmap_sentiment_score_by_airline_and_label")

    def plot_start_vs_end_sentiment(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.scores_df['start_sent'], self.scores_df['end_sent'], alpha=0.5)
        ax.plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
        ax.set_xlabel("Start Sentiment")
        ax.set_ylabel("End Sentiment")
        ax.set_title("Start vs End Sentiment")
        self.save_plot(fig, "start_vs_end_sentiment")

    #
    """ def plot_sentiment_progression_sample(self, num_samples=5):
        sample_convos = self.df_merged['conversation_id'].drop_duplicates().sample(num_samples, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 6))

        for convo_id in sample_convos:
            subset = self.df_merged[self.df_merged['conversation_id'] == convo_id].sort_values('tweet_index')
            ax.plot(subset['tweet_index'], subset['sentiment_score'], label=convo_id)

        ax.set_title("Sentiment Progression (Sampled Conversations)")
        ax.set_xlabel("Tweet Index")
        ax.set_ylabel("Sentiment Score")
        ax.legend(title="Conversation ID", bbox_to_anchor=(1.05, 1))
        self.save_plot(fig, "sampled_sentiment_progression") """

    def plot_conversation_count_by_airline(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.scores_df['airline'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Conversation Count by Airline")
        ax.set_ylabel("Count")
        ax.set_xlabel("Airline")
        self.save_plot(fig, "conversation_count_by_airline")

    def plot_sentiment_by_airline_and_role(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped = self.scores_df.groupby(['airline', 'role'])['evolution_score'].mean().unstack()
        grouped.plot(kind='bar', ax=ax)
        ax.set_title("Average Sentiment Score by Role and Airline")
        ax.set_ylabel("Avg Sentiment Score")
        ax.set_xlabel("Airline")
        ax.legend(title="Role")
        self.save_plot(fig, "avg_sentiment_score_by_airline_and_role")

    def plot_trend_distribution(self):
        if 'trend_score' in self.scores_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(self.scores_df['trend_score'], kde=True, bins=30, ax=ax)
            ax.set_title("Distribution of Trend Score (Sentiment Slope)")
            ax.set_xlabel("Trend Score")
            self.save_plot(fig, "trend_score_distribution")





    def create_all_visualizations(self):
        self.plot_evolution_score_distribution()
        self.plot_evolution_category_distribution()
        self.plot_conversation_trajectory_by_airline()
        self.plot_conversation_score_vs_delta_sent()
        self.plot_sentiment_distribution_by_role()
        self.plot_average_evolution_score_by_airline()
        self.plot_conversation_length_vs_evolution_score()
        self.plot_sentiment_heatmap()
        self.plot_start_vs_end_sentiment()
        #self.plot_sentiment_progression_sample()
        self.plot_conversation_count_by_airline()
        self.plot_sentiment_by_airline_and_role()
        self.plot_trend_distribution()


def main():
    vis = SentimentVisualizer(sample_size=10000)  # Use None for full dataset
    vis.load_data_from_mongodb()
    vis.create_all_visualizations()

if __name__ == "__main__":
    main()
