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
    def __init__(self):
        """Initialize the visualizer with MongoDB connection"""
        load_dotenv()
        self.client = MongoClient(os.getenv("DATABASE_URL"))
        self.db = self.client.twitter_db
        self.collection = self.db.conversations
        self.scores_collection = self.db.conversation_scores
        
        #should change latter mby
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.scores_df = None
        self.df_merged = None

        self.save_dir = os.path.join(os.path.dirname(__file__), "plots", "sentiment_evo")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def load_data_from_mongodb(self):
        """Load processed data"""
        print("Loading data from MongoDB...")
        
        # Load scores DataFrame from conversation_scores collection
        scores_cursor = self.scores_collection.find({})
        self.scores_df = pd.DataFrame(list(scores_cursor))
        
        if self.scores_df.empty:
            raise ValueError("No data found in conversation_scores collection. Run the main script first.")
        
        cursor = self.collection.find({}, {
            "_id": 1,
            "airline": 1,
            "thread.user.screen_name": 1,
            "thread.sentiment.label": 1,
            "thread.sentiment.score": 1,
            "computed_metrics": 1
        })
        
        rows = []
        for doc in cursor:
            airline = doc.get("airline")
            thread = doc.get("thread", [])
            computed_metrics = doc.get("computed_metrics", {})
            
            if not isinstance(thread, list):
                continue
            
            for i, tweet in enumerate(thread):
                screen_name = tweet.get("user", {}).get("screen_name")
                sentiment = tweet.get("sentiment", {})
                label = sentiment.get("label", "").lower()
                score = sentiment.get("score")
                
                if label and score is not None and screen_name and airline:
                    role = "support" if screen_name.lower() == airline.lower() else "user"
                    
                    rows.append({
                        "conversation_id": str(doc["_id"]),
                        "airline": airline,
                        "screen_name": screen_name,
                        "role": role,
                        "sentiment_label": label,
                        "sentiment_score": score,
                        "tweet_index": i,
                        "evolution_score": computed_metrics.get("evolution_score"),
                        "evolution_category": computed_metrics.get("evolution_category"),
                        "conversation_trajectory": computed_metrics.get("conversation_trajectory")
                    })
        
        self.df_merged = pd.DataFrame(rows)
        print(f"Loaded {len(self.scores_df)} conversations and {len(self.df_merged)} tweets")
    
    def load_data_from_dataframes(self, scores_df, df_merged):
        """Alternative - load data from existing DataFrames"""
        self.scores_df = scores_df.copy()
        self.df_merged = df_merged.copy()
        print(f"Loaded {len(self.scores_df)} conversations and {len(self.df_merged)} tweets from DataFrames")
    
    def create_main_dashboard(self, save_plots=True):
        """Create the main sentiment anal. dashboard"""
        if self.scores_df is None or self.df_merged is None:
            raise ValueError("Data not loaded. Call load_data_from_mongodb() or load_data_from_dataframes() first.")
        
        print("Creating main sentiment dashboard...")
        
        # Create a figure with multiple subplots - maybe chabge to each plot in a separate file
        # for now we will have one big dashboard
        #
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Sentiment Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Distribution of Evolution Scores by Airline
        plt.subplot(3, 3, 1)
        sns.boxplot(data=self.scores_df, x='airline', y='evolution_score')
        plt.title('Evolution Score Distribution by Airline', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Evolution Score')
        
        # 2. Evolution Categories Distribution
        plt.subplot(3, 3, 2)
        evolution_counts = self.scores_df['evolution_category'].value_counts()
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(evolution_counts)))
        plt.pie(evolution_counts.values, labels=evolution_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Distribution of Evolution Categories', fontweight='bold')
        
        # 3. Conversation Trajectory by Airline
        plt.subplot(3, 3, 3)
        trajectory_airline = pd.crosstab(self.scores_df['airline'], self.scores_df['conversation_trajectory'])
        trajectory_airline.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Conversation Trajectory by Airline', fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend(title='Trajectory', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Correlation between Conversation Score and Delta Score
        plt.subplot(3, 3, 4)
        plt.scatter(self.scores_df['conversation_score'], self.scores_df['delta_sent'], alpha=0.6, s=30)
        plt.xlabel('Conversation Score')
        plt.ylabel('Delta Sentiment')
        plt.title('Conversation Score vs Delta Sentiment', fontweight='bold')
        # Add correlation coefficient to corr b/w conversation_score and delta_sent
        correlation = self.scores_df['conversation_score'].corr(self.scores_df['delta_sent'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Sentiment Distribution by Role
        plt.subplot(3, 3, 5)
        role_sentiment = self.df_merged.groupby(['role', 'sentiment_label']).size().unstack(fill_value=0)
        role_sentiment.plot(kind='bar', ax=plt.gca())
        plt.title('Sentiment Distribution by Role', fontweight='bold')
        plt.xticks(rotation=0)
        plt.legend(title='Sentiment')
        
        # 6. Average Sentiment Score by Airline
        plt.subplot(3, 3, 6)
        avg_scores = self.scores_df.groupby('airline')['evolution_score'].mean().sort_values()
        avg_scores.plot(kind='barh', ax=plt.gca())
        plt.title('Average Evolution Score by Airline', fontweight='bold')
        plt.xlabel('Average Evolution Score')
        
        # 7. Conversation Length vs Evolution Score
        plt.subplot(3, 3, 7)
        plt.scatter(self.scores_df['total_tweets'], self.scores_df['evolution_score'], alpha=0.6, s=30)
        plt.xlabel('Total Tweets in Conversation')
        plt.ylabel('Evolution Score')
        plt.title('Conversation Length vs Evolution Score', fontweight='bold')
        
        # 8. Heatmap of Sentiment Patterns
        plt.subplot(3, 3, 8)
        # create a pivot table for heatmap idfk what a pivot table is
        sentiment_pivot = self.df_merged.pivot_table(
            values='sentiment_score', 
            index='airline', 
            columns='sentiment_label', 
            aggfunc='mean'
        )
        sns.heatmap(sentiment_pivot, annot=True, cmap='RdYlBu_r', center=0.5, ax=plt.gca())
        plt.title('Average Sentiment Score by Airline & Label', fontweight='bold')
        
        # 9. Start vs End Sentiment
        plt.subplot(3, 3, 9)
        plt.scatter(self.scores_df['start_sent'], self.scores_df['end_sent'], alpha=0.6, s=30)
        plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.5)  # Diagonal line
        plt.xlabel('Start Sentiment')
        plt.ylabel('End Sentiment')
        plt.title('Start vs End Sentiment', fontweight='bold')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = os.path.join(self.save_dir, 'sentiment_analysis_dashboard.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Main dashboard saved as '{save_path}'")
        plt.show()
    
    def create_detailed_analysis(self, save_plots=True):
        """Create detailed analysis visualizations"""
        print("Creating detailed analysis visualizations...")

        #second round of visualizations
        #again probably should be in separate files
        #idfk
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Box plot of evolution scores with outliers
        sns.boxplot(data=self.scores_df, x='airline', y='evolution_score', ax=axes[0,0])
        axes[0,0].set_title('Evolution Score Distribution (with outliers)', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Violin plot showing distribution shape
        sns.violinplot(data=self.scores_df, x='airline', y='evolution_score', ax=axes[0,1])
        axes[0,1].set_title('Evolution Score Distribution Shape', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Stacked bar chart of trajectories (normalized)
        trajectory_counts = pd.crosstab(self.scores_df['airline'], self.scores_df['conversation_trajectory'], normalize='index')
        trajectory_counts.plot(kind='bar', stacked=True, ax=axes[1,0])
        axes[1,0].set_title('Conversation Trajectory Proportions by Airline', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Trajectory', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Average metrics by airline
        airline_metrics = self.scores_df.groupby('airline').agg({
            'evolution_score': 'mean',
            'conversation_score': 'mean',
            'delta_sent': 'mean',
            'total_tweets': 'mean'
        })
        
        airline_metrics.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Average Metrics by Airline', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_plots:
            save_path = os.path.join(self.save_dir, 'detailed_sentiment_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed analysis saved as '{save_path}'")
        plt.show()
    
    def create_airline_comparison(self, save_plots=True):
        """Create airline-specific comparison charts"""
        print("Creating airline comparison visualizations...")

        #same here, should be in separate files mby
        #
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Airline Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Evolution score rankings
        airline_avg = self.scores_df.groupby('airline')['evolution_score'].mean().sort_values(ascending=False)
        airline_avg.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Airlines Ranked by Evolution Score', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylabel('Average Evolution Score')
        
        # 2. Conversation volume by airline
        conversation_counts = self.scores_df['airline'].value_counts()
        conversation_counts.plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Number of Conversations by Airline', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylabel('Number of Conversations')
        
        # 3. Average conversation length
        avg_length = self.scores_df.groupby('airline')['total_tweets'].mean().sort_values(ascending=False)
        avg_length.plot(kind='bar', ax=axes[0,2], color='lightgreen')
        axes[0,2].set_title('Average Conversation Length by Airline', fontweight='bold')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].set_ylabel('Average Number of Tweets')
        
        # 4. Sentiment improvement rate
        improving_rate = self.scores_df.groupby('airline')['conversation_trajectory'].apply(
            lambda x: (x == 'Improving').sum() / len(x) * 100
        ).sort_values(ascending=False)
        improving_rate.plot(kind='bar', ax=axes[1,0], color='gold')
        axes[1,0].set_title('Conversation Improvement Rate by Airline', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylabel('% of Improving Conversations')
        
        # 5. Standard deviation of evolution scores (consistency)
        evolution_std = self.scores_df.groupby('airline')['evolution_score'].std().sort_values()
        evolution_std.plot(kind='bar', ax=axes[1,1], color='orange')
        axes[1,1].set_title('Evolution Score Consistency (Lower = More Consistent)', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel('Standard Deviation')
        
        # 6. Correlation heatmap of airline metrics
        airline_summary = self.scores_df.groupby('airline').agg({
            'evolution_score': 'mean',
            'conversation_score': 'mean',
            'delta_sent': 'mean',
            'total_tweets': 'mean'
        })
        correlation_matrix = airline_summary.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Correlation Matrix of Airline Metrics', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            save_path = os.path.join(self.save_dir, 'airline_comparison_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Airline comparison saved as '{save_path}'")
        plt.show()
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal Conversations Analyzed: {len(self.scores_df)}")
        print(f"Total Tweets Analyzed: {len(self.df_merged)}")
        
        print(f"\nEvolution Score Statistics:")
        print(f"Mean: {self.scores_df['evolution_score'].mean():.3f}")
        print(f"Median: {self.scores_df['evolution_score'].median():.3f}")
        print(f"Std Dev: {self.scores_df['evolution_score'].std():.3f}")
        print(f"Min: {self.scores_df['evolution_score'].min():.3f}")
        print(f"Max: {self.scores_df['evolution_score'].max():.3f}")
        
        print(f"\nEvolution Categories:")
        for category, count in self.scores_df['evolution_category'].value_counts().items():
            percentage = (count / len(self.scores_df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print(f"\nConversation Trajectories:")
        for trajectory, count in self.scores_df['conversation_trajectory'].value_counts().items():
            percentage = (count / len(self.scores_df)) * 100
            print(f"  {trajectory}: {count} ({percentage:.1f}%)")
        
        print(f"\nAirline Performance (by average evolution score):")
        airline_performance = self.scores_df.groupby('airline').agg({
            'evolution_score': ['mean', 'count', 'std']
        }).round(3)
        airline_performance.columns = ['avg_score', 'count', 'std_dev']
        airline_performance = airline_performance.sort_values('avg_score', ascending=False)
        
        for airline, stats in airline_performance.iterrows():
            print(f"  {airline}: {stats['avg_score']:.3f} ± {stats['std_dev']:.3f} (n={stats['count']})")
        
        print("\n" + "="*60)
    
    def create_all_visualizations(self, save_plots=True):
        """Create all visualization sets"""
        self.create_main_dashboard(save_plots)
        self.create_detailed_analysis(save_plots)
        self.create_airline_comparison(save_plots)
        self.print_summary_statistics()
        
        if save_plots:
            print(f"\nAll visualizations saved successfully!")
            print("Generated files:")
            print(f"  - {os.path.join(self.save_dir, 'sentiment_analysis_dashboard.png')}")
            print(f"  - {os.path.join(self.save_dir, 'detailed_sentiment_analysis.png')}")
            print(f"  - {os.path.join(self.save_dir, 'airline_comparison_analysis.png')}")

def main():
    """Main function to run visualization if called directly"""
    viz = SentimentVisualizer()
    
    try:
        viz.load_data_from_mongodb()
        viz.create_all_visualizations(save_plots=True)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run the main sentiment analysis script first to populate the database.")

if __name__ == "__main__":
    main()