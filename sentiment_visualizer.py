from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import itertools
warnings.filterwarnings('ignore')

class SentimentVisualizer:
    custom_colors = ['#1f77b4',  # muted blue
                 '#ff7f0e',  # orange
                 '#2ca02c',  # green
                 '#d62728',  # red
                 '#9467bd',  # purple
                 '#8c564b',  # brown
                 '#e377c2',  # pink
                 '#7f7f7f',  # gray
                 '#bcbd22',  # olive
                 '#17becf']  # cyan
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
                "computed_metrics": 1,
                "thread.response_time": 1,
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
                response_time = tweet.get("response_time")
                if label and score is not None and screen_name and airline:
                    role = "support" if screen_name.lower() == airline.lower() else "user"
                    convo_metrics.append({"role" : role})
                    tweet_rows.append({
                        "conversation_id": convo_id,
                        "airline": airline,
                        "screen_name": screen_name,
                        "role": role,
                        "sentiment_label": label,
                        "sentiment_score": score,
                        "tweet_index": i,
                        "created_at": created_at,
                        "response_time": response_time
                        
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
        sns.boxplot(data=self.scores_df, x='airline', y='evolution_score', ax=ax, palette=self.custom_colors[:11])
        ax.set_title('Evolution Score Distribution by Airline')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        self.save_plot(fig, "evolution_score_distribution_by_airline")

    def plot_evolution_category_distribution(self):
        fig, ax = plt.subplots(figsize=(14, 7), nrows=1, ncols=2)
        
        # Get data
        counts_klm = self.scores_df[self.scores_df['airline']=='KLM']['evolution_category'].value_counts()
        counts_others = self.scores_df[self.scores_df['airline']!='KLM']['evolution_category'].value_counts()
        
        # Define a professional color palette
        colors = {
            'Very Negative': '#d32f2f',     # Deep red
            'Negative': '#f57c00',          # Orange  
            'Neutral': '#1976d2',           # Blue
            'Positive': '#388e3c',          # Green
            'Very Positive': '#7b1fa2'      # Purple
        }
        
        # Create ordered categories for consistency
        category_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        
        # Prepare data in consistent order
        klm_values = [counts_klm.get(cat, 0) for cat in category_order if counts_klm.get(cat, 0) > 0]
        klm_labels = [cat for cat in category_order if counts_klm.get(cat, 0) > 0]
        klm_colors = [colors[cat] for cat in klm_labels]
        
        others_values = [counts_others.get(cat, 0) for cat in category_order if counts_others.get(cat, 0) > 0]
        others_labels = [cat for cat in category_order if counts_others.get(cat, 0) > 0]
        others_colors = [colors[cat] for cat in others_labels]
        
        # KLM pie chart
        wedges1, texts1, autotexts1 = ax[0].pie(
            klm_values, 
            labels=klm_labels, 
            autopct='%1.1f%%',
            colors=klm_colors,
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        

        # Others pie chart  
        wedges2, texts2, autotexts2 = ax[1].pie(
            others_values,
            labels=others_labels,
            autopct='%1.1f%%', 
            colors=others_colors,
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        # Style the percentage text
        for autotext in autotexts1 + autotexts2:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        # Set titles with better styling
        ax[0].set_title('KLM Evolution Category Distribution', 
                    fontsize=14, fontweight='bold', pad=20)
        ax[1].set_title('Other Airlines Evolution Category Distribution', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Remove the problematic axhline and main title
        # The axhline doesn't make sense for pie charts and the main title is redundant
        
        # Add a subtle background
        fig.patch.set_facecolor('#f8f9fa')
        
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # Add a legend at the bottom center
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[cat], edgecolor='white', linewidth=1) 
                        for cat in category_order if cat in klm_labels or cat in others_labels]
        legend_labels = [cat for cat in category_order if cat in klm_labels or cat in others_labels]
        
        fig.legend(legend_elements, legend_labels, 
                loc='lower center', ncol=len(legend_labels), 
                bbox_to_anchor=(0.5, -0.05),
                fontsize=11, frameon=False)
        
        # Adjust subplot parameters to make room for legend
        plt.subplots_adjust(bottom=0.15)
        
        self.save_plot(fig, "evolution_category_distribution")

    def plot_conversation_trajectory_by_airline(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Normalize the crosstab by row (i.e., divide each row by its sum)
        tab = pd.crosstab(
            self.scores_df['airline'],
            self.scores_df['conversation_trajectory'],
            normalize='index'  # normalize each airline's row
        )
        
        # Reorder columns to have Worsening at bottom, Improving at top
        desired_order = ['Worsening', 'Stable', 'Improving']
        # Only include columns that actually exist in the data
        column_order = [col for col in desired_order if col in tab.columns]
        tab = tab[column_order]
        
        # Define colors with proper mapping (opposite of current)
        color_map = {
            'Improving': "#095C2D",    # Green (positive outcome)
            'Stable': "#9C9463B6",       # Gold (neutral outcome) 
            'Worsening': "#7B162A"     # Red (negative outcome)
        }
        
        # Get colors for available columns
        colors = [color_map[col] for col in tab.columns]
        
        # Create stacked bar chart
        tab.plot(
            kind='bar', 
            stacked=True, 
            ax=ax,
            color=colors,
            width=0.7,
            edgecolor='white',
            linewidth=1.2
        )
        
        # Enhanced styling
        ax.set_title("Conversation Trajectory Distribution by Airline", 
                    fontsize=16, fontweight='bold', pad=25)
        ax.set_ylabel("Proportion", fontsize=12, fontweight='bold')
        ax.set_xlabel("Airline", fontsize=12, fontweight='bold')
        
        # Format y-axis
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        
        # Improve x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
        # Enhanced legend
        legend = ax.legend(
            title='Trajectory',
            title_fontsize=12,
            fontsize=11,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95
        )
        legend.get_title().set_fontweight('bold')
        
        """ # Add percentage labels on bars (only for segments larger than 5%)
        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height()
                if height > 0.05:  # Only show label if segment is > 5%
                    labels.append(f'{height:.1%}')
                else:
                    labels.append('')
            ax.bar_label(container, labels=labels, label_type='center', 
                        fontsize=9, fontweight='bold', color='white') """
        
        # Add subtle grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Adjust layout to prevent cutoff
        plt.tight_layout()
        
        # Save the plot
        self.save_plot(fig, "normalized_conversation_trajectory_by_airline")

    """ def plot_conversation_score_vs_delta_sent(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.scores_df['conversation_score'], self.scores_df['delta_sent'], alpha=0.5)
        ax.set_title("Conversation Score vs Delta Sentiment")
        ax.set_xlabel("Conversation Score")
        ax.set_ylabel("Delta Sentiment")
        corr = self.scores_df['conversation_score'].corr(self.scores_df['delta_sent'])
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
        self.save_plot(fig, "conversation_score_vs_delta_sent") """

    def plot_sentiment_distribution_by_role(self):
        # Create figure with larger size for better presentation
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Group by role and sentiment, then create normalized data
        tab = self.df_merged.groupby(['role', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Normalize to percentages (0-100)
        tab_normalized = tab.div(tab.sum(axis=1), axis=0) * 100

        # Define color mapping
        color_map = {
            'positive': "#7abd22",    # Green
            'negative': "#911A1A",    # Red  
            'neutral': "#B99774"      # Cyan
        }
        
        # Get colors for available sentiment labels
        colors = [color_map.get(col, '#808080') for col in tab_normalized.columns]
        # Create stacked bar chart
        tab_normalized.plot(
            kind='bar', 
            ax=ax, 
            stacked=True,
            color=colors,
            width=0.6,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Styling improvements
        ax.set_title("Sentiment Distribution by Role", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Role", fontsize=12, fontweight='bold')
        
        # Format y-axis to show percentages
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)])
        
        # Improve x-axis labels
        ax.set_xticklabels(['User', 'Support'], rotation=0, fontsize=11)
        
        # Customize legend
        legend = ax.legend(
            title='Sentiment',
            title_fontsize=11,
            fontsize=10,
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9
        )
        legend.get_title().set_fontweight('bold')
        
        # Add percentage labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='center', 
                        fontsize=9, fontweight='bold', color='white')
        
        # Grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        self.save_plot(fig, "sentiment_distribution_by_role")

    def plot_average_evolution_score_by_airline(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_scores = self.scores_df.groupby('airline')['evolution_score'].mean().sort_values()
        avg_scores.plot(kind='barh', ax=ax, color=self.custom_colors[:len(avg_scores)])
        ax.set_title("Average Evolution Score by Airline")
        ax.set_xlabel("Evolution Score")
        ax.set_xlim(-0.6, 0.6)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        self.save_plot(fig, "average_evolution_score_by_airline")

    def plot_conversation_length_vs_evolution_score(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create hexbin plot to handle overplotting (use clean data)
        clean_df = self.scores_df[['total_tweets', 'evolution_score']].dropna()
        clean_df = clean_df[np.isfinite(clean_df['total_tweets']) & 
                        np.isfinite(clean_df['evolution_score'])]
        
        if len(clean_df) == 0:
            print("No valid data points found!")
            return
            
        hb = ax.hexbin(clean_df['total_tweets'], clean_df['evolution_score'], 
                    gridsize=30, cmap='YlOrRd', alpha=0.8, mincnt=1)
        
        # Add colorbar
        cb = plt.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label('Number of Conversations', fontsize=12, fontweight='bold')
        
        # Clean data for trend line calculation
        clean_data = self.scores_df[['total_tweets', 'evolution_score']].dropna()
        clean_data = clean_data[np.isfinite(clean_data['total_tweets']) & 
                            np.isfinite(clean_data['evolution_score'])]
        
        # Calculate and plot trend line (with error handling)
        try:
            if len(clean_data) > 1:
                z = np.polyfit(clean_data['total_tweets'], clean_data['evolution_score'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_data['total_tweets'].min(), 
                                    clean_data['total_tweets'].max(), 100)
                ax.plot(x_trend, p(x_trend), "b--", alpha=0.8, linewidth=2, 
                        label=f'Trend Line (slope: {z[0]:.3f})')
            else:
                print("Not enough valid data points for trend line")
        except np.linalg.LinAlgError:
            print("Could not calculate trend line due to numerical issues")
        
        # Calculate correlation (with error handling)
        try:
            correlation = clean_data['total_tweets'].corr(clean_data['evolution_score'])
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        # Styling
        ax.set_title('Conversation Length vs Evolution Score\nDensity Plot with Trend Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Total Tweets in Conversation', fontsize=14, fontweight='bold')
        ax.set_ylabel('Evolution Score', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add correlation info
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#f8f9fa')
        
        # Add legend
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        self.save_plot(fig, "conversation_length_vs_evolution_score")

    def plot_conversation_length_vs_evolution_score_alternative(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Clean data first
        clean_df = self.scores_df[['total_tweets', 'evolution_score']].dropna()
        clean_df = clean_df[np.isfinite(clean_df['total_tweets']) & 
                        np.isfinite(clean_df['evolution_score'])]
        
        if len(clean_df) == 0:
            print("No valid data points found for alternative plot!")
            return
        
        # Left plot: 2D histogram
        h = ax1.hist2d(clean_df['total_tweets'], clean_df['evolution_score'], 
                    bins=25, cmap='Blues', alpha=0.8)
        plt.colorbar(h[3], ax=ax1, shrink=0.8, label='Count')
        
        ax1.set_title('2D Histogram View', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Tweets', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Evolution Score', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Box plot by conversation length bins
        # Create bins for conversation length using clean data
        try:
            clean_df_copy = clean_df.copy()
            clean_df_copy['length_bin'] = pd.cut(clean_df_copy['total_tweets'], 
                                            bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
            
            # Remove any NaN bins
            clean_df_copy = clean_df_copy.dropna(subset=['length_bin'])
            
            # Box plot - only include groups with enough data
            box_data = []
            box_labels = []
            for name, group in clean_df_copy.groupby('length_bin'):
                if len(group) > 5:  # Only include bins with sufficient data
                    box_data.append(group['evolution_score'].values)
                    box_labels.append(name)
            
            if len(box_data) > 0:
                bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for box plot', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            
        except Exception as e:
            print(f"Error creating box plot: {e}")
            ax2.text(0.5, 0.5, 'Error creating box plot', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        
        ax2.set_title('Evolution Score by Conversation Length', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Conversation Length Category', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Evolution Score', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Style both plots
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.suptitle('Conversation Length vs Evolution Score - Multiple Views', 
                    fontsize=16, fontweight='bold', y=1.02)
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        self.save_plot(fig, "conversation_length_vs_evolution_score_alternative")


    def plot_conversation_count_by_airline(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get value counts and sort by count (descending)
        airline_counts = self.scores_df['airline'].value_counts()
        
        # Create a professional color palette - using a gradient from dark to light blue
        colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, len(airline_counts)))
        
        # Create the bar plot
        bars = ax.bar(range(len(airline_counts)), airline_counts.values, 
                    color=self.custom_colors[:len(self.scores_df['airline'].value_counts())], edgecolor='white', linewidth=1,
                    alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Airline', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Number of Conversations', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Conversation Count by Airline', fontsize=18, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(range(len(airline_counts)))
        ax.set_xticklabels(airline_counts.index, rotation=45, ha='right', fontsize=11)
        
        # Add value labels on top of bars
        for i, (bar, value) in enumerate(zip(bars, airline_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        # Add subtle grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Improve y-axis formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Set y-axis limits with some padding
        ax.set_ylim(0, max(airline_counts.values) * 1.1)
        
        # Add background color
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        # Improve layout
        plt.tight_layout()
        
        
        self.save_plot(fig, "conversation_count_by_airline")


    def plot_trend_distribution(self):
        if 'trend_score' in self.scores_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(self.scores_df['trend_score'], kde=True, bins=30, ax=ax)
            ax.set_title("Distribution of Trend Score (Sentiment Slope)")
            ax.set_xlabel("Trend Score")
            self.save_plot(fig, "trend_score_distribution")


    def plot_sentiment_over_time_by_airline(self, aggregation):
        fig, ax = plt.subplots(figsize=(14, 8))

        merged = pd.merge(
        self.scores_df[['conversation_id', 'airline', 'evolution_score']],
        self.df_merged[['conversation_id', 'created_at']],
        on='conversation_id',
        how='inner').dropna(subset=['airline', 'created_at', 'evolution_score'])
        
        # Ensure date column is datetime
        merged['created_at'] = pd.to_datetime(merged['created_at'])
        
        # Choose aggregation method and create time groups
        if aggregation == 'day':
            # Group by day
            merged['time_period'] = merged['created_at'].dt.date
            freq_label = 'Daily'
            date_format = '%Y-%m-%d'
        elif aggregation == 'week':
            # Group by week (Monday as start of week)
            merged['time_period'] = merged['created_at'].dt.to_period('W-MON').dt.start_time
            freq_label = 'Weekly'
            date_format = '%Y-%m-%d'
        elif aggregation == 'month':
            # Group by month
            merged['time_period'] = merged['created_at'].dt.to_period('M').dt.start_time
            freq_label = 'Monthly'
            date_format = '%Y-%m'
        
        merged_klm = merged[merged['airline'] == 'KLM'].copy()
        merged_others = merged[merged['airline'] != 'KLM'].copy()

        # Group by time period and airline, calculate mean sentiment
        sentiment_over_time_klm = (merged_klm
                            .groupby(['time_period'])['evolution_score']
                            .mean()
                            .reset_index())
        
        sentiment_over_time_others = (merged_others
                            .groupby(['time_period'])['evolution_score']
                            .mean()
                            .reset_index())
        

        # Convert time_period back to datetime if it isn't already
        sentiment_over_time_klm['time_period'] = pd.to_datetime(sentiment_over_time_klm['time_period'])
        sentiment_over_time_others['time_period'] = pd.to_datetime(sentiment_over_time_others['time_period'])
        
        # Sort by date to ensure proper line plotting
        sentiment_over_time_klm = sentiment_over_time_klm.sort_values('time_period')
        sentiment_over_time_others = sentiment_over_time_others.sort_values('time_period')
        
    
            
        # Ensure data is sorted by time
        
        ax.plot(sentiment_over_time_klm['time_period'], sentiment_over_time_klm['evolution_score'], 
                marker='o', linewidth=2.5, markersize=5,
                label='KLM', 
                color=self.custom_colors[0],
                alpha=0.8)
        
        ax.plot(sentiment_over_time_others['time_period'], sentiment_over_time_others['evolution_score'], 
                marker='o', linewidth=2.5, markersize=5,
                label='Other Airlines', 
                color=self.custom_colors[1],
                alpha=0.8)

        
        # Styling improvements
        ax.set_title(f"{freq_label} Mean Sentiment Over Time by Airline", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel("Mean Sentiment Score", fontsize=12, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        
        # Enhanced legend
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left',
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10
        )
        legend.get_title().set_fontweight('bold')
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Format x-axis based on aggregation
        if aggregation == 'day':
            ax.tick_params(axis='x', rotation=45)
            # Show fewer ticks for daily data to avoid crowding
            ax.locator_params(axis='x', nbins=10)
        elif aggregation == 'week':
            ax.tick_params(axis='x', rotation=45)
            ax.locator_params(axis='x', nbins=12)
        else:  # monthly
            ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal line at sentiment = 0 if your sentiment scores can be negative
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        # Adjust layout to prevent cutoff
        plt.tight_layout()
        
        # Clean up the temporary column
        if 'time_period' in merged.columns:
            merged.drop('time_period', axis=1, inplace=True)
        
        self.save_plot(fig, f"sentiment_over_time_{aggregation}_by_airline")
    
    def plot_delta_sent_against_response_time(self):
        merged = pd.merge(
        self.scores_df[['conversation_id', 'airline', 'delta_sent']],
        self.df_merged[self.df_merged['role']=='user'][['conversation_id', 'response_time']],
        on='conversation_id',
        how='inner')

        per_convo = merged.groupby('conversation_id').agg({
            'delta_sent': 'first',
            'response_time': 'mean'
        }).reset_index()

        ax = sns.scatterplot(data=per_convo, x='response_time', y='delta_sent')
        ax.set_title("Delta Sentiment vs Average Response Time")
        ax.set_xlabel("Average Response Time (Minutes)")
        ax.set_ylabel("Delta Sentiment")
        corr = per_convo['response_time'].corr(per_convo['delta_sent'])

        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.5))
        
        ax.set_xlim(0, 400)
        self.save_plot(ax.get_figure(), "delta_sent_vs_response_time")
    
    def plot_mean_initial_sentiment_over_response_time(self):
        df_merged_initial = self.df_merged[self.df_merged['role'] == 'user'].copy()     
        df_filtered = df_merged_initial.groupby('conversation_id').filter(lambda x: len(x) >= 2)
        # Sort so tweet_index is in order within each conversation
        df_sorted = df_filtered.sort_values(['conversation_id', 'tweet_index'])
        # Get first tweet per conversation
        firsts = df_sorted.groupby('conversation_id').nth(0)
        # Get second tweet per conversation
        seconds = df_sorted.groupby('conversation_id').nth(1)

        # Combine into one DataFrame
        delta_df = pd.DataFrame({
            'initial_delta_sentiment': seconds['sentiment_score'].values - firsts['sentiment_score'].values,
            'response_time': seconds['response_time'].values
        })

        bins = [0, 10, 30, 60, float('inf')]
        labels = ['0-10 mins', '10-30 mins', '30-60 mins', '60+ mins']
        delta_df['response_time_bin'] = pd.cut(
            delta_df['response_time'], 
            bins=bins, 
            labels=labels, 
            right=False
                )

        bin_stats = delta_df.groupby('response_time_bin')['initial_delta_sentiment'].agg(['mean', 'count', 'std'])

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bar chart on the axes
        ax.bar(
            x=bin_stats.index,
            height=bin_stats['mean'],
            yerr=bin_stats['std'],
            capsize=5,
            alpha=0.7
        )

        # Customize the axes
        ax.set_xlabel('Response Time Bins')
        ax.set_ylabel('Mean Delta Sentiment')
        ax.set_title('Delta Sentiment vs. Binned Response Time')
        ax.grid(axis='y', linestyle='--')

        # Annotate sample sizes (optional)
        for i, count in enumerate(bin_stats['count']):
            ax.text(
                x=i,
                y=bin_stats['mean'][i] + 0.02,  # Adjust vertical offset as needed
                s=f'n={count}',
                ha='center'  # Horizontal alignment
                )


        self.save_plot(ax.get_figure(), "initial_delta_sent_vs_response_time")
            
        
    def create_all_visualizations(self):
        self.plot_evolution_score_distribution()
        self.plot_evolution_category_distribution()
        self.plot_conversation_trajectory_by_airline()
        #self.plot_conversation_score_vs_delta_sent()
        self.plot_sentiment_distribution_by_role()
        self.plot_average_evolution_score_by_airline()
        self.plot_conversation_length_vs_evolution_score()
        #self.plot_sentiment_heatmap()
        #self.plot_start_vs_end_sentiment()
        #self.plot_sentiment_progression_sample()
        self.plot_conversation_count_by_airline()
        """ self.plot_sentiment_by_airline_and_role() """
        self.plot_trend_distribution()
        self.plot_conversation_length_vs_evolution_score_alternative()
        self.plot_sentiment_over_time_by_airline(aggregation='week')
        self.plot_delta_sent_against_response_time()
        self.plot_mean_initial_sentiment_over_response_time()


def main():
    vis = SentimentVisualizer(sample_size=100000)  # Use None for full dataset
    vis.load_data_from_mongodb()
    vis.create_all_visualizations()

if __name__ == "__main__":
    main()
