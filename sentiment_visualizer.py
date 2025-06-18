from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
from scipy.ndimage import gaussian_filter1d
warnings.filterwarnings('ignore')
from scipy import stats

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
                "resolved": 1,
                "topic" : 1
            }}
        ])

        tweet_rows = []
        convo_metrics = []

        for doc in convo_cursor:
            convo_id = str(doc.get("_id"))
            airline = doc.get("airline")
            thread = doc.get("thread", [])
            computed = doc.get("computed_metrics", {})
            resolved = doc.get("resolved", None)
            topic = doc.get("topic", None)

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
                "total_tweets": len(thread),
                "resolved": resolved,
                "topic": topic
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

    def apply_enhanced_styling(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)

    def save_plot(self, fig, name):
        path = os.path.join(self.save_dir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {path}")

    def plot_evolution_score_distribution(self):

        scores = self.scores_df['evolution_score'].tolist()
        
        if not scores:
            print("No evolution scores found.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))

        n, bins, patches = ax.hist(scores, bins=30, alpha=0.7, color=self.custom_colors[0], 
                                edgecolor='white', linewidth=1)
        
        for i, patch in enumerate(patches):
            alpha = 0.4 + 0.6 * (i / len(patches))
            patch.set_alpha(alpha)
        
        density = stats.gaussian_kde(scores)
        xs = np.linspace(min(scores), max(scores), 200)

        density_values = density(xs)
        density_scaled = density_values * len(scores) * (bins[1] - bins[0])

        ax.plot(xs, density_scaled, color=self.custom_colors[3], linewidth=3, 
                alpha=0.8, label='Density Curve')
        
        mean_score = sum(scores) / len(scores)
        median_score = np.median(scores)
        
        ax.axvline(mean_score, color=self.custom_colors[1], linestyle='--', linewidth=2.5, 
                alpha=0.9, label=f'Mean: {mean_score:.2f}')
        ax.axvline(median_score, color=self.custom_colors[2], linestyle=':', linewidth=2.5, 
                alpha=0.9, label=f'Median: {median_score:.2f}')
        
        self.apply_enhanced_styling(ax, "Evolution Score Distribution", 
                            "Evolution Score", "Number of Conversations")
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                framealpha=0.9, edgecolor='gray')
    
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
        tab = self.df_merged[self.df_merged['airline'] == "KLM"].groupby(['role', 'sentiment_label']).size().unstack(fill_value=0)
        
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
        ax.set_title("Sentiment Distribution by Role for KLM", 
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


        if aggregation == "day":
            sentiment_over_time_klm['evolution_score'] = gaussian_filter1d(sentiment_over_time_klm['evolution_score'], sigma=2)
            sentiment_over_time_others['evolution_score'] = gaussian_filter1d(sentiment_over_time_others['evolution_score'], sigma=2)
    
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
        ax.set_title(f"{freq_label} Mean Sentiment (Evolution Score) Over Time by Airline", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel("Mean Sentiment Score", fontsize=12, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylim(-0.7,0.7)
        
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
    
    def plot_evolution_score_distribution(self):
        fig,ax = plt.subplots(figsize=(12, 4))

        sns.histplot(data=self.scores_df['evolution_score'], kde=True, bins=30, color=self.custom_colors[0], ax=ax)
        ax.set_title("Evolution Score Distribution")
        ax.set_xlabel("Evolution Score")
        ax.set_ylabel("Frequency")
        ax.axvline(0, color='red', linestyle='--', label='Neutral Score (0)')
        ax.legend()

        self.save_plot(fig, "evolution_score_distribution")
            
    def plot_convo_length_vs_evolution_patterns(self):
        fig, ax = plt.subplots(figsize=(14, 8))
    
        # Define bins and labels
        length_bins = [0, 3, 7, 12, 20, float('inf')]
        labels = ['Very Short (1-3)', 'Short (4-7)', 'Medium (8-12)', 'Long (13-20)', 'Very Long (20+)']
        
        df = self.scores_df.copy()
        df['length_bin'] = pd.cut(df['length'], bins=length_bins, labels=labels)

        # Boxplot of evolution_score by binned length
        bin_count = df['length_bin'].nunique()
        box_plot = sns.boxplot(
            data=df, 
            x='length_bin', 
            y='evolution_score', 
            palette=self.custom_colors,
            linewidth=2.5,
            boxprops=dict(alpha=0.8, linewidth=2),
            whiskerprops=dict(linewidth=2.5),
            capprops=dict(linewidth=2.5),
            medianprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7, markeredgecolor='darkred')
        )
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        self.apply_enhanced_styling(ax=ax, title="Evolution Score by Conversation Length", xlabel="Conversation Length Bin", ylabel="Evolution Score")

        plt.tight_layout()

        self.save_plot(fig, "convo_length_vs_evolution_patterns")

    def plot_sentiment_journey(self):
        fig,ax = plt.subplots(figsize=(12, 6))   
        """ sns.scatterplot(data=self.scores_df, x='start_sent', y='end_sent', hue='evolution_score', ax=ax) """
        plt.scatter(self.scores_df['start_sent'], self.scores_df['end_sent'], alpha=0.6, c=self.scores_df['evolution_score'], 
           cmap='RdYlBu', s=50)
        plt.colorbar(label='Evolution Score')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Start Sentiment')
        plt.ylabel('End Sentiment')
        plt.title('Conversation Sentiment Trajectories')

        # Add quadrant labels
        plt.text(0.5, 0.5, 'Positive→Positive\n(Maintained)', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        plt.text(-0.5, 0.5, 'Negative→Positive\n(Improved)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        plt.text(0.5, -0.5, 'Positive→Negative\n(Deteriorated)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
        plt.text(-0.5, -0.5, 'Negative→Negative\n(Persisted)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        self.save_plot(fig, "sentiment_journey")

    def plot_comparative_analysis(self):
        
        fig,ax = plt.subplots(figsize=(16, 8))
        success_rate = self.scores_df.groupby('airline')['evolution_score'].apply(lambda x: (x > 0).mean())
        sns.barplot(data=success_rate.reset_index(), x='airline', y='evolution_score', ax=ax, palette=self.custom_colors)
        ax.set_title("Success Rate by Airline (Mean evolution score for cases when it's above 0)")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Airline")
        ax.set_ylabel("Success Rate")

        self.save_plot(fig, "comparative_analysis_success_rate")

        fig, ax = plt.subplots(figsize=(16, 8))
        avg_length = self.scores_df.groupby('airline')['total_tweets'].mean().reset_index()
        avg_length = avg_length.sort_values(by='total_tweets')
        sns.barplot(data=avg_length, x='airline', y='total_tweets', ax=ax, palette=self.custom_colors)
        ax.set_ylim(0, 4)
        ax.set_title("Average Conversation Length by Airline")
        ax.set_xlabel("Airline")
        ax.set_ylabel("Average Length (Tweets)")

        self.save_plot(fig, "comparative_analysis_avg_length")

        fig, ax = plt.subplots(figsize=(16, 8))
        components_avg = self.scores_df.groupby('airline')[['conversation_score', 'delta_sent']].mean()
        components_avg.plot(kind='bar', ax=ax, color=self.custom_colors)
        ax.set_title("Average Conversation Components by Airline")
        ax.set_xlabel("Airline")
        ax.legend(['Conversation Score', 'Delta Sentiment'])
        ax.set_ylim(-0.7, 0.7)

        self.save_plot(fig, "comparative_analysis_airlines")
    
    def plot_score_drivers(self):
        fig,ax = plt.subplots(figsize=(12, 6))

        correlation_matrix = self.scores_df[['conversation_score', 'delta_sent', 'evolution_score', 
                                'start_sent', 'end_sent', 'total_tweets']].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', center=0, 
                    square=True, fmt='.3f', ax=ax)
        plt.title('Score Component Correlations')
        self.save_plot(fig, "score_drivers_correlation_heatmap")

        # Individual component distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        components = ['conversation_score', 'delta_sent', 'start_sent', 'end_sent', 'evolution_score']
        for i, component in enumerate(components):
            axes[i].hist(self.scores_df[component], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{component.replace("_", " ").title()} Distribution')
            axes[i].axvline(self.scores_df[component].mean(), color='red', linestyle='--', label='Mean')
            axes[i].legend()

        axes[5].axis('off')  # Hide the last subplot
        plt.tight_layout()
        self.save_plot(fig, "score_drivers_component_distributions")


    def plot_extreme_cases(self):
        extreme_cases = pd.DataFrame({
            'Best Improvement': [self.scores_df.loc[self.scores_df['evolution_score'].idxmax(), 'evolution_score']],
            'Worst Deterioration': [self.scores_df.loc[self.scores_df['evolution_score'].idxmin(), 'evolution_score']],
            'Highest Start Negative': [self.scores_df.loc[self.scores_df['start_sent'].idxmin(), 'start_sent']],
            'Biggest Turnaround': [self.scores_df.loc[self.scores_df['delta_sent'].idxmax(), 'delta_sent']],
            'Biggest Decline': [self.scores_df.loc[self.scores_df['delta_sent'].idxmin(), 'delta_sent']]
        })

        print("Extreme Cases for Manual Review:")
        print(extreme_cases)

        # Scatter plot highlighting outliers
        fig,ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(
            self.scores_df['conversation_score'],
            self.scores_df['delta_sent'],
            alpha=0.6,
            s=self.scores_df['total_tweets'] * 2,
            c=self.scores_df['evolution_score'],
            cmap='RdYlBu'
        )
        
        plt.colorbar(sc, ax=ax, label='Evolution Score')
        ax.set_xlabel('Conversation Score')
        ax.set_ylabel('Delta Sentiment')
        ax.set_title('Component Relationship (Size = Conversation Length)')

        self.save_plot(fig, "extreme_cases_scatter_plot_1")

        # Highlight extreme cases
        extremes = [self.scores_df['evolution_score'].idxmax(), self.scores_df['evolution_score'].idxmin(), 
           self.scores_df['delta_sent'].idxmax(), self.scores_df['delta_sent'].idxmin(), self.scores_df['conversation_score'].idxmax(),
             self.scores_df['conversation_score'].idxmin()]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        df_with_extremes = pd.DataFrame(columns=['conversation_id', 'conversation_score', 'delta_sent', 'evolution_score'])
        for i,idx in enumerate(extremes):
            sc = ax.scatter(
                self.scores_df.loc[idx, 'conversation_score'],
                self.scores_df.loc[idx, 'delta_sent'],
                s=200,
                c= self.scores_df.loc[idx, 'evolution_score'],
                vmin=-1.0,
                vmax=1.0,
                cmap='RdYlBu',
                edgecolor='black',
            )
            

            df_with_extremes.loc[i] = [self.scores_df.loc[idx, 'conversation_id'], self.scores_df.loc[idx, 'conversation_score'],
                self.scores_df.loc[idx, 'delta_sent'], self.scores_df.loc[idx, 'evolution_score']]
        plt.colorbar(sc, ax=ax, label='Evolution Score')    
        ax.set_xlabel('Conversation Score')
        ax.set_ylabel('Delta Sentiment')
        ax.set_title('Extreme Cases Highlighted in Scatter Plot')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        self.save_plot(fig, "extreme_cases_scatter_plot_2")
        print(df_with_extremes)
        
    
    def plot_different_evo_score_weights(self):
        fig, ax = plt.subplots(figsize=(12, 6), nrows=2, ncols=2)
        for i, weight in enumerate([0.5, 0.6, 0.8, 0.9]):
            alternative_score = weight * self.scores_df['conversation_score'] + (1-weight) * self.scores_df['delta_sent']
            sns.scatterplot(x=self.scores_df['evolution_score'],y=alternative_score, ax=ax[i//2][i%2], alpha=0.6)
            ax[i//2][i%2].plot([self.scores_df['evolution_score'].min(), self.scores_df['evolution_score'].max()], 
             [self.scores_df['evolution_score'].min(), self.scores_df['evolution_score'].max()], 'r--')
            
            ax[i//2][i%2].set_title(f"Evolution Score vs Alternative Score (Weight: {weight})")
            ax[i//2][i%2].set_xlabel("Current Evolution Score")
            ax[i//2][i%2].set_ylabel("Alternative Evolution Score")

            correlation = self.scores_df['evolution_score'].corr(alternative_score)
            plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes)

        self.save_plot(fig, "evolution_score_weights_comparison")
    
    def plot_resolve_rate_per_trend(self):
        # Resolve rate per trend
        fig,ax = plt.subplots(figsize=(12,6))

        by_topic = self.scores_df.groupby(by='topic')['resolved'].mean().reset_index()

        sns.barplot(data=by_topic, x='topic', y='resolved', palette=self.custom_colors)
        ax.set_title('Resolution rate per topic')

        ax.set_xlabel('Topic')
        ax.set_ylabel('Resolution Rate')
        ax.set_ylim(0, 1.0)

        self.save_plot(fig, "resolve_rate_per_trend")

    def plot_average_evo_score_per_topic(self):
        
        topic_mapping = {
        'delay': 'Flight Delays',
        'customer_service': 'Customer Service',
        'booking': 'Booking & Reservations',
        'luggage': 'Luggage & Baggage',
        'other': 'Other Issues',
    }

        fig, ax = plt.subplots(figsize=(12, 7))

        df_plot = self.scores_df.copy()
        df_plot['topic_display'] = df_plot['topic'].map(topic_mapping).fillna(df_plot['topic'])
        unique_topics = df_plot['topic_display'].nunique()

        box_plot = sns.boxplot(
            data=df_plot, 
            x='topic_display', 
            y='evolution_score', 
            ax=ax,
            palette=self.custom_colors[:unique_topics],
            linewidth=2,
            boxprops=dict(alpha=0.8),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            medianprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.6)
        )


        self.apply_enhanced_styling(ax, 'Average Evolution Score per Topic', 'Topic', 'Evolution Score')
        ax.set_ylim(-1, 1)
        plt.style.use('seaborn-v0_8-whitegrid')
        self.save_plot(fig, 'average_evolution_score_per_topic')

    
    def plot_resolved_per_topic_and_airline(self):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)

        self.cores_df['airline_group'] = self.scores_df['airline'].apply(lambda x: 'KLM' if x == 'KLM' else 'Other Airlines')

        grouped = self.scores_df.groupby(['airline_group', 'topic', 'resolved']).size().reset_index(name='count')
        grouped['total'] = grouped.groupby(['airline_group', 'topic'])['count'].transform('sum')
        grouped['percentage'] = (grouped['count'] / grouped['total']) * 100
        resolved_only = grouped[grouped['resolved'] == True]

        topic_mapping = {
            'booking': 'Booking &\nReservations',
            'customer_service': 'Customer\nService',
            'delay': 'Flight\nDelays',
            'luggage': 'Luggage &\nBaggage',
            'other': 'Other\nIssues'
        }

        resolved_only['topic_display'] = resolved_only['topic'].map(topic_mapping).fillna(resolved_only['topic'])


        for i, group in enumerate(['KLM', 'Other Airlines']):
            data = resolved_only[resolved_only['airline_group'] == group]

            topic_count = data['topic'].nunique()

            bars = sns.barplot(
                data=data, 
                x='topic_display', 
                y='percentage', 
                palette=self.custom_colors,
                ax=ax[i],
                alpha=0.8,
                edgecolor='white',
                linewidth=2
            )
            if group == 'KLM':
                ax[i].set_title('KLM Royal Dutch Airlines - Resolution Rate by Topic', 
                            fontsize=16, fontweight='bold', pad=15,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.7))
            else:
                ax[i].set_title('Other Airlines - Resolution Rate by Topic', 
                            fontsize=16, fontweight='bold', pad=15,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", alpha=0.7))
            
            ax[i].set_title(f'{group} - Resolution % by Topic')
            ax[i].set_ylabel('Resolved %')
            ax[i].set_ylim(0, 100)

            for bar in bars.patches:
                height = bar.get_height()
                if height > 0:  # Only add label if there's a bar
                    ax[i].annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2., height),
                                xytext=(0, 5),  # 5 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=11, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                    
        ax[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        # Set x-axis label only for bottom subplot
        ax[1].set_xlabel('Topic Category', fontsize=13, fontweight='semibold', labelpad=10)   
            

        self.save_plot(fig, 'resolved_per_topic_and_airlines')

    def plot_response_time_vs_resolution_rate(self):
        merged = pd.merge(self.df_merged[['conversation_id', 'response_time']], self.scores_df[['conversation_id', 'resolved', 'evolution_score', 'total_tweets', 'topic']],
                           on='conversation_id', how='inner').dropna(subset=['response_time', 'resolved', 'total_tweets', 'evolution_score', 'topic'])

        topic_stats = merged.groupby('topic').agg({
        'response_time': 'mean',
        'resolved': 'mean',
        'evolution_score': 'mean',
        'total_tweets': 'mean'
        }).reset_index()

        fig, ax = plt.subplots(figsize = (12,6))

        scatter = ax.scatter(topic_stats['response_time'], topic_stats['resolved'], 
                                s=topic_stats['total_tweets']*10, alpha=0.7, c=topic_stats['evolution_score'], 
                                cmap='RdYlBu', vmin=-0.5, vmax=0.5)
        ax.set_xlabel('Average Response Time')
        ax.set_ylabel('Resolution Rate')
        ax.set_xlim(0, 0.5)
        ax.set_title('Topic Difficulty Analysis\n(Size=Avg Length, Color=Evolution Score)')
        plt.colorbar(scatter, ax=ax)

        # Add topic labels
        for i, topic in enumerate(topic_stats['topic']):
            ax.annotate(topic, (topic_stats['response_time'].iloc[i], 
                                    topic_stats['resolved'].iloc[i]), fontsize=8)
            
        self.save_plot(fig, "response_time_vs_resolved_rate")

    def plot_average_response_time_per_topic(self):
        fig,ax = plt.subplots(figsize = (12,6))
        merged = pd.merge(
            self.scores_df[['conversation_id', 'topic']],
            self.df_merged[['conversation_id', 'response_time']],
            how='inner', on='conversation_id'
        ).dropna(subset=['topic', 'response_time'])

        #Option 1 - barplots
        per_topic = merged.groupby('topic')['response_time'].mean().reset_index()    
        per_topic = per_topic.sort_values(by='response_time')
        sns.barplot(data=per_topic, x='topic', y='response_time', palette=self.custom_colors, ax=ax)
        ax.set_title('Average Response Time Per Topic')
        ax.set_xlabel('Topic')
        ax.set_ylabel('Average Response Time [minutes]')

        self.save_plot(fig, "average_response_time_per_topic")

    def plot_convo_length_vs_resolution(self):
        fig, ax = plt.subplots(figsize = (12,6))
        df = self.scores_df.copy()
        
        resolved_lengths = df[df['resolved'] == True].groupby('topic')['total_tweets'].mean()
        unresolved_lengths = df[df['resolved'] == False].groupby('topic')['total_tweets'].mean()
        topic_length_comparison = pd.DataFrame({
            'Resolved': resolved_lengths,
            'Unresolved': unresolved_lengths
        }).fillna(0)
        topic_length_comparison.plot(kind='bar', ax=ax)
        ax.set_title('Average Conversation Length by Topic and Resolution')
        ax.set_ylabel('Average Length')
        ax.tick_params(axis='x', rotation=45)

        self.save_plot(fig, "convo_length_vs_resolution")
    
    def plot_response_time_by_day(self):
        # Extract hour from response times for temporal analysis
        df = self.df_merged.copy()
        df['response_day'] = pd.to_datetime(df['created_at']).dt.day_name()


        # Create response time heatmap by hour and day
        response_pivot = df.groupby(['response_day'])['response_time'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        response_pivot['response_day'] = pd.Categorical(
            response_pivot['response_day'],
            categories=day_order,
            ordered=True
        )
        response_pivot = response_pivot.sort_values('response_day')

        fig,ax = plt.subplots(figsize = (16,8))
        sns.barplot(data=response_pivot, x='response_day', y='response_time', palette=self.custom_colors)
        ax.set_title('Average Response Time by Day')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Response Time [minutes]')
        self.save_plot(fig, "average_response_time_by_day")
    

    def plot_label_distribution(self):
        if self.df_merged.empty:
            print("No tweet data available for sentiment plot.")
            return
        
        # Get sentiment counts and capitalize the labels
        sentiment_counts = self.df_merged["sentiment_label"].value_counts()
        
        # Create a mapping for capitalized labels
        capitalized_labels = [label.capitalize() for label in sentiment_counts.index]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = self.custom_colors[:len(sentiment_counts)]
        bars = ax.bar(capitalized_labels, sentiment_counts.values,
                    color=colors, alpha=0.85, 
                    edgecolor='white', linewidth=2,
                    width=0.7)
        
        for bar, color in zip(bars, colors):
            
            bar.set_alpha(0.9)
            bar.set_edgecolor('white')
            bar.set_linewidth(2)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(sentiment_counts.values) * 0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
        self.apply_enhanced_styling(ax, "Sentiment Distribution Analysis", 
                            "Sentiment Category", "Number of Tweets")
        
        self.save_plot(fig, "label_distribution")
        

    ####################################################################
    #This is some bulshit completely coocked by Claude, I have no fucking idea what any of this is.
    def plot_three_way_visualisation(self): #CRAAAZY name btw
        # 3D analysis of topic, response time, and resolution
        fig = plt.figure(figsize=(15, 5))

        df = pd.merge(
            self.scores_df[['conversation_id', 'resolved', 'topic', 'total_tweets', 'evolution_score']],
            self.df_merged[['conversation_id', 'response_time']],
            how='inner', on='conversation_id'
        ).dropna(subset=['resolved', 'topic', 'total_tweets', 'response_time', 'evolution_score'])

        # Bubble chart: topics by response time and resolution rate
        ax1 = fig.add_subplot(131)
        topic_bubble = df.groupby('topic').agg({
            'response_time': 'mean',
            'resolved': 'mean',
            'conversation_id': 'count'  # for bubble size
        }).reset_index()

        scatter = ax1.scatter(topic_bubble['response_time'], topic_bubble['resolved'],
                            s=topic_bubble.index*5, alpha=0.7)
        ax1.set_xlabel('Average Response Time')
        ax1.set_ylabel('Resolution Rate')
        ax1.set_title('Topic Performance\n(Size = # Conversations)')
        ax1.set_ylim(0.0, 0.5)

        # Add topic labels
        for i, topic in enumerate(topic_bubble['topic']):
            ax1.annotate(topic, (topic_bubble['response_time'].iloc[i], 
                                topic_bubble['resolved'].iloc[i]), fontsize=8)

        # Response time distribution by topic (violin plot)
        ax2 = fig.add_subplot(132)
        topics_for_violin = df['topic'].value_counts().head(6).index  # Top 6 topics
        df_top_topics = df[df['topic'].isin(topics_for_violin)]
        sns.violinplot(data=df_top_topics, x='topic', y='response_time', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Response Time Distribution by Topic')
        ax2.set_ylim(0, 800)

        # Evolution score by topic and resolution
        ax3 = fig.add_subplot(133)
        topic_resolution_evolution = df.groupby(['topic', 'resolved'])['evolution_score'].mean().unstack()
        topic_resolution_evolution.plot(kind='bar', ax=ax3, color=self.custom_colors)
        ax3.set_title('Evolution Score by Topic and Resolution')
        ax3.set_xlabel('Topic')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(-0.6, 0.6)

        plt.tight_layout()

        self.save_plot(fig, 'three_way_visualisation')

    def plot_response_time_evolution(self):
        # Sample a few conversations to show response time patterns
        df = pd.merge(
            self.scores_df[['conversation_id', 'resolved']],
            self.df_merged[['conversation_id', 'response_time', 'tweet_index']],
            on = "conversation_id", how="inner"
        ).dropna(subset=['resolved', 'response_time', 'tweet_index'])
        sample_convos = df['conversation_id'].sample(10)
        
        fig = plt.figure(figsize=(15, 8))
        
        for i, conv_id in enumerate(sample_convos):
            conv_data = df[df['conversation_id'] == conv_id].sort_values('tweet_index')
            
            plt.subplot(2, 5, i+1)
            response_times = conv_data['response_time'].dropna()
            plt.plot(range(len(response_times)), response_times, 'o-', alpha=0.7)
            plt.title(f'Conv {i+1}\nResolved: {conv_data["resolved"].iloc[0]}')
            plt.xlabel('Tweet Index')
            plt.ylabel('Response Time')
        
        plt.tight_layout()
        plt.suptitle('Response Time Evolution in Sample Conversations', y=1.02)

        self.save_plot(fig, "response_time_evolution")

    # Response time vs sentiment interaction
    def plot_response_sentiment_analysis(self):
        fig, ax = plt.subplots(figsize=(14, 9))
        df = self.df_merged[self.df_merged['airline'] == 'KLM']

        grp = df.groupby(['sentiment_label', 'role'])['response_time'].mean().unstack()

        bar_width = 0.35
        x_pos = np.arange(len(grp.index))
        roles = grp.columns.tolist()

        for i, role in enumerate(roles):
            values = grp[role].values
            bars = ax.bar(x_pos + i * bar_width, values, 
                        width=bar_width, 
                        label=role.capitalize(),
                        color=self.custom_colors[i],
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=1.5)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(grp.max()) * 0.02,
                        f'{height:.0f}',
                        ha='center', va='bottom', 
                        fontweight='bold', fontsize=11,
                        color='#2C3E50')

        ax.set_xticks(x_pos + bar_width / 2)
        ax.set_xticklabels([label.capitalize() for label in grp.index], 
                        rotation=0, fontsize=12)

        legend = ax.legend(title='Role', 
                        frameon=True, 
                        fancybox=True, 
                        shadow=True,
                        title_fontsize=12,
                        fontsize=11,
                        loc='upper right')
        legend.get_frame().set_facecolor('#FFFFFF')
        legend.get_frame().set_alpha(0.9)

        
        self.apply_enhanced_styling(ax, "KLM: Average Response Time by Sentiment and Role", 'Sentiment', 'Response Time (minutes)')
        self.save_plot(fig, "response_sentiment_analysis")

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
        self.plot_trend_distribution()
        self.plot_conversation_length_vs_evolution_score_alternative()
        self.plot_sentiment_over_time_by_airline(aggregation='week')
        self.plot_sentiment_over_time_by_airline(aggregation='month')
        self.plot_sentiment_over_time_by_airline(aggregation='day')
        self.plot_delta_sent_against_response_time()
        self.plot_mean_initial_sentiment_over_response_time()
        self.plot_evolution_score_distribution()
        self.plot_convo_length_vs_evolution_patterns()
        self.plot_sentiment_journey()
        self.plot_comparative_analysis()    
        self.plot_score_drivers()
        self.plot_extreme_cases()
        self.plot_different_evo_score_weights()
        self.plot_resolve_rate_per_trend() 
        self.plot_average_evo_score_per_topic()
        self.plot_resolved_per_topic_and_airline()
        self.plot_resolved_per_topic_and_airline_2()
        self.plot_response_time_vs_resolution_rate()
        self.plot_convo_length_vs_resolution()
        self.plot_average_response_time_per_topic()
        self.plot_response_time_by_day()
        self.plot_three_way_visualisation()
        self.plot_response_sentiment_analysis()
        self.plot_response_time_evolution()


def main():
    vis = SentimentVisualizer(sample_size=None)  # Use None for full dataset
    vis.load_data_from_mongodb()
    vis.create_all_visualizations()

if __name__ == "__main__":
    main()
