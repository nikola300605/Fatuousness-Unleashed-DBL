import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from pymongo_interface import get_documents_batch
from processing_data import process_batch
from email.utils import parsedate_to_datetime
from scipy import stats
import time
from collections import Counter, defaultdict
from scipy.ndimage import gaussian_filter1d

PLOT_DIR = "plots/demo"
os.makedirs(PLOT_DIR, exist_ok=True)
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for fname in os.listdir(PLOT_DIR):
    if fname.endswith(".png"):
        os.remove(os.path.join(PLOT_DIR, fname))


def configure_plot_style():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.edgecolor'] = '#dee2e6'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#e9ecef'
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def apply_enhanced_styling(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.8)


def save_plot_with_enhancements(filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')


def validate_month_input(month_str):
    try:
        datetime.strptime(month_str, "%Y-%m")
        return True
    except ValueError:
        return False


def load_conversations_for_month(year_month: str):
    year, month = map(int, year_month.split("-"))
    matched_convs = []

    projection = {
        "thread.created_at": 1,
        "thread.sentiment": 1,
        "computed_metrics": 1,
        "airline": 1,
        "length": 1,
        "topic": 1,
        "resolved": 1,
        "thread.response_time": 1,
        "thread.user.screen_name": 1
    }

    for batch in get_documents_batch(collection='conversations', projection=projection):
        for convo in batch:
            thread = convo.get("thread", [])
            if not thread:
                continue
            try:
                created_at_raw = thread[0].get("created_at", "")
                created_at = parsedate_to_datetime(created_at_raw)
                if created_at.year == year and created_at.month == month:
                    matched_convs.append(convo)
            except Exception:
                continue
    return matched_convs


def flatten_conversations(conversations):
    tweets = []
    convo_metrics = []
    for conv in conversations:
        conv_id = str(conv.get('_id', ''))
        airline = conv.get('airline')
        topic = conv.get('topic')
        resolved = conv.get('resolved')
        computed = conv.get('computed_metrics', {})
        length = conv.get('length', 0)
        convo_metrics.append({
            'conversation_id': conv_id,
            'airline': airline,
            'topic': topic,
            'resolved': resolved,
            'length': length,
            'conversation_score': computed.get('conversation_score'),
            'delta_sent': computed.get('delta_sent'),
            'evolution_score': computed.get('evolution_score'),
            'start_sent': computed.get('start_sent'),
            'end_sent': computed.get('end_sent'),
            'conversation_trajectory': computed.get('conversation_trajectory'),
            'evolution_category': computed.get('evolution_category')
        })

        for i, tweet in enumerate(conv.get('thread', [])):
            sentiment = tweet.get('sentiment')
            if not sentiment:
                continue
            score = sentiment.get('score')
            label = sentiment.get('label')
            created_at = tweet.get('created_at')
            response_time = tweet.get('response_time')
            screen_name = tweet.get('user', {}).get('screen_name')
            role = 'support' if screen_name and airline and screen_name.lower() == airline.lower() else 'user'
            tweets.append({
                'conversation_id': conv_id,
                'airline': airline,
                'sentiment_label': label,
                'sentiment_score': score,
                'role': role,
                'created_at': created_at,
                'tweet_index': i,
                'response_time': response_time
            })
    return pd.DataFrame(convo_metrics), pd.DataFrame(tweets)


def plot_sentiment_distribution(df):
    if df.empty:
        print("No tweet data available for sentiment plot.")
        return
    
    # Get sentiment counts and capitalize the labels
    sentiment_counts = df["sentiment.label"].value_counts()
    
    # Create a mapping for capitalized labels
    capitalized_labels = [label.capitalize() for label in sentiment_counts.index]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = custom_colors[:len(sentiment_counts)]
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
    apply_enhanced_styling(ax, "Sentiment Distribution Analysis", 
                          "Sentiment Category", "Number of Tweets")
    file_path = os.path.join(PLOT_DIR, "sentiment_distribution.png")
    save_plot_with_enhancements(file_path)
    



def plot_evolution_score(convos):
    scores = []
    for conv in convos:
        score = conv.get("computed_metrics", {}).get("evolution_score")
        if isinstance(score, (int, float)):
            scores.append(score)
    if not scores:
        print("No evolution scores found.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(scores, bins=30, alpha=0.7, color=custom_colors[0], 
                               edgecolor='white', linewidth=1)
    for i, patch in enumerate(patches):
        alpha = 0.4 + 0.6 * (i / len(patches))
        patch.set_alpha(alpha)
    density = stats.gaussian_kde(scores)
    xs = np.linspace(min(scores), max(scores), 200)
    density_values = density(xs)
    density_scaled = density_values * len(scores) * (bins[1] - bins[0])
    ax.plot(xs, density_scaled, color=custom_colors[3], linewidth=3, 
            alpha=0.8, label='Density Curve')
    mean_score = sum(scores) / len(scores)
    median_score = np.median(scores)
    ax.axvline(mean_score, color=custom_colors[1], linestyle='--', linewidth=2.5, 
               alpha=0.9, label=f'Mean: {mean_score:.2f}')
    ax.axvline(median_score, color=custom_colors[2], linestyle=':', linewidth=2.5, 
               alpha=0.9, label=f'Median: {median_score:.2f}')
    apply_enhanced_styling(ax, "Evolution Score Distribution", 
                          "Evolution Score", "Number of Conversations")
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, edgecolor='gray')
    file_path = os.path.join(PLOT_DIR, "evolution_score_distribution.png")
    save_plot_with_enhancements(file_path)

def plot_average_evo_score_per_topic(scores_df):
    topic_mapping = {
        'delay': 'Flight Delays',
        'customer_service': 'Customer Service',
        'booking': 'Booking & Reservations',
        'luggage': 'Luggage & Baggage',
        'other': 'Other Issues',
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    df_plot = scores_df.copy()
    df_plot['topic_display'] = df_plot['topic'].map(topic_mapping).fillna(df_plot['topic'])
    unique_topics = df_plot['topic_display'].nunique()

    box_plot = sns.boxplot(
        data=df_plot, 
        x='topic_display', 
        y='evolution_score', 
        ax=ax,
        palette=custom_colors[:unique_topics],
        linewidth=2,
        boxprops=dict(alpha=0.8),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.6)
    )


    apply_enhanced_styling(ax, 'Average Evolution Score per Topic', 'Topic', 'Evolution Score')
    ax.set_ylim(-1, 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'average_evolution_score_per_topic.png'))


def plot_average_response_time_per_topic(scores_df, df_merged):
    merged = pd.merge(scores_df[['conversation_id', 'topic']],
                      df_merged[['conversation_id', 'response_time']],
                      on='conversation_id', how='inner').dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    per_topic = merged.groupby('topic')['response_time'].mean().reset_index().sort_values(by='response_time')
    unique_topics = per_topic['topic'].nunique()
    sns.barplot(data=per_topic, x='topic', y='response_time', hue='topic', palette=custom_colors[:unique_topics], ax=ax, legend=False)
    apply_enhanced_styling(ax, 'Average Response Time Per Topic', 'Topic', 'Avg Response Time [min]')
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'average_response_time_per_topic.png'))



def plot_convo_length_clipped(scores_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    clipped_lengths = scores_df['length'].dropna().clip(upper=10)
    sns.histplot(clipped_lengths, bins=10, kde=False, color=custom_colors[0], ax=ax)
    apply_enhanced_styling(ax, "Conversation Length Distribution (Clipped at 10)", "Number of Tweets", "Frequency")
    save_plot_with_enhancements(os.path.join(PLOT_DIR, "convo_length_clipped_distribution.png"))


def plot_sentiment_journey(scores_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(scores_df['start_sent'], scores_df['end_sent'], c=scores_df['evolution_score'],
                         cmap='RdYlBu', s=50, alpha=0.6)
    plt.colorbar(scatter, label='Evolution Score')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    apply_enhanced_styling(ax, 'Conversation Sentiment Trajectories', 'Start Sentiment', 'End Sentiment')
    ax.text(0.5, 0.5, 'Positive→Positive\n(Maintained)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    ax.text(-0.5, 0.5, 'Negative→Positive\n(Improved)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    ax.text(0.5, -0.5, 'Positive→Negative\n(Deteriorated)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
    ax.text(-0.5, -0.5, 'Negative→Negative\n(Persisted)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'sentiment_journey.png'))



def plot_convo_count_by_airline(scores_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    airline_counts = scores_df['airline'].value_counts()
    bars = ax.bar(range(len(airline_counts)), airline_counts.values, 
                  color=custom_colors[:len(airline_counts)], edgecolor='white', linewidth=1, alpha=0.8)
    ax.set_xlabel('Airline')
    ax.set_ylabel('Number of Conversations')
    ax.set_title('Conversation Count by Airline')
    ax.set_xticks(range(len(airline_counts)))
    ax.set_xticklabels(airline_counts.index, rotation=45, ha='right', fontsize=11)
    for i, (bar, value) in enumerate(zip(bars, airline_counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2., value + 5, f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(airline_counts.values) * 1.1)
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'convo_count_by_airline.png'))


# === Remaining Plot Functions ===
def plot_evolution_category_distribution(scores_df):
    fig, ax = plt.subplots(figsize=(14, 7), nrows=1, ncols=2)
    category_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    color_map = {
        'Very Negative': '#d32f2f',
        'Negative': '#f57c00',
        'Neutral': '#1976d2',
        'Positive': '#388e3c',
        'Very Positive': '#7b1fa2'
    }
    for idx, group in enumerate(['KLM', 'Others']):
        data = scores_df[scores_df['airline'] == 'KLM'] if group == 'KLM' else scores_df[scores_df['airline'] != 'KLM']
        counts = data['evolution_category'].value_counts()
        labels = [cat for cat in category_order if counts.get(cat, 0) > 0]
        values = [counts.get(cat, 0) for cat in labels]
        colors = [color_map[cat] for cat in labels]
        ax[idx].pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
                    pctdistance=0.85, wedgeprops=dict(width=0.7, edgecolor='white'))
        ax[idx].set_title(f"{group} Evolution Category")
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'evolution_category_distribution.png'))


def plot_conversation_trajectory_by_airline(scores_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    tab = pd.crosstab(scores_df['airline'], scores_df['conversation_trajectory'], normalize='index')
    tab = tab[['Worsening', 'Stable', 'Improving']].fillna(0)
    tab.plot(kind='bar', stacked=True, ax=ax, color=["#7B162A", "#9C9463B6", "#095C2D"], edgecolor='white', linewidth=1.5, alpha=0.85)
    apply_enhanced_styling(ax, "Conversation Trajectory by Airline", "Airline", "Proportion")
    ax.set_ylim(0, 1.0)
    ax.legend(title='Trajectory', 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            title_fontsize=12)
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'conversation_trajectory_by_airline.png'))


def plot_resolved_per_topic_and_airline(scores_df):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)

    scores_df['airline_group'] = scores_df['airline'].apply(lambda x: 'KLM' if x == 'KLM' else 'Other Airlines')

    grouped = scores_df.groupby(['airline_group', 'topic', 'resolved']).size().reset_index(name='count')
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
            palette=custom_colors,
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
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'resolved_per_topic_and_airline.png'))


def plot_response_sentiment_analysis(df_merged):
    fig, ax = plt.subplots(figsize=(14, 9))
    df = df_merged[df_merged['airline'] == 'KLM']

    grp = df.groupby(['sentiment_label', 'role'])['response_time'].mean().unstack()

    bar_width = 0.35
    x_pos = np.arange(len(grp.index))
    roles = grp.columns.tolist()

    for i, role in enumerate(roles):
        values = grp[role].values
        bars = ax.bar(x_pos + i * bar_width, values, 
                     width=bar_width, 
                     label=role.capitalize(),
                     color=custom_colors[i],
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

    
    apply_enhanced_styling(ax, "KLM: Average Response Time by Sentiment and Role", 'Sentiment', 'Response Time (minutes)')
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'response_sentiment_analysis.png'))



def plot_extreme_cases(scores_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(scores_df['conversation_score'], scores_df['delta_sent'], alpha=0.6,
                    s=scores_df['length'] * 2, c=scores_df['evolution_score'], cmap='RdYlBu')
    plt.colorbar(sc, ax=ax, label='Evolution Score')
    ax.set_xlabel('Conversation Score')
    ax.set_ylabel('Delta Sentiment')
    ax.set_title('Component Relationship (Size = Conv Length)')
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'extreme_cases.png'))


def plot_sentiment_over_time_by_airline(scores_df, df_merged):
    fig, ax = plt.subplots(figsize=(14, 8))

    merged = pd.merge(scores_df[['conversation_id', 'evolution_score']],
                      df_merged[['conversation_id', 'airline','created_at']], on='conversation_id', how="inner").dropna(subset=["airline", "evolution_score", "created_at"])
    
    print(f"Total merged data: {len(merged)}")
    print(f"Date range: {merged['created_at'].min()} to {merged['created_at'].max()}")
    print(merged[merged['airline'] == 'KLM'])
        
    merged['created_at'] = pd.to_datetime(merged['created_at'])

    merged['week'] = merged['created_at'].dt.to_period('W-MON').dt.start_time

    merged_klm = merged[merged['airline'] == 'KLM']
    merged_others = merged[merged['airline'] != 'KLM']
    print(f"KLM data: {merged_klm}")
    print(f"Other airlines data: {len(merged_others)}")

    # Group by time period and airline, calculate mean sentiment
    sentiment_over_time_klm = (merged_klm
                        .groupby(['week'])['evolution_score']
                        .mean()
                        .reset_index())
    
    sentiment_over_time_others = (merged_others
                        .groupby(['week'])['evolution_score']
                        .mean()
                        .reset_index())
    
        # Convert time_period back to datetime if it isn't already
    sentiment_over_time_klm['week'] = pd.to_datetime(sentiment_over_time_klm['week'])
    sentiment_over_time_others['week'] = pd.to_datetime(sentiment_over_time_others['week'])
    
    # Sort by date to ensure proper line plotting
    sentiment_over_time_klm = sentiment_over_time_klm.sort_values('week')
    sentiment_over_time_others = sentiment_over_time_others.sort_values('week')

    print(f"KLM weekly data points: {len(sentiment_over_time_klm)}")
    print(f"Other airlines weekly data points: {len(sentiment_over_time_others)}")
    print("KLM weekly data:")
    print(sentiment_over_time_klm)
    print("Others weekly data:")
    print(sentiment_over_time_others)
    
    ax.plot(sentiment_over_time_klm['week'], sentiment_over_time_klm['evolution_score'], 
                marker='o', linewidth=2.5, markersize=5,
                label='KLM', 
                color=custom_colors[0],
                alpha=0.8)
        
    ax.plot(sentiment_over_time_others['week'], sentiment_over_time_others['evolution_score'], 
            marker='o', linewidth=2.5, markersize=5,
            label='Other Airlines', 
            color=custom_colors[1],
            alpha=0.8)

    
    # Styling improvements
    ax.set_title(f"Weekly Mean Sentiment (Evolution Score) Over Time by Airline", 
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
    ax.tick_params(axis='x', rotation=45)
    # Show fewer ticks for daily data to avoid crowding
    ax.locator_params(axis='x', nbins=10)

    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    apply_enhanced_styling(ax, "Weekly Avg Sentiment by Airline", "Week", "Mean Evolution Score")

    
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'sentiment_over_time_weekly_by_airline.png'))

def plot_success_rate_by_airline(scores_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    success_rate = scores_df.groupby('airline')['evolution_score'].apply(lambda x: (x > 0).mean()).reset_index(name='success_rate')
    unique_airlines = success_rate['airline'].nunique()
    sns.barplot(data=success_rate, x='airline', y='success_rate', hue='airline', palette=custom_colors[:unique_airlines], ax=ax, legend=False)
    apply_enhanced_styling(ax, "Success Rate by Airline", "Airline", "Proportion of Positive Evolution")
    ax.set_ylim(0, 1.0)
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'success_rate_by_airline.png'))

def plot_avg_convo_length_by_airline(scores_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_length = scores_df.groupby('airline')['length'].mean().reset_index()
    unique_airlines = avg_length['airline'].nunique()
    sns.barplot(data=avg_length, x='airline', y='length', hue='airline', palette=custom_colors[:unique_airlines], ax=ax, legend=False)
    apply_enhanced_styling(ax, "Average Conversation Length by Airline", "Airline", "Average Length")
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'avg_convo_length_by_airline.png'))



def plot_convo_length_vs_evo_patterns(scores_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define bins and labels
    length_bins = [0, 3, 7, 12, 20, float('inf')]
    labels = ['Very Short (1-3)', 'Short (4-7)', 'Medium (8-12)', 'Long (13-20)', 'Very Long (20+)']
    
    df = scores_df.copy()
    df['length_bin'] = pd.cut(df['length'], bins=length_bins, labels=labels)

    # Boxplot of evolution_score by binned length
    bin_count = df['length_bin'].nunique()
    box_plot = sns.boxplot(
        data=df, 
        x='length_bin', 
        y='evolution_score', 
        palette=custom_colors,
        linewidth=2.5,
        boxprops=dict(alpha=0.8, linewidth=2),
        whiskerprops=dict(linewidth=2.5),
        capprops=dict(linewidth=2.5),
        medianprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7, markeredgecolor='darkred')
    )
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    apply_enhanced_styling(ax=ax, title="Evolution Score by Conversation Length", xlabel="Conversation Length Bin", ylabel="Evolution Score")

    plt.tight_layout()
    save_plot_with_enhancements(os.path.join(PLOT_DIR, 'convo_length_vs_evo_patterns.png'))

def plot_sentiment_distribution_by_role(df):
        # Create figure with larger size for better presentation
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Group by role and sentiment, then create normalized data
        tab = df[df['airline'] == "KLM"].groupby(['role', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Normalize to percentages (0-100)
        tab_normalized = tab.div(tab.sum(axis=1), axis=0) * 100

        # Define color mapping
        color_map = {
            'positive': "#095C2D",    # Green
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
        
        
        
        # Format y-axis to show percentages
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)])
        apply_enhanced_styling(ax, "Sentiment Distribution by Role for KLM", "Role", "Percentage (%)")
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
        save_plot_with_enhancements(os.path.join(PLOT_DIR, 'sentiment_distribution_by_role.png'))
        plt.tight_layout()


# Updated main function with all plots

def main():
    configure_plot_style()
    month = input("Enter month (YYYY-MM): ").strip()
    if not validate_month_input(month):
        print("Invalid format. Please enter in YYYY-MM format.")
        return
    
    conversations = load_conversations_for_month(month)
    if not conversations:
        print("No conversations found for that month.")
        return
    scores_df, df_merged = flatten_conversations(conversations)
    tweets_df = pd.json_normalize([tweet for conv in conversations for tweet in conv.get("thread", []) if "sentiment" in tweet])
    print(f"\nSummary for {month}:")
    print(f" - Conversations found: {len(conversations)}")
    print(f" - Tweets with sentiment: {len(tweets_df)}")

    plot_sentiment_distribution(tweets_df)
    plot_evolution_score(conversations)
    plot_average_evo_score_per_topic(scores_df)
    plot_average_response_time_per_topic(scores_df, df_merged)
    #plot_convo_length_clipped(scores_df)
    plot_sentiment_journey(scores_df)
    plot_success_rate_by_airline(scores_df)
    plot_avg_convo_length_by_airline(scores_df)
    plot_convo_count_by_airline(scores_df)
    plot_convo_length_vs_evo_patterns(scores_df)
    plot_evolution_category_distribution(scores_df)
    plot_conversation_trajectory_by_airline(scores_df)
    plot_resolved_per_topic_and_airline(scores_df)
    plot_response_sentiment_analysis(df_merged)
    plot_extreme_cases(scores_df)
    plot_sentiment_over_time_by_airline(scores_df, df_merged)
    plot_sentiment_distribution_by_role(df_merged)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    duration_minutes = (end - start) / 60
    print(f"\nTotal execution time: {duration_minutes:.2f} minutes")


