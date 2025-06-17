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

PLOT_DIR = "plots/demo"
os.makedirs(PLOT_DIR, exist_ok=True)
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

# Clean old plots for demo clarity
for fname in os.listdir(PLOT_DIR):
    if fname.endswith(".png"):
        os.remove(os.path.join(PLOT_DIR, fname))

#
# Visual Enhancements
# 

def configure_plot_style():
    """Configure global matplotlib and seaborn styling"""
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
    """Apply consistent enhanced styling to any plot axis"""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add subtle shadow effect to spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.8)

def save_plot_with_enhancements(filepath):
    """Save plot with high quality settings"""
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

# 
# Data Processing Section
# 

def validate_month_input(month_str):
    """Validate month format"""
    try:
        datetime.strptime(month_str, "%Y-%m")
        return True
    except ValueError:
        return False

def load_conversations_for_month(year_month: str):
    """Load conversations for a specific month"""
    year, month = map(int, year_month.split("-"))
    matched_convs = []

    projection = {
        "thread.created_at": 1,
        "thread.sentiment": 1,
        "computed_metrics": 1,
        "airline": 1,
        "length": 1
    }

    print(f"\nSearching for conversations from {year_month}...")
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
                    if len(matched_convs) % 100 == 0:
                        print(f"  → {len(matched_convs)} matched so far...", end="\r")
            except Exception as e:
                print(f"Skipping convo due to date parse error: {e}")
    print(f"\nFound {len(matched_convs)} conversations.")
    return matched_convs

# 
# Plotting Functions
# 

def plot_sentiment_distribution(df):
    """Plot sentiment distribution with enhanced styling"""
    if df.empty:
        print("No tweet data available for sentiment plot.")
        return

    # Core functionality: prepare data
    sentiment_counts = df["sentiment.label"].value_counts()
    
    # Enhanced visuals: create styled plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = custom_colors[:len(sentiment_counts)]
    
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(sentiment_counts.values) * 0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Apply styling
    apply_enhanced_styling(ax, "Sentiment Distribution Analysis", 
                          "Sentiment Category", "Number of Tweets")
    
    # Save and show
    file_path = os.path.join(PLOT_DIR, "sentiment_distribution.png")
    save_plot_with_enhancements(file_path)
    plt.show()

def plot_conversation_score(convos):
    """Plot conversation score distribution with enhanced styling"""
    # Core functionality: extract scores
    scores = []
    for conv in convos:
        score = conv.get("computed_metrics", {}).get("conversation_score")
        if isinstance(score, (int, float)):
            scores.append(score)

    if not scores:
        print("No conversation scores found.")
        return

    # Enhanced visuals: create styled plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with gradient effect
    n, bins, patches = ax.hist(scores, bins=30, alpha=0.7, color=custom_colors[0], 
                               edgecolor='white', linewidth=1)
    
    # Apply gradient coloring to bars
    for i, patch in enumerate(patches):
        alpha = 0.4 + 0.6 * (i / len(patches))
        patch.set_alpha(alpha)
    
    # Add KDE curve
    density = stats.gaussian_kde(scores)
    xs = np.linspace(min(scores), max(scores), 200)
    density_values = density(xs)
    density_scaled = density_values * len(scores) * (bins[1] - bins[0])
    ax.plot(xs, density_scaled, color=custom_colors[3], linewidth=3, 
            alpha=0.8, label='Density Curve')
    
    # Add statistical lines
    mean_score = sum(scores) / len(scores)
    median_score = np.median(scores)
    
    ax.axvline(mean_score, color=custom_colors[1], linestyle='--', linewidth=2.5, 
               alpha=0.9, label=f'Mean: {mean_score:.2f}')
    ax.axvline(median_score, color=custom_colors[2], linestyle=':', linewidth=2.5, 
               alpha=0.9, label=f'Median: {median_score:.2f}')
    
    # Apply styling and legend
    apply_enhanced_styling(ax, "Conversation Score Distribution", 
                          "Conversation Score", "Number of Conversations")
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, edgecolor='gray')
    
    # Save and show
    file_path = os.path.join(PLOT_DIR, "conversation_score_distribution.png")
    save_plot_with_enhancements(file_path)
    plt.show()

# 
# Main Functionality
# 

def main():
    # Initialize styling
    configure_plot_style()
    
    # Core functionality
    month = input("Enter month (YYYY-MM): ").strip()
    if not validate_month_input(month):
        print("Invalid format. Please enter in YYYY-MM format.")
        return

    conversations = load_conversations_for_month(month)
    if not conversations:
        print("No conversations found for that month.")
        return

    tweets = [tweet for conv in conversations for tweet in conv.get("thread", []) if "sentiment" in tweet]
    df = pd.json_normalize(tweets)

    print(f"\nSummary for {month}:")
    print(f" - Conversations found: {len(conversations)}")
    print(f" - Tweets with sentiment: {len(df)}")

    # Generate plots
    plot_sentiment_distribution(df)
    plot_conversation_score(conversations)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    duration_minutes = (end - start) / 60
    print(f"\nTotal execution time: {duration_minutes:.2f} minutes")