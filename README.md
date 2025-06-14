# Airline Customer Support Sentiment Analysis via Twitter
**Group 18 – Fatuousness Unleashed**

---

This project evaluates the quality of airline customer support, specifically focusing on KLM, by analyzing Twitter conversations. It uses data mining, natural language processing, and sentiment modeling to assess user satisfaction, support efficiency, and conversation outcomes.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)  
2. [Setup & Installation](#setup--installation)  
3. [Environment Configuration](#environment-configuration)
4. [File Descriptions](#file-descriptions)  
   - [Data Cleaning & Loading](#41-data-cleaning--loading)  
   - [Conversation Mining & Sentiment Analysis](#42-conversation-mining--sentiment-analysis)  
   - [Metric Computation](#43-metric-computation)  
   - [Data Exploration & Visualization](#44-data-exploration--visualization)  
   - [Statistical Evaluation](#45-statistical-evaluation)  
5. [Execution Workflow](#execution-workflow)  
6. [Output Structure](#output-structure)  
7. [Methodology Summary](#methodology-summary)  
8. [Notes](#notes)

---

## 1. Project Overview

The goal is to assess airline customer support—especially KLM—by mining Twitter threads and analyzing sentiment shifts within them. The system constructs conversation threads between users and airlines, classifies sentiment using pre-trained NLP models, and evaluates support effectiveness using various metrics such as resolution likelihood and trajectory.

---

## 2. Setup & Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/airline-support-analysis.git
cd airline-support-analysis
```

(Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Create a `.env` file with your MongoDB connection string:
```
DATABASE_URL=mongodb://localhost:27017
```

---

## 4. File Descriptions

### 4.1 Data Cleaning & Loading

- `clean_json.py`: Cleans and validates tweet JSON files.  
  **Output:** Cleaned tweets → `./cleaned_tweets_json/`

- `load_initial_data.py`: Loads cleaned JSON into MongoDB (`tweets_try` collection).  
  **Usage:** `python load_initial_data.py`

- `processing_data.py`: Helper functions to normalize tweet structure.  
  **Usage:** Imported into multiple scripts.

### 4.2 Conversation Mining & Sentiment Analysis

- `insert_convos.py`: Extracts conversations, assigns sentiment scores, and saves to `conversations`.  
  **Output:** MongoDB `conversations` collection.

- `sentiment_analysis.py`: Uses `cardiffnlp/twitter-xlm-roberta-base-sentiment`.  
  **Output:** Adds `sentiment.label` and `sentiment.score` to each tweet.

- `evaluation_model.py`: Evaluates sentiment model accuracy using labeled datasets.  
  **Output:** Classification report and accuracy score.

### 4.3 Metric Computation

- `sentiment_evolution.py`: Computes:
  - `conversation_score`
  - `delta_sent`
  - `start_sent`, `end_sent`
  - `conversation_trajectory`  
  **Output:** Updates MongoDB conversation documents.

- `mark_resolved.py`: Flags resolved conversations based on improvement.  
  **Output:** `conversation_data_resolved.json`

- `add_topics_labels.py`: Assigns topics to conversations using regex-based keyword matching.  
  **Output:** `conversation_data_with_topics.json`

### 4.4 Data Exploration & Visualization

- `convo_eda.py`: Computes summary stats and plots sentiment, response time, and counts.  
  **Output:** Plots saved to `./plots/`

- `sentiment_visualizer.py`: Generates advanced visualizations (score distribution, response impact, role-based trends).  
  **Output:** Plots saved to `./plots/sentiment_evo/`

- `eda.py`: Language distribution, tweet volume over time, and airline mentions.

- `extract_first_tweets.py`: Exports structured conversation data and first tweets.  
  **Output:** `conversation_data_cleaned.json`

### 4.5 Statistical Evaluation

- `hypothesis_testing.py`: Runs chi-square test on topic vs. resolution likelihood.  
  **Output:** Console report (significance, p-value).

- `pymongo_interface.py`: MongoDB read/write batch functions.  
  **Used by:** All processing and sentiment scripts.

---

## 5. Execution Workflow

```bash
# Step 1: Clean raw tweet data
python clean_json.py

# Step 2: Load cleaned data into MongoDB
python load_initial_data.py

# Step 3: Extract conversations and apply sentiment analysis
python insert_convos.py

# Step 4: Compute scoring and metrics
python sentiment_evolution.py

# Step 5: Add topic labels and resolution flags
python add_topics_labels.py
python mark_resolved.py

# Step 6: Perform hypothesis testing
python hypothesis_testing.py

# Step 7: Generate visualizations
python sentiment_visualizer.py
```

---

## 6. Output Structure

### MongoDB Collections
- `tweets_try`: Individual tweet records
- `conversations`: Annotated and analyzed conversation threads

### Local Files
- `conversation_data_cleaned.json`: Export of processed conversation threads
- `conversation_data_with_topics.json`: Topic-labeled conversation threads
- `conversation_data_resolved.json`: Resolved vs unresolved flag

### Visualizations
- `/plots/`: Summary charts
- `/plots/sentiment_evo/`: Role, airline, trajectory, and score visualizations

---

## 7. Methodology Summary

- **Cleaning:** Remove corrupted and malformed tweet entries.
- **Loading:** Insert valid tweets into MongoDB.
- **Thread Mining:** Reconstruct full user–airline conversations using reply chains.
- **Sentiment Analysis:** Classify sentiment (positive, neutral, negative) using XLM-RoBERTa.
- **Scoring:** Apply role-based and positional weights to derive conversation-level sentiment metrics.
- **Labeling:** Detect common complaint topics via regex. Determine resolution status based on sentiment delta.
- **Visualization:** Track trends, trajectory, sentiment shifts, and airline comparison metrics.
- **Statistics:** Run chi-square test to identify if topic affects resolution likelihood.

---

## 8. Notes

- Ensure MongoDB is running and accessible via the `.env` `DATABASE_URL`.
- Some scripts use multiprocessing—check CPU core allocation.
- Designed for multilingual tweet data, but results focus on English and KLM-related conversations.
- Outputs can be used for additional research, dashboards, or presentations.

---
