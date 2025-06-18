# Airline Customer Support Sentiment Analysis via Twitter
**Group 18 – Fatuousness Unleashed**

---

This project evaluates the quality of airline customer support by analyzing Twitter conversations, with a primary focus on KLM and comparisons to other airlines. It uses data mining, natural language processing, and sentiment modeling to assess user satisfaction, support efficiency, and conversation outcomes.

---

## 📂 Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Setup & Installation](#2-setup--installation)  
3. [Environment Configuration](#3-environment-configuration)  
4. [File Descriptions](#4-file-descriptions)  
   - [Data Cleaning & Loading](#41-data-cleaning--loading)  
   - [Conversation Mining & Sentiment Analysis](#42-conversation-mining--sentiment-analysis)  
   - [Metric Computation](#43-metric-computation)  
   - [Data Exploration & Visualization](#44-data-exploration--visualization)  
   - [Evaluation & Demonstration](#45-evaluation--demonstration)  
5. [Execution Workflow](#5-execution-workflow)  
6. [Output Structure](#6-output-structure)  
7. [Methodology Summary](#7-methodology-summary)  
8. [Notes](#8-notes)

---

## 1. Project Overview

The goal is to assess airline customer support on Twitter by mining user–airline conversations, classifying sentiment using pre-trained NLP models, and evaluating support effectiveness. Core metrics include sentiment shift, response time, and resolution trajectory.

---

## 2. Setup & Installation

```bash
git clone https://github.com/nikola300605/Fatuousness-Unleashed-DBL.git
cd airline-support-analysis
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Create a `.env` file at the root of the project:
```dotenv
DATABASE_URL=mongodb://localhost:27017
```
Ensure MongoDB is running locally or update the URL appropriately.

---

## 4. File Descriptions

### 4.1 Data Cleaning & Loading

- `clean_json.py`: Validates and filters raw tweet JSON lines.  
  **Output:** `./cleaned_tweets_json/`

- `load_initial_data.py`: Loads cleaned tweets into MongoDB (`tweets_try`).  
  **Usage:** `python load_initial_data.py`

- `processing_data.py`: Extracts and normalizes tweet threads.  
  **Used in:** `insert_convos.py`, `eda.py`, and more.

- `pymongo_interface.py`: MongoDB interaction layer with batch utilities.

### 4.2 Conversation Mining & Sentiment Analysis

- `insert_convos.py`: Reconstructs conversations and applies sentiment analysis using XLM-RoBERTa.  
  **Output:** MongoDB `conversations` collection.

- `sentiment_analysis.py`: Assigns sentiment labels and scores to tweets in conversation threads.

- `evaluation_model.py`: Benchmarks a multilingual sentiment model on a public dataset.  
  **Output:** Accuracy + classification report.

### 4.3 Metric Computation

- `sentiment_evolution.py`: Computes advanced metrics for each conversation:  
  - `conversation_score` (weighted sentiment)
  - `delta_sent` (start to end sentiment)
  - `evolution_score`, `evolution_category`, `conversation_trajectory`  
  **Usage:** `python sentiment_evolution.py`

### 4.4 Data Exploration & Visualization

- `convo_eda.py`: Generates statistical summaries and plots:
  - Response times, daily sentiment, length distributions, airline comparisons.  
  **Output:** Saved to `./plots/`

- `eda.py`: General tweet-level EDA. Language distribution, mention frequency, volume trends.

- `sentiment_visualizer.py`: Creates advanced visualizations for:
  - Role-based trends
  - Sentiment trajectories
  - Airline-level and topic-based comparisons  
  **Output:** `./plots/sentiment_evo/`

- `demo_monthly_summary.py`: Interactive monthly analysis tool. 
  - Prompts for a date (e.g., `2022-06`) and generates focused reports.  
  **Usage:** `python demo_monthly_summary.py`

### 4.5 Evaluation & Demonstration

- `model_eval.py`: Compares predicted labels in the database to manually labeled CSV for accuracy and confusion matrices.  
  **Input:** `tweet_evals.csv`  
  **Usage:** `python model_eval.py`

---


### 4.6 Business Case Analysis & Topic Modeling

- `extract_first_tweets.py`: Extracts the first tweet from each conversation and compiles relevant metadata for downstream classification.  
  **Output:** `conversation_data_cleaned.json`

- `add_topics_labels.py`: Uses keyword-based rules to classify each conversation into topics like `delay`, `luggage`, `booking`, and `customer_service`.  
  **Input:** `conversation_data_cleaned.json`  
  **Output:** `conversation_data_with_topics.json`

- `topic_classification.py`: Performs zero-shot topic classification using Facebook's BART model.  
  **Output:** `conversation_data_with_topics_zeroshot.json`

- `hybrid.py`: Combines regex and zero-shot methods conservatively to improve accuracy.  
  **Output:** `conversation_data_hybrid_conservative.json`

- `mark_resolved.py`: Flags conversations as resolved or not based on sentiment evolution.  
  **Input:** `conversation_data_hybrid_conservative.json`  
  **Output:** `conversation_data_resolved.json`

- `save_to_db.py`: Pushes `topic` and `resolved` flags back into the MongoDB `conversations` collection.

- `hypothesis_testing.py`: Runs chi-square tests to determine if topics significantly affect resolution likelihood.  
  **Input:** `conversation_data_resolved.json`

## 5. Execution Workflow

```bash
# Step 1: Clean raw tweet data
python clean_json.py

# Step 2: Load tweets into MongoDB
python load_initial_data.py

# Step 3: Extract conversations and apply sentiment
python insert_convos.py

# Step 4: Generate visual analytics on conversations
python convo_eda.py

# Step 5: Compute sentiment metrics and classifications
python sentiment_evolution.py

# Step 6: Generate visual analytics
python sentiment_visualizer.py

# Step 7: Run monthly showcase demo (optional)
python demo_monthly_summary.py

# Step 8: Evaluate using labeled ground truth (optional)
python model_eval.py

# --- Business Case Analysis ---

# Step 9: Extract first tweets from conversations
python extract_first_tweets.py

# Step 10: Assign conversation topics
# (Choose one method below)
python add_topics_labels.py              # Regex-based
python topic_classification.py           # Zero-shot BART
python hybrid.py                         # Combined conservative (quite slower)

# Step 11: Mark resolution status of each conversation
python mark_resolved.py

# Step 12: Save resolved status and topics to MongoDB
python save_to_db.py

# Step 13: Perform hypothesis testing on resolution likelihood by topic
python hypothesis_testing.py

```

---

## 6. Output Structure

### MongoDB Collections
- `tweets_try`: All raw and cleaned tweets.
- `conversations`: Annotated threads with sentiment and metrics.

### Local Outputs
- `./cleaned_tweets_json/`: Filtered and valid tweets.
- `./plots/`: Summary and analytics charts.
- `./plots/sentiment_evo/`: Advanced sentiment trajectory and comparative plots.

---

## 7. Methodology Summary

- **Cleaning:** Remove malformed tweets and irrelevant fields.
- **Thread Mining:** Reconstruct reply chains between users and airline accounts.
- **Sentiment Classification:** Assign tweet sentiment with multilingual transformer models.
- **Metrics:** Calculate aggregate sentiment shifts and classify outcomes.
- **Storage:** Store enriched conversations in MongoDB.
- **Visualization:** Analyze per airline/topic trends, response patterns, sentiment evolution.
- **Evaluation:** Benchmark models using labeled data and confusion matrices.

---

## 8. Notes

- Make sure MongoDB is accessible and running before any processing.
- Set up your `.env` file with the correct `DATABASE_URL`.
- Some scripts use `tqdm` or multiprocessing—console output may vary.
- Supports multilingual tweets but KLM/English is the main focus.

---
