import pandas as pd
from bson import ObjectId
from pymongo_interface import get_documents_batch
from processing_data import process_batch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import evaluate
import matplotlib.pyplot as plt
import numpy as np


def flatten_to_df(documents):
    flattened_rows = []

    for doc in documents:
        base_id = doc['_id']
        evolution_category = doc.get('computed_metrics', {}).get('evolution_category', None)

        for tweet in doc.get('thread', []):
            tweet_flat = {
                "tweet_id": tweet.get('id', None),
                "label": tweet.get('sentiment', {}).get('label', None),
                "parent_doc_id": base_id,
                "evolution_category": evolution_category
            }

            flattened_rows.append(tweet_flat)

    return pd.DataFrame(flattened_rows)

def test_model_without_running(data_csv):
    """ 
    Testing the model accuracy without running the model.
    This function reads a CSV file which is self-labeled tweets and conversation.
    It queries the MongoDB for the conversations and merges the data.
    It then compares the true sentiment with the predicted sentiment and the true evolution category with the predicted evolution category.

    """

    df = pd.read_csv(data_csv)
    df.columns = ["tweet_id", "true_tweet_sentiment", "conversation_id", "true_evolution_category"]
    tweet_ids = df['tweet_id'].tolist()
    conversation_ids = df['conversation_id'].tolist()

    object_ids = [ObjectId(cid) for cid in conversation_ids]
    
    query = {"_id": {"$in": object_ids}}
    projection = {
        '_id': True,
        'thread': True,
        'computed_metrics.evolution_category': True
    }

    cursor = get_documents_batch(query=query, projection = projection, collection='conversations')

    querried_data_convo = pd.DataFrame()
    querried_data_tweet = pd.DataFrame()

    for batch in cursor:
        df_batch = process_batch(batch)
        if querried_data_convo.empty:
            querried_data_convo = df_batch
        else:
            querried_data_convo = pd.concat([querried_data_convo, df_batch], ignore_index=True)
        
        if querried_data_tweet.empty:
            querried_data_tweet = flatten_to_df(batch)
        else:
            querried_data_tweet = pd.concat([querried_data_tweet, flatten_to_df(batch)], ignore_index=True)
        
    querried_data_convo = querried_data_convo.rename(columns={'_id.$oid': 'conversation_id'})
    merged_convo_df = pd.merge(df, querried_data_convo, on='conversation_id', how='left')

    querried_data_tweet = querried_data_tweet.rename(columns={'parent_doc_id': 'conversation_id'})
    print(querried_data_tweet)
    merged_tweet_df = pd.merge(df, querried_data_tweet,on='tweet_id', how='left')
    
    merged_convo_df['true_evolution_category'] = merged_convo_df['true_evolution_category'].str.title()
    merged_tweet_df['true_tweet_sentiment'] = merged_tweet_df['true_tweet_sentiment'].str.title()
    merged_tweet_df['label'] = merged_tweet_df['label'].str.title()

    y_true = merged_tweet_df['true_tweet_sentiment'].tolist()
    y_pred = merged_tweet_df['label'].tolist()

    predicted_evolution_category = merged_convo_df['computed_metrics.evolution_category'].tolist()
    true_evolution_category = merged_convo_df['true_evolution_category'].tolist()


    custom_classes_1 = ['Negative', 'Neutral', 'Positive']
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(custom_classes_1)
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    custom_classes_2 = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    label_encoder.classes_ = np.array(custom_classes_2)
    true_evolution_category_encoded = label_encoder.transform(true_evolution_category)
    predicted_evolution_category_encoded = label_encoder.transform(predicted_evolution_category)


    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    metrics_sentiment_label = {
    'accuracy': accuracy_metric.compute(predictions=y_pred_encoded, references=y_true_encoded)['accuracy'],
    'f1': f1_metric.compute(predictions=y_pred_encoded, references=y_true_encoded, average='macro')['f1'],
    'precision': precision_metric.compute(predictions=y_pred_encoded, references=y_true_encoded, average='macro')['precision'],
    'recall': recall_metric.compute(predictions=y_pred_encoded, references=y_true_encoded, average='macro')['recall'],
    }

    metrics_conversation_category = {
        'accuracy': accuracy_metric.compute(predictions=predicted_evolution_category_encoded, references=true_evolution_category_encoded)['accuracy'],
        'f1': f1_metric.compute(predictions=predicted_evolution_category_encoded, references=true_evolution_category_encoded, average='macro')['f1'],
        'precision': precision_metric.compute(predictions=predicted_evolution_category_encoded, references=true_evolution_category_encoded, average='macro')['precision'],
        'recall': recall_metric.compute(predictions=predicted_evolution_category_encoded, references=true_evolution_category_encoded, average='macro')['recall'],
    }

    fig,ax = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)
    cm_1 = confusion_matrix(y_true, y_pred, labels=['Negative', 'Neutral', 'Positive'])
    disp_1 = ConfusionMatrixDisplay(cm_1, display_labels=['Negative', 'Neutral', 'Positive'])
    disp_1.plot(cmap='Blues', ax=ax[0])
    fig.suptitle('Sentiment Analysis Confusion Matrix', fontsize=16)
    ax[0].set_title('Confusion Matrix for tweet sentiment')

    cm_2 = confusion_matrix(true_evolution_category, predicted_evolution_category, labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
    disp_2 = ConfusionMatrixDisplay(cm_2, display_labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
    disp_2.plot(cmap='Blues', ax=ax[1])
    ax[1].set_title('Confusion Matrix for conversation evolution category')

    print("Metrics for tweet sentiment:" , metrics_sentiment_label)
    print("Metrics for conversation evolution category:", metrics_conversation_category)
    plt.show()

if __name__ == "__main__":
    data_csv = 'tweet_evals.csv'  
    test_model_without_running(data_csv) 