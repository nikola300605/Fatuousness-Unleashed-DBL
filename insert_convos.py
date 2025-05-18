from processing_data import mine_conversations
from pymongo_interface import save_conversations_to_mongo
from tqdm import tqdm
from sentiment_analysis import get_sentiment

for conv_batch in tqdm(mine_conversations(), desc="Processing conversation batches"):

    sent_batch = get_sentiment(conv_batch)
    save_conversations_to_mongo(sent_batch, 'conversations')