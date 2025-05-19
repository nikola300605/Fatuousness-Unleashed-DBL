from processing_data import mine_conversations
from pymongo_interface import save_conversations_to_mongo
from tqdm import tqdm
from sentiment_analysis import get_sentiment
import time

def main():
    for conv_batch in tqdm(mine_conversations(), desc="Processing conversation batches"):
        sent_batch = get_sentiment(conv_batch)
        save_conversations_to_mongo(sent_batch, 'conversations')

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")